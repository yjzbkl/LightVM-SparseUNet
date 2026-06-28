import torch
from torch import nn
import torch.nn.functional as F


class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target = target.flatten(1)
        intersection = probs * target
        dice = (2 * intersection.sum(dim=1) + self.smooth) / (
            probs.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class BCEDiceLossWithLogits(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossWithLogits()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(
            logits, target
        )


def build_loss(cfg):
    name = cfg.get("name", "bce_dice_logits")
    if name == "bce_dice_logits":
        return BCEDiceLossWithLogits(
            bce_weight=float(cfg.get("bce_weight", 1.0)),
            dice_weight=float(cfg.get("dice_weight", 1.0)),
        )
    if name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    if name == "dice_logits":
        return DiceLossWithLogits()
    raise ValueError(f"unsupported loss: {name}")
