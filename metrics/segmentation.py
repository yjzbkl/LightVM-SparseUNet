from typing import Dict

import torch


@torch.no_grad()
def binary_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1.0e-7,
) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).to(torch.float32)
    target = (target >= 0.5).to(torch.float32)
    pred = pred.flatten()
    target = target.flatten()
    tp = torch.sum(pred * target)
    tn = torch.sum((1 - pred) * (1 - target))
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "iou": float(iou.item()),
        "dice": float(dice.item()),
        "acc": float(acc.item()),
        "sensitivity": float(sensitivity.item()),
        "specificity": float(specificity.item()),
    }
