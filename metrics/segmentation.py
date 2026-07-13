import math
from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return 1 for two equally empty sets, as commonly done in segmentation."""
    return numerator / denominator if denominator > 0 else 1.0


def _hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Hausdorff distance between binary boundaries, in pixels."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.hypot(*pred.shape[-2:]))

    pred_surface = pred ^ binary_erosion(pred, border_value=0)
    target_surface = target ^ binary_erosion(target, border_value=0)
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_pred = distance_transform_edt(~pred_surface)
    return float(
        max(
            distance_to_target[pred_surface].max(initial=0.0),
            distance_to_pred[target_surface].max(initial=0.0),
        )
    )


@torch.no_grad()
def binary_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1.0e-7,
    compute_hd: bool = True,
) -> Dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).to(torch.float32)
    target = (target >= 0.5).to(torch.float32)
    del eps  # Kept in the public signature for backward compatibility.
    per_image = []
    for pred_i, target_i in zip(pred, target):
        pred_flat = pred_i.flatten()
        target_flat = target_i.flatten()
        tp = float(torch.sum(pred_flat * target_flat).item())
        tn = float(torch.sum((1 - pred_flat) * (1 - target_flat)).item())
        fp = float(torch.sum(pred_flat * (1 - target_flat)).item())
        fn = float(torch.sum((1 - pred_flat) * target_flat).item())
        foreground_iou = _safe_ratio(tp, tp + fp + fn)
        background_iou = _safe_ratio(tn, tn + fp + fn)
        # Binary mIoU = 1/2 * [TP/(TP+FP+FN) + TN/(TN+FP+FN)].
        metrics = {
            "iou": foreground_iou,
            "miou": (foreground_iou + background_iou) / 2.0,
            "dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
            "acc": _safe_ratio(tp + tn, tp + tn + fp + fn),
            "sensitivity": _safe_ratio(tp, tp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
        }
        if compute_hd:
            metrics["hd"] = _hausdorff_distance(
                pred_i.squeeze().detach().cpu().numpy(),
                target_i.squeeze().detach().cpu().numpy(),
            )
        per_image.append(metrics)
    return {
        key: sum(item[key] for item in per_image) / max(1, len(per_image))
        for key in per_image[0]
    } if per_image else {}
