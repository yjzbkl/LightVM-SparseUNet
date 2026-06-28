import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from losses import build_loss
from metrics import binary_segmentation_metrics
from models import LightVMSparseUNet
from utils import load_checkpoint, load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    _, val_dataset = build_datasets(cfg)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = LightVMSparseUNet.from_config(cfg["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    criterion = build_loss(cfg["loss"])
    threshold = float(cfg["training"].get("threshold", 0.5))

    totals = {}
    loss_total = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss_total += float(criterion(logits, masks).item())
        metrics = binary_segmentation_metrics(logits, masks, threshold=threshold)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    count = max(1, len(loader))
    print({"loss": loss_total / count, **{k: v / count for k, v in totals.items()}})


if __name__ == "__main__":
    main()
