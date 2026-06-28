import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from losses import build_loss
from metrics import binary_segmentation_metrics
from models import LightVMSparseUNet
from utils import load_config, save_checkpoint, set_seed


def build_optimizer(cfg, model):
    train_cfg = cfg["training"]
    if train_cfg["optimizer"] != "Adam":
        raise ValueError("The paper setting uses Adam; set optimizer: Adam in YAML.")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def build_scheduler(cfg, optimizer):
    train_cfg = cfg["training"]
    name = train_cfg["scheduler"]
    params = train_cfg.get("scheduler_params", {})
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    if name == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    raise ValueError(f"unsupported scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, amp=False):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    metric_sum = {}
    seen = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        batch = images.size(0)
        total_loss += float(loss.item()) * batch
        metrics = binary_segmentation_metrics(logits, masks, threshold=threshold)
        for key, value in metrics.items():
            metric_sum[key] = metric_sum.get(key, 0.0) + value * batch
        seen += batch
    metrics = {key: value / max(1, seen) for key, value in metric_sum.items()}
    metrics["loss"] = total_loss / max(1, seen)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = torch.device(args.device)
    out_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )

    model = LightVMSparseUNet.from_config(cfg["model"]).to(device)
    criterion = build_loss(cfg["loss"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    start_epoch = 1
    best_iou = -1.0
    if args.resume:
        from utils import load_checkpoint

        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=str(device))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_iou = float(ckpt.get("metrics", {}).get("iou", -1.0))

    amp = bool(cfg["training"].get("amp", False)) and device.type == "cuda"
    threshold = float(cfg["training"].get("threshold", 0.5))
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, amp=amp)
        scheduler.step()
        metrics = evaluate(model, val_loader, criterion, device, threshold)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={metrics['loss']:.5f} "
            f"iou={metrics['iou']:.5f} dice={metrics['dice']:.5f}"
        )
        save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, scheduler, epoch, metrics)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scheduler, epoch, metrics)


if __name__ == "__main__":
    main()
