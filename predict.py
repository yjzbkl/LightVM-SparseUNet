import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models import LightVMSparseUNet
from utils import load_checkpoint, load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = LightVMSparseUNet.from_config(cfg["model"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_size = tuple(cfg["dataset"].get("input_size", (224, 224)))
    threshold = float(cfg["training"].get("threshold", 0.5))

    paths = sorted(Path(args.input).glob("*")) if Path(args.input).is_dir() else [Path(args.input)]
    for path in paths:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        image = Image.open(path).convert("RGB")
        orig_size = image.size[::-1]
        tensor = TF.to_tensor(TF.resize(image, input_size, InterpolationMode.BILINEAR)).unsqueeze(0)
        logits = model(tensor.to(device))
        prob = torch.sigmoid(logits)[0, 0].cpu()
        prob = TF.resize(prob.unsqueeze(0), orig_size, InterpolationMode.BILINEAR).squeeze(0)
        mask = (prob >= threshold).to(torch.uint8) * 255
        Image.fromarray(mask.numpy()).save(output_dir / f"{path.stem}_mask.png")


if __name__ == "__main__":
    main()
