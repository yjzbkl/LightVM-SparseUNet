import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(path: Path) -> List[Path]:
    return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def _match_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = _list_images(image_dir)
    masks = _list_images(mask_dir)
    masks_by_stem = {m.stem.lower(): m for m in masks}
    pairs = []
    for img in images:
        stem = img.stem.lower()
        mask = masks_by_stem.get(stem)
        if mask is None:
            candidates = [m for m in masks if m.stem.lower().startswith(stem)]
            if candidates:
                mask = candidates[0]
        if mask is not None:
            pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs found in {image_dir} and {mask_dir}")
    return pairs


def _write_txt_split(path: Path, pairs: Sequence[Tuple[Path, Path]], image_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for image_path, _ in pairs:
            f.write(str(image_path.relative_to(image_dir)).replace("\\", "/") + "\n")


def _read_txt_split(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_mmotu_splits(
    root: Path,
    image_dir: str,
    mask_dir: str,
    train_split_file: str,
    val_split_file: str,
    train_size: int,
    val_size: int,
) -> None:
    train_path = root / train_split_file
    val_path = root / val_split_file
    if train_path.exists() and val_path.exists():
        return
    pairs = _match_pairs(root / image_dir, root / mask_dir)
    expected = train_size + val_size
    if len(pairs) < expected:
        raise ValueError(f"MMOTU split expects at least {expected} pairs, found {len(pairs)}")
    _write_txt_split(train_path, pairs[:train_size], root / image_dir)
    _write_txt_split(val_path, pairs[train_size : train_size + val_size], root / image_dir)


def ensure_isic_split(
    root: Path,
    image_dir: str,
    mask_dir: str,
    split_file: str,
    split_ratio: float,
    seed: int,
) -> None:
    split_path = root / split_file
    if split_path.exists():
        return
    pairs = _match_pairs(root / image_dir, root / mask_dir)
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    train_count = int(round(len(indices) * split_ratio))
    payload = {
        "seed": seed,
        "train": [
            str(pairs[i][0].relative_to(root / image_dir)).replace("\\", "/")
            for i in indices[:train_count]
        ],
        "val": [
            str(pairs[i][0].relative_to(root / image_dir)).replace("\\", "/")
            for i in indices[train_count:]
        ],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class PairedSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_dir: str,
        mask_dir: str,
        split_file: Optional[str],
        split_key: str,
        input_size: Sequence[int] = (224, 224),
        train: bool = True,
        augmentation: Optional[Dict[str, object]] = None,
    ) -> None:
        self.root = Path(root)
        if not str(root):
            raise ValueError("dataset.root is empty; set it in the YAML config")
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.input_size = tuple(input_size)
        self.train = train
        self.augmentation = augmentation or {}

        pairs = _match_pairs(self.image_dir, self.mask_dir)
        pairs_by_name = {
            str(img.relative_to(self.image_dir)).replace("\\", "/"): (img, mask)
            for img, mask in pairs
        }
        if split_file is None:
            self.pairs = pairs
        else:
            split_path = self.root / split_file
            if split_path.suffix.lower() == ".json":
                with split_path.open("r", encoding="utf-8") as f:
                    split = json.load(f)
                names = split[split_key]
            else:
                names = _read_txt_split(split_path)
            self.pairs = [pairs_by_name[name] for name in names if name in pairs_by_name]
            if not self.pairs:
                raise ValueError(f"split {split_path} did not match any image files")

    def __len__(self) -> int:
        return len(self.pairs)

    def _apply_geometry(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        aug = self.augmentation
        size = tuple(self.input_size)
        crop_cfg = aug.get("random_resized_crop", {})
        if self.train and crop_cfg.get("enabled", False):
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=tuple(crop_cfg.get("scale", (0.75, 1.0))),
                ratio=tuple(crop_cfg.get("ratio", (0.9, 1.1))),
            )
            image = TF.resized_crop(image, i, j, h, w, size, InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, size, InterpolationMode.NEAREST)
        else:
            image = TF.resize(image, size, InterpolationMode.BILINEAR)
            mask = TF.resize(mask, size, InterpolationMode.NEAREST)

        if self.train and aug.get("horizontal_flip", {}).get("enabled", False):
            if random.random() < float(aug["horizontal_flip"].get("p", 0.5)):
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        if self.train and aug.get("vertical_flip", {}).get("enabled", False):
            if random.random() < float(aug["vertical_flip"].get("p", 0.5)):
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        rot_cfg = aug.get("random_rotation", {})
        if self.train and rot_cfg.get("enabled", False):
            degrees = float(rot_cfg.get("degrees", 20))
            angle = random.uniform(-degrees, degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return image, mask

    def _apply_color(self, image: Image.Image) -> Image.Image:
        jitter_cfg = self.augmentation.get("color_jitter", {})
        if self.train and jitter_cfg.get("enabled", False):
            jitter = transforms.ColorJitter(
                brightness=float(jitter_cfg.get("brightness", 0.0)),
                contrast=float(jitter_cfg.get("contrast", 0.0)),
                saturation=float(jitter_cfg.get("saturation", 0.0)),
                hue=float(jitter_cfg.get("hue", 0.0)),
            )
            image = jitter(image)
        return image

    def _apply_tensor_aug(self, image: torch.Tensor) -> torch.Tensor:
        noise_cfg = self.augmentation.get("gaussian_noise", {})
        if self.train and noise_cfg.get("enabled", False):
            if random.random() < float(noise_cfg.get("p", 0.25)):
                image = torch.clamp(image + torch.randn_like(image) * float(noise_cfg.get("std", 0.03)), 0, 1)
        cutout_cfg = self.augmentation.get("cutout", {})
        if self.train and cutout_cfg.get("enabled", False):
            if random.random() < float(cutout_cfg.get("p", 0.25)):
                cut = int(cutout_cfg.get("size", 32))
                _, height, width = image.shape
                top = random.randint(0, max(0, height - cut))
                left = random.randint(0, max(0, width - cut))
                image[:, top : top + cut, left : left + cut] = 0
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self._apply_geometry(image, mask)
        image = self._apply_color(image)
        image_tensor = TF.to_tensor(image)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float()
        image_tensor = self._apply_tensor_aug(image_tensor)
        return image_tensor, mask_tensor


def build_datasets(cfg: Dict[str, object]) -> Tuple[Dataset, Dataset]:
    dataset_cfg = cfg["dataset"]
    exp_cfg = cfg.get("experiment", {})
    aug_cfg = cfg.get("augmentation", {})
    if not str(dataset_cfg.get("root", "")):
        raise ValueError("dataset.root is empty; set it in the YAML config")
    root = Path(dataset_cfg["root"])
    name = dataset_cfg["name"]
    if name == "MMOTU":
        ensure_mmotu_splits(
            root,
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["train_split_file"],
            dataset_cfg["val_split_file"],
            int(dataset_cfg["train_size"]),
            int(dataset_cfg["val_size"]),
        )
        train = PairedSegmentationDataset(
            str(root),
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["train_split_file"],
            "train",
            dataset_cfg["input_size"],
            train=True,
            augmentation=aug_cfg,
        )
        val = PairedSegmentationDataset(
            str(root),
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["val_split_file"],
            "val",
            dataset_cfg["input_size"],
            train=False,
            augmentation=aug_cfg,
        )
    elif name == "ISIC-2018":
        ensure_isic_split(
            root,
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["split_file"],
            float(dataset_cfg["split_ratio"]),
            int(exp_cfg.get("seed", 42)),
        )
        train = PairedSegmentationDataset(
            str(root),
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["split_file"],
            "train",
            dataset_cfg["input_size"],
            train=True,
            augmentation=aug_cfg,
        )
        val = PairedSegmentationDataset(
            str(root),
            dataset_cfg["image_dir"],
            dataset_cfg["mask_dir"],
            dataset_cfg["split_file"],
            "val",
            dataset_cfg["input_size"],
            train=False,
            augmentation=aug_cfg,
        )
    else:
        raise ValueError(f"unsupported dataset: {name}")
    return train, val
