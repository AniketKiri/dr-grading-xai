"""PyTorch dataset and transforms over the cached manifest."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int) -> T.Compose:
    # Fundus photographs have no canonical orientation for grading, so flips and full
    # rotations are safe here (unlike OCT B-scans, where a vertical flip inverts layer order).
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(180),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int) -> T.Compose:
    return T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def degrade(img: Image.Image, blur_sigma: float = 0.0, gamma: float = 1.0) -> Image.Image:
    """Simulated acquisition defects for the image-quality sensitivity probe."""
    if blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    if gamma != 1.0:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img = Image.fromarray((np.clip(arr ** gamma, 0, 1) * 255).astype(np.uint8))
    return img


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_root: Path, transform, blur_sigma=0.0, gamma=1.0):
        self.paths = [Path(cache_root) / p for p in df["cache_path"]]
        self.labels = df["grade"].to_numpy(dtype=np.int64)
        self.ids = df["image_id"].astype(str).tolist()
        self.transform = transform
        self.blur_sigma, self.gamma = blur_sigma, gamma

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.blur_sigma or self.gamma != 1.0:
            img = degrade(img, self.blur_sigma, self.gamma)
        return self.transform(img), torch.tensor(self.labels[i]), i


def class_balanced_sampler(labels: np.ndarray, mode: str, seed: int):
    """sqrt_inverse softens the correction: full inverse weighting over-samples the rare grades
    so heavily that the model sees the same few severe/PDR images many times per epoch."""
    if mode == "none":
        return None
    counts = np.bincount(labels)
    w_class = 1.0 / np.sqrt(counts) if mode == "sqrt_inverse" else 1.0 / counts
    weights = w_class[labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels),
                                 replacement=True, generator=g)
