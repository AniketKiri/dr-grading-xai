"""Mapping lesion annotations into model coordinates and scoring explanation agreement. NumPy only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .config import resolve
from .data import CropGeometry

GEOM_FIELDS = ["orig_w", "orig_h", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "pad_x", "pad_y", "scale"]


def geometry(row) -> CropGeometry:
    return CropGeometry(**{k: (float(row[k]) if k == "scale" else int(row[k])) for k in GEOM_FIELDS})


class LesionSource:
    """Loads a binary lesion mask at model input resolution, or None where an image has no annotation."""

    def __init__(self, lesion_cfg: dict, img_size: int, cache_size: int):
        self.fmt = lesion_cfg.get("format")
        self.img_size, self.cache_size = img_size, cache_size
        self.mask_files: dict[str, list[Path]] = {}
        self.boxes = None
        if self.fmt == "mask_dir":
            for p in sorted(resolve(lesion_cfg["mask_dir"]).rglob("*.png")):
                stem = p.stem
                key = stem.split("_")[0].split("-")[0]   # <id>, <id>_HE, <id>-MA ... all map to <id>
                self.mask_files.setdefault(key, []).append(p)
        elif self.fmt == "bbox_csv":
            self.boxes = pd.read_csv(resolve(lesion_cfg["bbox_csv"]), dtype={"image_id": str})
        elif self.fmt is not None:
            raise ValueError(f"unknown lesion format {self.fmt!r}")

    @property
    def enabled(self) -> bool:
        return self.fmt is not None

    def mask(self, image_id: str, geom: CropGeometry) -> np.ndarray | None:
        if self.fmt == "mask_dir":
            files = self.mask_files.get(image_id)
            if not files:
                return None
            union = np.zeros((geom.orig_h, geom.orig_w), dtype=bool)
            for f in files:
                m = np.asarray(Image.open(f).convert("L")) > 0
                if m.shape != union.shape:
                    raise ValueError(f"{f.name}: mask {m.shape} does not match image {union.shape}")
                union |= m
            return geom.warp_mask(union, self.img_size) if union.any() else None
        if self.fmt == "bbox_csv":
            rows = self.boxes[self.boxes["image_id"] == image_id]
            if rows.empty:
                return None
            out = np.zeros((self.img_size, self.img_size), dtype=bool)
            f = self.img_size / self.cache_size
            for _, r in rows.iterrows():
                xs, ys = geom.to_cached([r["x0"], r["x1"]], [r["y0"], r["y1"]])
                x0, x1 = np.clip(np.round(np.sort(xs) * f).astype(int), 0, self.img_size)
                y0, y1 = np.clip(np.round(np.sort(ys) * f).astype(int), 0, self.img_size)
                out[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
            return out
        return None


def pointing_game(cam: np.ndarray, lesion: np.ndarray, fov: np.ndarray, tol_px: float) -> dict:
    """Does the heatmap's peak land on (or within tol_px of) an annotated lesion?

    The floor is the chance that a point drawn uniformly from the field of view lands in the
    same tolerance region. Lesion-dense images have a high floor, so a raw hit rate means
    little without it.
    """
    near = distance_transform_edt(~lesion) <= tol_px
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    return {"hit": bool(near[peak]), "chance": float((near & fov).sum() / max(fov.sum(), 1))}


def energy_enrichment(cam: np.ndarray, lesion: np.ndarray, fov: np.ndarray) -> float:
    """Share of positive heatmap mass inside lesions divided by the lesions' share of the field of view.
    1.0 means the heatmap is no more concentrated on lesions than uniform attention would be."""
    c = np.clip(cam, 0, None) * fov
    area = (lesion & fov).sum() / max(fov.sum(), 1)
    if c.sum() == 0 or area == 0:
        return float("nan")
    return float((c[lesion & fov].sum() / c.sum()) / area)
