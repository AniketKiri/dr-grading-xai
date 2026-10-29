"""Torch-free data utilities: fundus cropping, geometry bookkeeping, duplicate hashing, grouping.

Kept free of PyTorch so the integrity checks can be run and unit-tested anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------- id -> file resolution
def index_images(image_dir: Path) -> dict[str, Path]:
    """Map file stem -> path. Refuses ambiguity: two files sharing a stem is an error, not a guess.

    This is deliberately strict. A resolver that silently picks one of several candidates can
    attach a label to the wrong photograph with no visible symptom.
    """
    index: dict[str, Path] = {}
    clashes: list[str] = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.stem in index:
                clashes.append(p.stem)
            index[p.stem] = p
    if clashes:
        raise RuntimeError(f"{len(clashes)} image stems map to more than one file, e.g. {clashes[:5]}")
    return index


def resolve_ids(ids, index: dict[str, Path]) -> list[Path]:
    """Exact stem match only. Every id must resolve, and the resolved stem must equal the id."""
    missing = [i for i in ids if str(i) not in index]
    if missing:
        raise RuntimeError(f"{len(missing)} labelled ids have no image file, e.g. {missing[:5]}")
    paths = [index[str(i)] for i in ids]
    for i, p in zip(ids, paths):
        assert p.stem == str(i), f"resolver returned {p.name} for id {i}"
    return paths


# ---------------------------------------------------------------- fundus crop geometry
@dataclass
class CropGeometry:
    orig_w: int
    orig_h: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    pad_x: int
    pad_y: int
    scale: float   # cached pixels per original pixel

    def to_cached(self, x, y):
        """Original pixel coordinates -> cached-image coordinates."""
        x = (np.asarray(x, float) - self.crop_x0 + self.pad_x) * self.scale
        y = (np.asarray(y, float) - self.crop_y0 + self.pad_y) * self.scale
        return x, y

    def warp_mask(self, mask: np.ndarray, out_size: int) -> np.ndarray:
        """Apply the same crop, pad and resize to an original-resolution binary mask."""
        crop = mask[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]
        side = int(round(max(crop.shape) if crop.size else 1))
        square = np.zeros((side, side), dtype=np.uint8)
        square[self.pad_y:self.pad_y + crop.shape[0], self.pad_x:self.pad_x + crop.shape[1]] = crop > 0
        return np.asarray(Image.fromarray(square * 255).resize((out_size, out_size), Image.NEAREST)) > 0

    def as_dict(self) -> dict:
        return asdict(self)


FOV_THRESH = 10        # brightest channel above this is illuminated retina, when the surround is black
SURROUND_MAX = 100     # corners brighter than this are a white canvas around a black ring, not a camera surround


def surround_level(rgb: np.ndarray, frac: float = 0.04) -> float:
    """Brightness of the unlit surround: median of the four corner-patch medians (brightest channel).
    Medians, so JPEG noise or one corner clipped by the field of view cannot move the estimate."""
    v = rgb.max(axis=2)
    h, w = v.shape
    k = max(4, int(frac * min(h, w)))
    return float(np.median([np.median(c) for c in (v[:k, :k], v[:k, -k:], v[-k:, :k], v[-k:, -k:])]))


def surround_threshold(rgb: np.ndarray) -> tuple[int, bool]:
    """(threshold, corrected). One MMRDR camera leaves a grey surround (about 20) that the fixed
    threshold counts as retina; that surround is almost never seen at grade 0, so it would let a model
    read the grade off the border. For such images the threshold follows the surround. Black
    surrounds (level below the fixed threshold) and white canvases keep the fixed threshold, so
    their processing is exactly as before."""
    level = surround_level(rgb)
    if FOV_THRESH <= level <= SURROUND_MAX:
        return int(level) + FOV_THRESH, True
    return FOV_THRESH, False


def fov_bbox(rgb: np.ndarray, thresh: int | None = None, min_frac: float = 0.01) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1), end-exclusive, of the illuminated fundus field of view."""
    mask = fov_mask(rgb, thresh)
    rows = np.where(mask.mean(axis=1) > min_frac)[0]
    cols = np.where(mask.mean(axis=0) > min_frac)[0]
    h, w = mask.shape
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop_to_square(img: Image.Image, size: int) -> tuple[Image.Image, CropGeometry]:
    """Crop the black border, pad to a square (keeping the disc centred), resize to size x size.
    A grey surround is set to black first, so every cached image has the same black border."""
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    thresh, corrected = surround_threshold(rgb)
    if corrected:
        rgb = np.where(retina_region(rgb, thresh)[..., None], rgb, 0).astype(np.uint8)
    x0, y0, x1, y1 = fov_bbox(rgb, thresh)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    pad_y, pad_x = (side - ch) // 2, (side - cw) // 2
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[pad_y:pad_y + ch, pad_x:pad_x + cw] = crop
    out = Image.fromarray(square).resize((size, size), Image.BICUBIC)
    geom = CropGeometry(w, h, x0, y0, x1, y1, pad_x, pad_y, size / side)
    return out, geom


def fov_mask(rgb: np.ndarray, thresh: int | None = None) -> np.ndarray:
    """Boolean mask of the illuminated field of view, by brightness threshold."""
    if thresh is None:
        thresh = surround_threshold(rgb)[0]
    return rgb.max(axis=2) > thresh


def retina_region(rgb: np.ndarray, thresh: int, margin_frac: float = 0.005) -> np.ndarray:
    """The field of view as a solid region, for blacking out a grey surround. Holes are filled so dark
    retina inside the field (a haemorrhage, a shadowed vessel) is kept. The frame margin is cleared
    first: some images carry a thin bright line round the edge, which would otherwise enclose the
    whole surround and make it a 'hole'."""
    from scipy.ndimage import binary_fill_holes

    core = fov_mask(rgb, thresh)
    m = max(2, int(round(margin_frac * min(core.shape))))
    core[:m], core[-m:], core[:, :m], core[:, -m:] = False, False, False, False
    return binary_fill_holes(core)


# ---------------------------------------------------------------- near-duplicate detection
def fingerprint(img: Image.Image, size: int = 48) -> np.ndarray:
    """Unit-norm vector of fine retinal structure, for finding the same photograph under two ids.

    Every fundus photograph shares a dark border, a bright disc and central vignetting, so a plain
    thumbnail or perceptual hash rates unrelated images as near-identical. Subtracting a blurred
    copy removes that shared low-frequency structure and leaves vessels and lesions, which differ
    between photographs. Only the central 60% is used, so the field-of-view rim (shared by all) is excluded.
    """
    from PIL import ImageFilter

    g = img.convert("L").resize((4 * size, 4 * size), Image.BILINEAR)
    hp = np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.GaussianBlur(6)), np.float32)
    m = int(0.2 * hp.shape[0])   # central 60%: its corners stay inside a centred circular field of view
    hp = hp[m:-m, m:-m]
    v = np.asarray(Image.fromarray(hp).resize((size, size), Image.BOX), np.float32).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def near_duplicate_pairs(fps: np.ndarray, min_corr: float, chunk: int = 1024) -> np.ndarray:
    """All index pairs (i < j) whose fingerprints correlate at or above min_corr. Returns (P, 3): i, j, corr."""
    fps = np.asarray(fps, np.float32)
    out = []
    for start in range(0, len(fps), chunk):
        c = fps[start:start + chunk] @ fps.T
        ii, jj = np.nonzero(c >= min_corr)
        ii = ii + start
        keep = ii < jj
        if keep.any():
            out.append(np.stack([ii[keep], jj[keep], c[ii[keep] - start, jj[keep]]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 3))


def connected_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    """Union-find over duplicate pairs: images that are near-copies share a group id."""
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs[:, :2].astype(int):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


# ---------------------------------------------------------------- image quality
QUALITY_COLS = ["q_sharpness", "q_brightness", "q_contrast", "q_fov_fraction"]


def image_quality(img: Image.Image) -> dict:
    """Simple, interpretable acquisition-quality measures on the cached image, inside the field of view.

    sharpness: variance of the Laplacian of the green channel (low = blurred or out of focus)
    brightness / contrast: mean and SD of the green channel (low = under-exposed, e.g. small pupil)
    fov_fraction: share of the square occupied by the illuminated field (low = cropped or vignetted)
    The green channel is used because it carries most retinal contrast in fundus photographs.
    """
    from scipy.ndimage import laplace

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    fov = fov_mask(rgb.astype(np.uint8))
    g = rgb[..., 1]
    if fov.sum() < 100:
        return {c: float("nan") for c in QUALITY_COLS}
    inner = fov.copy()
    inner[1:-1, 1:-1] = fov[:-2, 1:-1] & fov[2:, 1:-1] & fov[1:-1, :-2] & fov[1:-1, 2:]  # avoid the rim edge
    return {"q_sharpness": float(laplace(g)[inner].var()), "q_brightness": float(g[fov].mean()),
            "q_contrast": float(g[fov].std()), "q_fov_fraction": float(fov.mean())}
