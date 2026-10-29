"""A small synthetic dataset laid out exactly like MMRDR-CFP (FP.csv with type, image, grade, lesion, lr;
images under img/), for exercising the whole pipeline without real data.

Lesions are drawn as bright spots whose number rises with grade, and saved as masks, so the
lesion-agreement code has something real to agree with. Image ts000003 is also planted in the
training pool under another id (resized and re-encoded) to check the duplicate detector.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]


def fake_fundus(rng, size=400) -> Image.Image:
    im = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 20, size - 20, size - 20], fill=(170, 80, 40))
    for _ in range(12):
        pts = [tuple(int(v) for v in rng.integers(60, size - 60, 2)) for _ in range(4)]
        d.line(pts, fill=(110, 30, 20), width=int(rng.integers(2, 5)))
    return im


# n_train is sized so validation (15% of it) can hold the 20 positives and 20 negatives per lesion
# that the lesion probe requires; at 240 it had 35 images and every lesion was skipped.
def build(root: Path, n_train=480, n_test=60, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    data = root / "DatasetRT" / "MMRDR-CFP"
    (data / "img").mkdir(parents=True, exist_ok=True)
    masks = root / "DatasetRT" / "lesion_masks"
    masks.mkdir(parents=True, exist_ok=True)
    rows = []
    for pre, n in [("tr", n_train), ("ts", n_test)]:
        for i in range(n):
            iid = f"{pre}{i:06d}"
            g = int(rng.choice(5, p=[.45, .15, .2, .1, .1]))
            canvas = Image.new("RGB", (600, 450))
            canvas.paste(fake_fundus(rng).resize((500, 400)), (50, 25))
            d, m = ImageDraw.Draw(canvas), Image.new("L", (600, 450))
            dm = ImageDraw.Draw(m)
            for _ in range(g * 5):
                x, y = (int(v) for v in rng.integers(180, 420, 2))
                d.ellipse([x, y, x + 9, y + 9], fill=(230, 220, 90))
                dm.ellipse([x, y, x + 9, y + 9], fill=255)
            canvas.save(data / "img" / f"{iid}.jpg", quality=92)
            if g > 0:
                m.save(masks / f"{iid}_EX.png")
            flags = [int(g >= 1), int(g >= 2), int(g >= 2), int(g >= 3), int(g == 4), int(g == 4 and rng.uniform() < .5), 0]
            rows.append((iid, g, int(rng.integers(0, 2)), flags))
    src = [r for r in rows if r[0] == "ts000003"][0]
    Image.open(data / "img" / "ts000003.jpg").resize((660, 495)).save(data / "img" / "tr000999.jpg", quality=80)
    rows.append(("tr000999", src[1], 0, src[3]))
    pd.DataFrame({"type": "FP", "image": [f"img/{r[0]}.jpg" for r in rows], "grade": [r[1] for r in rows],
                  "lesion": [str(r[3]) for r in rows], "lr": [r[2] for r in rows]}).to_csv(data / "FP.csv", index=False)

    cfg = yaml.safe_load(open(REPO / "configs" / "default.yaml"))
    cfg["paths"].update(data_root=str(data), label_csv=None, cache_dir=str(root / "cache"), runs_dir=str(root / "runs"))
    cfg["columns"] = {"image_id": "image", "grade": "grade", "eye": "lr", "patient_id": None, "split": None}
    cfg["split_from_id_prefix"] = {"tr": "train", "ts": "test"}
    cfg["data"].update(cache_size=96, img_size=64)
    cfg["model"].update(backbone="tiny", pretrained=False)
    cfg["train"].update(epochs=2, batch_size=16, num_workers=0, patience=5)
    cfg["eval"].update(bootstrap=50, quality_blur_sigmas=[0, 2], quality_gammas=[1.0, 2.0])
    cfg["explain"].update(n_per_group=1)
    cfg["explain"]["lesions"].update(format="mask_dir", mask_dir=str(masks))
    path = root / "config.yaml"
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path
