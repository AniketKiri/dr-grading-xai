"""Single-image path through the existing pipeline, shared by the demo app and its check script.

Nothing here reimplements a step: preprocessing is prepare's (surround handling, field-of-view crop,
square pad, resize, quality), inference is evaluate's (infer on raw logits, temperature from the run's
metrics.json applied to confidence only), Grad-CAM is explain's (expected-grade target).
"""
from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from drxai.calibration import confidence
from drxai.config import resolve
from drxai.data import crop_to_square, image_quality, surround_threshold
from drxai.explain import GradCAM, cams_for
from drxai.inference import device, infer, load_run
from drxai.model import gradcam_layer
from drxai.ordinal import outputs_to_pred, outputs_to_probs


@dataclass
class Result:
    pred: int
    confidence: float
    probs: np.ndarray
    p_referable: float
    action: str
    sharpness: float
    low_sharpness: bool
    grey_surround: bool
    display: np.ndarray      # the network's view, resized for display (as explain shows it)
    cam: np.ndarray


class Pipeline:
    def __init__(self, run: str = "runs/resnet50_corn"):
        self.run_dir = resolve(run)
        self.cfg, self.net = load_run(self.run_dir)
        m = json.loads((self.run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.T = float(m["temperature"])
        self.tau = float(m["referral_rule"]["threshold"])
        # lower edge of the middle sharpness band, set on training images by evaluate
        self.sharp_low = float(m["quality_strata_sharpness"]["cuts"][0])
        self.head, self.backbone = self.cfg["model"]["head"], self.cfg["model"]["backbone"]
        self.K, self.ref = self.cfg["data"]["num_classes"], self.cfg["data"]["referable_min_grade"]
        # Grad-CAM hooks fail under no_grad, so inference and explanation use separate copies
        cam_net = copy.deepcopy(self.net)
        self.cam = GradCAM(cam_net, gradcam_layer(cam_net, self.backbone), self.head)
        self.tmp = Path(tempfile.mkdtemp(prefix="drxai_demo_"))
        self.device = device().type

    def preprocess(self, img: Image.Image) -> tuple[Image.Image, bool, float]:
        """prepare._process_one for one image: the cached crop, the grey-surround flag, sharpness."""
        img = img.convert("RGB")
        grey = surround_threshold(np.asarray(img))[1]
        crop, _ = crop_to_square(img, self.cfg["data"]["cache_size"])
        return crop, grey, image_quality(crop)["q_sharpness"]

    def __call__(self, img: Image.Image) -> Result:
        crop, grey, sharp = self.preprocess(img)
        # PNG is lossless, so the network reads exactly what prepare would have cached
        path = self.tmp / "input.png"
        crop.save(path)
        df = pd.DataFrame({"image_id": ["input"], "grade": [0], "cache_path": [str(path)]})
        logits = infer(self.net, df, self.cfg)
        pred = outputs_to_pred(logits, self.head)
        probs = outputs_to_probs(logits, self.head, self.T)
        conf = float(confidence(probs, pred)[0])
        raws, maps = cams_for(df, self.cam, self.cfg)
        p = probs[0]
        return Result(pred=int(pred[0]), confidence=conf, probs=p, p_referable=float(p[self.ref:].sum()),
                      action="no flag: routine examination" if conf >= self.tau else "flag for extended optometrist review",
                      sharpness=float(sharp), low_sharpness=bool(sharp < self.sharp_low), grey_surround=bool(grey),
                      display=raws[0], cam=maps[0])


def example_ids(pred: pd.DataFrame) -> list[str]:
    """Fixed rule: the lowest test id of each true grade, then ts000729 (a confident miss of a referable
    eye, the case-card example the referral rule cannot catch)."""
    first = pred.sort_values("image_id").groupby("grade")["image_id"].first()
    return [first[g] for g in sorted(first.index)] + ["ts000729"]


def original_path(cfg: dict, image_id: str) -> Path:
    d = resolve(cfg["paths"]["data_root"]) / cfg["paths"]["image_dir"]
    hits = sorted(d.glob(f"{image_id}.*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one original file for {image_id} in {d}, found {len(hits)}")
    return hits[0]
