"""Check that the demo reproduces evaluate for its six examples.

    python demo/check_examples.py

Compares the demo's crop with prepare's cached image (pixel for pixel), and its grade and confidence
with predictions_test.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import Pipeline, example_ids, original_path  # noqa: E402

from drxai.config import cache_dir  # noqa: E402

CONF_TOL = 1e-3
from drxai.inference import load_manifest  # noqa: E402


def main() -> None:
    pipe = Pipeline()
    pred = pd.read_csv(pipe.run_dir / "predictions_test.csv", dtype={"image_id": str}).set_index("image_id")
    man = load_manifest(pipe.cfg).set_index("image_id")
    rows = []
    for i in example_ids(pred.reset_index()):
        img = Image.open(original_path(pipe.cfg, i))
        crop, _, _ = pipe.preprocess(img)
        cached = np.asarray(Image.open(cache_dir(pipe.cfg) / man.loc[i, "cache_path"]).convert("RGB"))
        r = pipe(img)
        e = pred.loc[i]
        rows.append({"image_id": i, "true": int(e["grade"]), "crop_identical": bool(np.array_equal(np.asarray(crop), cached)),
                     "pred_csv": int(e["pred"]), "pred_app": r.pred,
                     "conf_csv": f"{e['confidence']:.6f}", "conf_app": f"{r.confidence:.6f}",
                     "abs_diff": abs(float(e["confidence"]) - r.confidence),
                     "sharp_manifest": round(float(man.loc[i, "q_sharpness"]), 3), "sharp_app": round(r.sharpness, 3)})
    df = pd.DataFrame(rows)
    print(f"device: {pipe.device}")
    print(df.to_string(index=False))
    # Grades and crops must match exactly. Confidence may not: the app infers one image at a time under mixed
    # precision while evaluate used batches of 32, and fp16 rounding depends on the batch (each example matches
    # exactly inside its original batch; see the README's demo section). 1e-3 bounds that rounding.
    exact = df["crop_identical"].all() and (df["pred_csv"] == df["pred_app"]).all()
    if pipe.device == "cuda":
        ok = exact and (df["abs_diff"] <= CONF_TOL).all()
        print(f"PASS (grades exact, confidence within {CONF_TOL:g})" if ok else
              f"FAIL (grade, crop or confidence outside {CONF_TOL:g})")
    else:
        # The reference confidences were computed on the GPU under mixed precision, so a full-precision CPU run
        # differs by more than the GPU rounding; only grades are held to a pass/fail here.
        print("PASS (grades exact)" if exact else "FAIL (grade or crop differs)")
        print(f"Confidence: maximum absolute difference {df['abs_diff'].max():.4f} "
              "(full-precision CPU against the mixed-precision GPU reference; no pass/fail applied)")


if __name__ == "__main__":
    main()
