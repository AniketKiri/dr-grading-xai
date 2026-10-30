"""Grey-surround accuracy gap after standardising for grade mix. POST HOC analysis.

    python analysis/grey_standardised.py

Post hoc: written after the test results were seen, to ask whether the grey-surround accuracy gap in
results.md is explained by grade mix (the grey test images are almost all grades 1 to 4, which are harder
than grade 0). It is a measurement on the already-measured primary run and chooses nothing.

Statistic: accuracy of the grey-surround test images minus the accuracy they would be expected to have if
each grade were classified as accurately as black-surround images of that grade, weighted by the grey
images' own grade mix (direct standardisation). A value near zero means the raw gap is grade mix.

CI: grouped bootstrap over manifest `group` (near-duplicate clusters), 2,000 draws, numpy default_rng(42),
2.5/97.5 percentiles; both subsets and the black per-grade accuracies are recomputed in every draw.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drxai.metrics import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
K = 5


def main() -> None:
    pred = pd.read_csv(ROOT / "runs/resnet50_corn/predictions_test.csv", dtype={"image_id": str})
    man = pd.read_csv(ROOT / "cache/manifest.csv", dtype={"image_id": str, "group": str},
                      usecols=["image_id", "grey_surround", "group", "split"])
    df = pred.merge(man, on="image_id", how="left", validate="one_to_one")
    assert df["group"].notna().all() and (df["split"] == "test").all(), "prediction not matched to a test row"
    y, p = df["grade"].to_numpy(), df["pred"].to_numpy()
    grey, groups = df["grey_surround"].astype(bool).to_numpy(), df["group"].to_numpy()

    def stat(idx: np.ndarray) -> float:
        yi, ok, gi = y[idx], (y[idx] == p[idx]), grey[idx]
        if not gi.any():
            return np.nan
        expected = 0.0
        for g in range(K):
            in_grey = gi & (yi == g)
            if not in_grey.any():
                continue
            black = ~gi & (yi == g)
            if not black.any():          # no black reference for this grade in this draw
                return np.nan
            expected += in_grey.sum() / gi.sum() * ok[black].mean()
        return float(ok[gi].mean() - expected)

    r = bootstrap_ci(stat, groups, B=2000, seed=42)
    print(f"grey n = {grey.sum()}, black n = {(~grey).sum()}")
    print(f"grey accuracy minus grade-standardised expectation: "
          f"{r['point']:+.3f} ({r['lo']:+.3f} to {r['hi']:.3f})")


if __name__ == "__main__":
    main()
