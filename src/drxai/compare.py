"""Step 5. Compare two trained runs and choose between them on validation, never on test.

    python -m drxai.compare --runs runs/resnet50_corn runs/resnet50_ce

The primary model is the one with the higher validation QWK. The test difference is then reported
with a paired, group-resampled bootstrap interval, so a small test gap is not mistaken for a finding.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .config import resolve, save_json
from .metrics import bootstrap_ci, qwk
from .ordinal import outputs_to_pred


def paired_difference(y, pred_a, pred_b, groups, K: int, B: int, seed: int) -> dict:
    """QWK(a) - QWK(b) on the same test images, resampling groups jointly for both models."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    return bootstrap_ci(lambda i: qwk(y[i], pred_a[i], K) - qwk(y[i], pred_b[i], K), groups, B, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs=2, required=True)
    args = ap.parse_args()
    from .inference import load_manifest

    runs = [resolve(r) for r in args.runs]
    cfgs = [json.loads((r / "config_used.json").read_text(encoding="utf-8")) for r in runs]
    mets = [json.loads((r / "metrics.json").read_text(encoding="utf-8")) for r in runs]
    logits = [np.load(r / "logits.npz", allow_pickle=True) for r in runs]
    assert (logits[0]["test_ids"] == logits[1]["test_ids"]).all(), "runs were evaluated on different test images"

    cfg, K = cfgs[0], cfgs[0]["data"]["num_classes"]
    man = load_manifest(cfg).set_index("image_id")
    ids = logits[0]["test_ids"]
    y = man.loc[ids, "grade"].to_numpy()
    groups = man.loc[ids, "group"].to_numpy()
    preds = [outputs_to_pred(l["test"], c["model"]["head"]) for l, c in zip(logits, cfgs)]

    val = [m["validation"]["qwk"] for m in mets]
    primary = int(np.argmax(val))
    order = [primary, 1 - primary]
    diff = paired_difference(y, preds[order[0]], preds[order[1]], groups, K, cfg["eval"]["bootstrap"], cfg["seed"])
    out = {
        "selection_rule": "higher validation QWK",
        "runs": [{"run": str(r), "head": c["model"]["head"], "val_qwk": m["validation"]["qwk"],
                  "test_qwk": m["performance"]["qwk"]} for r, c, m in zip(runs, cfgs, mets)],
        "primary": str(runs[primary]),
        "test_qwk_difference_primary_minus_other": diff,
        "interval_excludes_zero": bool(diff["lo"] > 0 or diff["hi"] < 0),
    }
    save_json(out, runs[primary].parent / "comparison.json")
    for r in out["runs"]:
        print(f"{r['run']}: head {r['head']}  val QWK {r['val_qwk']:.3f}  test QWK {r['test_qwk']:.3f}")
    print(f"primary (chosen on validation): {out['primary']}")
    print(f"test QWK difference {diff['point']:+.3f} (95% CI {diff['lo']:+.3f} to {diff['hi']:+.3f})"
          + ("" if out["interval_excludes_zero"] else "  -> interval includes zero: no demonstrated difference"))


if __name__ == "__main__":
    main()
