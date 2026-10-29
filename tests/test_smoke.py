"""End-to-end run of every command on synthetic data (about a minute on a laptop CPU).

    pytest tests/test_smoke.py -s

Run this before touching real data: it proves the environment and every stage work together.
"""
import json
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("torch")
from synthetic import build  # noqa: E402


def run(*args):
    r = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    assert r.returncode == 0, f"{' '.join(args)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_pipeline_end_to_end(tmp_path):
    cfg = str(build(tmp_path))
    run("drxai.inspect_data", "--config", cfg)
    out = run("drxai.prepare", "--config", cfg, "--workers", "1")
    assert "1 training images dropped" in out, "planted duplicate was not detected"

    for head in ("corn", "ce"):
        run("drxai.train", "--config", cfg, "--run-name", f"tiny_{head}", "--head", head)
        rd = tmp_path / "runs" / f"tiny_{head}"
        run("drxai.evaluate", "--run", str(rd), "--quality")
        run("drxai.explain", "--run", str(rd))
        run("drxai.lesions_report", "--run", str(rd))
        lr = json.loads((rd / "lesions_report.json").read_text())
        assert lr["consistency"]["any_rule"]["total"] == 0, "synthetic flags are built to be consistent"
        assert any(not r["skipped"] for r in lr["probe"]["trained"]), "no lesion had enough positives to probe"

        m = json.loads((rd / "metrics.json").read_text())
        assert m["n_test"] == 60
        assert {"quality_probe", "referral_curve", "calibration"} <= set(m)
        preds = pd.read_csv(rd / "predictions_test.csv", dtype={"image_id": str})
        assert preds["image_id"].str.startswith("ts").all(), "a non-test image reached the test predictions"
        e = json.loads((rd / "explain_summary.json").read_text())
        assert isinstance(e["lesion_agreement"], dict) and e["lesion_agreement"]["n_annotated_test_images"] > 0
        assert "referral_rule" in m and "quality_strata_sharpness" in m
        for f in ("confusion_matrix", "reliability", "referral_curve", "quality_probe", "gradcam_gallery",
                  "sanity_check", "case_cards"):
            assert (rd / "figures" / f"{f}.png").exists(), f

    out = run("drxai.compare", "--runs", str(tmp_path / "runs" / "tiny_corn"), str(tmp_path / "runs" / "tiny_ce"))
    assert "primary (chosen on validation)" in out
