"""Step 3. Evaluate a trained run on the held-out test split.

    python -m drxai.evaluate --run runs/resnet50_corn [--quality]

Temperature is fitted on validation and applied to test. Writes metrics.json, results.md,
predictions_test.csv and figures/ inside the run folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import (apply_referral_rule, confidence, ece, fit_temperature, referral_curve,
                          threshold_for_coverage)
from .config import resolve, save_json
from .metrics import (bootstrap_ci, classification_metrics, error_direction, majority_floor, qwk,
                      quality_strata, referable_auc, subgroup_report)
from .ordinal import outputs_to_pred, outputs_to_probs


def analyse(y_test, logits_test, y_val, logits_val, y_train, groups_test, cfg: dict,
            quality_test=None, quality_train=None, grey_test=None) -> tuple[dict, dict]:
    """All test-set numbers, computed from saved logits. Pure NumPy so it is unit-testable.

    Everything that is *chosen* (temperature, referral threshold, quality band edges) is chosen on
    validation or training data. The test split is only ever measured.
    """
    K, ref = cfg["data"]["num_classes"], cfg["data"]["referable_min_grade"]
    head = cfg["model"]["head"]
    y_test, y_val = np.asarray(y_test), np.asarray(y_val)

    pred = outputs_to_pred(logits_test, head)
    T = fit_temperature(logits_val, y_val, head)
    probs_raw = outputs_to_probs(logits_test, head, 1.0)
    probs = outputs_to_probs(logits_test, head, T)
    conf = confidence(probs, pred)
    correct = (pred == y_test).astype(float)

    perf = classification_metrics(y_test, pred, K)
    floor = majority_floor(y_train, y_test, K)
    B, seed = cfg["eval"]["bootstrap"], cfg["seed"]
    ci = {
        "qwk": bootstrap_ci(lambda i: qwk(y_test[i], pred[i], K), groups_test, B, seed),
        "accuracy": bootstrap_ci(lambda i: float((y_test[i] == pred[i]).mean()), groups_test, B, seed),
        "macro_f1": bootstrap_ci(lambda i: classification_metrics(y_test[i], pred[i], K)["macro_f1"],
                                 groups_test, B, seed),
    }
    margins = {k: perf[k] - floor[k] for k in ("accuracy", "macro_f1", "qwk")}

    pred_val = outputs_to_pred(logits_val, head)
    conf_val = confidence(outputs_to_probs(logits_val, head, T), pred_val)
    tau = threshold_for_coverage(conf_val, cfg["eval"]["referral_target_coverage"])

    results = {
        "n_test": int(len(y_test)),
        "performance": perf,
        "bootstrap_95ci": ci,
        "majority_floor": floor,
        "margin_above_floor": margins,
        "error_direction": error_direction(y_test, pred, ref),
        "referable_auc": referable_auc(y_test, probs, ref),
        "temperature": T,
        "calibration": {"ece_uncalibrated": ece(confidence(probs_raw, pred), correct),
                        "ece_temperature_scaled": ece(conf, correct)},
        "referral_curve": referral_curve(y_test, pred, conf, cfg["eval"]["coverage_grid"], K, ref),
        "referral_rule": {"target_coverage": cfg["eval"]["referral_target_coverage"],
                          **apply_referral_rule(y_test, pred, conf, tau, K, ref)},
        "validation": {"qwk": qwk(y_val, pred_val, K), "accuracy": float((pred_val == y_val).mean()),
                       "note": "used for early stopping and temperature, so optimistic; for model choice only"},
    }
    if quality_test is not None and quality_train is not None:
        qt = np.asarray(quality_train, float)
        cuts = np.nanquantile(qt, [1 / 3, 2 / 3])
        results["quality_strata_sharpness"] = quality_strata(y_test, pred, conf, quality_test, cuts, K)
    if grey_test is not None:
        # reported only; no threshold, epoch or model choice reads this
        results["grey_surround_subgroup"] = subgroup_report(y_test, pred, grey_test, groups_test, K, ref, B, seed)
    arrays = {"pred": pred, "probs": probs, "conf": conf}
    return results, arrays


def subgroup_markdown(s: dict) -> list[str]:
    def cell(c):   # blank when the subset is below min_n
        return "" if c is None else f"{c['point']:.3f} ({c['lo']:.3f} to {c['hi']:.3f})"

    lines = ["", "## Grey camera surround subgroup (measurement only; nothing is chosen from it)",
             f"Cells with fewer than {s['min_n']} images show the count and no metric.", "",
             "| Subset | n | Grades 0/1/2/3/4 | Accuracy (95% CI) | QWK (95% CI) |", "|---|---|---|---|---|"]
    for name, label in (("in_group", "Grey surround"), ("out_group", "Black surround")):
        v = s["subsets"][name]
        lines.append(f"| {label} | {v['n']} | {'/'.join(str(c) for c in v['grade_mix'].values())} | "
                     f"{cell(v['accuracy'])} | {cell(v['qwk'])} |")
    d = s["difference_in_minus_out"]
    lines.append(f"| Difference (grey minus black) | | | {cell(d['accuracy'])} | {cell(d['qwk'])} |")
    lines += ["", "Share predicted referable within each true grade (separates a surround effect from grade mix):", "",
              "| True grade | n grey | Grey | n black | Black |", "|---|---|---|---|---|"]
    def share(v):
        return "" if v is None else f"{v:.3f}"

    for row in s["predicted_referable_by_true_grade"]:
        lines.append(f"| {row['true_grade']} | {row['n_in_group']} | {share(row['share_predicted_referable_in_group'])} | "
                     f"{row['n_out_group']} | {share(row['share_predicted_referable_out_group'])} |")
    return lines


def results_markdown(r: dict) -> str:
    p, ci, fl, m, e = (r["performance"], r["bootstrap_95ci"], r["majority_floor"],
                       r["margin_above_floor"], r["error_direction"])
    cal = r["calibration"]
    lines = [
        f"# Test results (n = {r['n_test']})", "",
        "| Metric | Model | 95% CI | Majority-class floor | Margin |", "|---|---|---|---|---|",
    ]
    for k, name in [("qwk", "Quadratic weighted kappa"), ("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]:
        lines.append(f"| {name} | {p[k]:.3f} | {ci[k]['lo']:.3f} to {ci[k]['hi']:.3f} | {fl[k]:.3f} | {m[k]:+.3f} |")
    lines += ["", "| Grade | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for k, v in p["per_class"].items():
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    lines += ["", "## Error direction",
              f"- Errors: {e['n_errors']}; under-graded {e['share_of_errors_under']:.1%}, "
              f"over-graded {e['share_of_errors_over']:.1%}; two or more grades out: {e['errors_two_or_more_grades']}",
              f"- Referable DR (grade >= 2): sensitivity {e['referable_sensitivity']:.3f}, "
              f"specificity {e['referable_specificity']:.3f}, AUROC {r['referable_auc']:.3f}, "
              f"missed referable cases {e['missed_referable']}",
              "", "## Calibration",
              f"- Temperature (fitted on validation): {r['temperature']:.3f}",
              f"- ECE uncalibrated {cal['ece_uncalibrated']['ece']:.3f}; "
              f"temperature-scaled {cal['ece_temperature_scaled']['ece']:.3f}",
              "", "## Referral curve",
              "| Coverage | Referred | Accuracy | QWK | Missed referable caught | Random-referral floor |",
              "|---|---|---|---|---|---|"]
    rr = r["referral_rule"]
    lines_rule = ["", "## Referral rule (threshold fixed on validation)",
                  f"- Refer if confidence < {rr['threshold']:.3f} (target coverage {rr['target_coverage']:.0%}); "
                  f"achieved coverage on test {rr['achieved_coverage']:.1%}, {rr['n_referred']} referred",
                  f"- Accuracy on automatically decided cases {rr['accuracy_kept']:.3f}, QWK {rr['qwk_kept']:.3f}; "
                  f"accuracy on referred cases {rr['accuracy_referred']:.3f}",
                  f"- Missed referable cases sent to review: {rr['missed_referable_caught_by_review']} of "
                  f"{rr['missed_referable_total']} (random referral of the same number: "
                  f"{rr['caught_if_referred_at_random']:.1f})"]
    if "quality_strata_sharpness" in r:
        qs = r["quality_strata_sharpness"]
        lines_rule += ["", "## Performance by image sharpness (band edges from training images)",
                       "Sharpness falls with grade, so compare the grade mix before reading band accuracy.", "",
                       "| Band | n | Grades 0/1/2/3/4 | Accuracy | QWK | Mean confidence |",
                       "|---|---|---|---|---|---|"]
        lines_rule += [f"| {b['band']} | {b['n']} | {'/'.join(str(v) for v in b['grade_mix'].values())} | "
                       f"{b['accuracy']:.3f} | {b['qwk']:.3f} | {b['mean_confidence']:.3f} |" for b in qs["bands"]]
        lines_rule.append(f"- Spearman, sharpness vs confidence {qs['spearman_quality_vs_confidence']:.3f}; "
                          f"sharpness vs correct {qs['spearman_quality_vs_correct']:.3f}")
    if "grey_surround_subgroup" in r:
        lines_rule += subgroup_markdown(r["grey_surround_subgroup"])
    for row in r["referral_curve"]:
        lines.append(f"| {row['coverage']:.2f} | {row['n_referred']} | {row['accuracy']:.3f} | {row['qwk']:.3f} | "
                     f"{row['missed_referable_caught_by_review']} | {row['caught_if_referred_at_random']:.1f} |")
    lines += lines_rule
    lines += ["", f"Validation QWK (for model choice only): {r['validation']['qwk']:.3f}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--quality", action="store_true", help="also run the image-quality sensitivity probe")
    args = ap.parse_args()

    from . import plots
    from .inference import infer, load_manifest, load_run

    run_dir = resolve(args.run)
    cfg, net = load_run(run_dir)
    df = load_manifest(cfg)
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"inferring val ({len(va)}) and test ({len(te)}) ...")
    lv, lt = infer(net, va, cfg), infer(net, te, cfg)
    np.savez(run_dir / "logits.npz", val=lv, test=lt, val_ids=va["image_id"].to_numpy(),
             test_ids=te["image_id"].to_numpy())

    has_q = "q_sharpness" in te.columns
    res, arr = analyse(te["grade"].to_numpy(), lt, va["grade"].to_numpy(), lv, tr["grade"].to_numpy(),
                       te["group"].to_numpy(), cfg,
                       quality_test=te["q_sharpness"].to_numpy() if has_q else None,
                       quality_train=tr["q_sharpness"].to_numpy() if has_q else None,
                       grey_test=te["grey_surround"].astype(bool).to_numpy() if "grey_surround" in te.columns
                       else None)

    K = cfg["data"]["num_classes"]
    out = te[["image_id", "grade"]].copy()
    out["pred"], out["confidence"] = arr["pred"], arr["conf"]
    for k in range(K):
        out[f"p{k}"] = arr["probs"][:, k]
    out.to_csv(run_dir / "predictions_test.csv", index=False)

    fig = run_dir / "figures"
    plots.confusion(res["performance"]["confusion"], fig / "confusion_matrix.png")
    plots.reliability(res["calibration"]["ece_uncalibrated"], res["calibration"]["ece_temperature_scaled"],
                      fig / "reliability.png")
    plots.referral(res["referral_curve"], fig / "referral_curve.png")

    if args.quality:
        rows = []
        head, T = cfg["model"]["head"], res["temperature"]
        conditions = [(f"blur {s}", s, 1.0) for s in cfg["eval"]["quality_blur_sigmas"]]
        conditions += [(f"dark gamma {g}", 0.0, g) for g in cfg["eval"]["quality_gammas"] if g != 1.0]
        for name, s, g in conditions:
            lz = lt if (s == 0 and g == 1.0) else infer(net, te, cfg, blur_sigma=s, gamma=g)
            p = outputs_to_pred(lz, head)
            c = confidence(outputs_to_probs(lz, head, T), p)
            rows.append({"condition": name, "qwk": qwk(te["grade"].to_numpy(), p, K),
                         "accuracy": float((p == te["grade"].to_numpy()).mean()),
                         "mean_confidence": float(c.mean())})
            print(f"  {name:16s} QWK {rows[-1]['qwk']:.3f}  mean confidence {rows[-1]['mean_confidence']:.3f}")
        res["quality_probe"] = rows
        plots.quality(rows, fig / "quality_probe.png")

    save_json(res, run_dir / "metrics.json")
    md = results_markdown(res)
    (run_dir / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
