"""Step 6. Use MMRDR's image-level lesion flags: label consistency, lesion probes, errors by lesion.

    python -m drxai.lesions_report --run runs/resnet50_corn

Requires evaluate to have been run (reads predictions_test.csv). Writes lesions_report.json,
lesions_report.md, lesion_consistency.csv and figures/lesion_probe.png inside the run folder.

The probe asks whether each lesion can be read off the model's penultimate features by a linear
classifier. The same probe on an ImageNet-only network (never trained on fundus images) is the floor:
a gain over it means DR training made that lesion more decodable. Probe weights and C are fitted on
train and validation only; test is measured once.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import cache_dir, resolve, save_json
from .datasets import FundusDataset, eval_transform
from .inference import device, eval_workers, load_manifest, load_run
from .labels import detect_columns, normalise_id, read_labels
from .lesion_analysis import (LESIONS, consistency_flags, consistency_summary, errors_by_lesion, parse_lesions,
                              probe_difference, probe_lesions)
from .model import build_model


@torch.no_grad()
def penultimate_features(net, df: pd.DataFrame, cfg: dict, batch_size: int = 32) -> np.ndarray:
    """Input to the final linear layer, i.e. the pooled image representation the grade is read from."""
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    store = {}
    h = last.register_forward_hook(lambda m, i, o: store.__setitem__("x", i[0].detach()))
    ds = FundusDataset(df, cache_dir(cfg), eval_transform(cfg["data"]["img_size"]))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=eval_workers(cfg))
    dev = device()
    out = np.zeros((len(ds), last.in_features), dtype=np.float32)
    try:
        for x, _, idx in dl:
            net(x.to(dev))
            out[idx.numpy()] = store["x"].float().cpu().numpy()
    finally:
        h.remove()
    return out


def lesion_table(cfg: dict, lesion_col: str) -> pd.DataFrame:
    raw = read_labels(cfg)
    cols = detect_columns(raw, cfg["columns"])
    if lesion_col not in raw.columns:
        raise KeyError(f"lesion column {lesion_col!r} not in {list(raw.columns)}")
    flags = parse_lesions(raw[lesion_col])
    df = pd.DataFrame(flags, columns=LESIONS)
    df.insert(0, "image_id", raw[cols["image_id"]].map(normalise_id).values)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lesion-col", default="lesion")
    args = ap.parse_args()
    run_dir = resolve(args.run)
    cfg, net = load_run(run_dir)
    K, ref = cfg["data"]["num_classes"], cfg["data"]["referable_min_grade"]
    B, seed = cfg["eval"]["bootstrap"], cfg["seed"]

    man = load_manifest(cfg)
    les = lesion_table(cfg, args.lesion_col)
    man = man.merge(les, on="image_id", how="left", validate="one_to_one")
    assert man[LESIONS].notna().all().all(), "an image in the manifest has no lesion flags"
    L = man[LESIONS].to_numpy(int)

    # ---- 1. label consistency (all splits; this is a property of the dataset, not the model)
    flags = consistency_flags(man["grade"].to_numpy(), L)
    summary = {"consistency": consistency_summary(flags, man["split"].to_numpy())}
    bad = man.loc[flags.any(axis=1), ["image_id", "split", "grade"] + LESIONS].copy()
    bad["rules"] = flags[flags.any(axis=1)].apply(lambda r: ";".join(c for c in flags.columns if r[c]), axis=1)
    bad.to_csv(run_dir / "lesion_consistency.csv", index=False)
    print(f"label consistency: {summary['consistency']['any_rule']['total']} images break at least one rule")

    # ---- 2. lesion probes: trained network against an ImageNet-only network of the same architecture
    parts = {s: man[man["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    groups_test = parts["test"]["group"].to_numpy()
    les_by = {s: parts[s][LESIONS].to_numpy(int) for s in parts}
    imagenet = build_model(cfg["model"]["backbone"], cfg["model"]["head"], K, pretrained=True,
                           dropout=cfg["model"]["dropout"]).to(device()).eval()
    probes = {}
    for name, model in [("trained", net), ("imagenet_only", imagenet)]:
        print(f"extracting {name} features ...")
        feats = {s: penultimate_features(model, parts[s], cfg) for s in parts}
        probes[name] = probe_lesions(feats, les_by, groups_test, B, seed)
    diffs = probe_difference(probes["trained"], probes["imagenet_only"], les_by["test"], groups_test, B, seed)
    summary["probe"] = {k: [{kk: vv for kk, vv in r.items() if kk != "_score"} for r in v] for k, v in probes.items()}
    summary["probe_difference_trained_minus_imagenet"] = diffs

    # ---- 3. errors by lesion (test split, primary predictions from evaluate)
    pred = pd.read_csv(run_dir / "predictions_test.csv", dtype={"image_id": str})
    pt = pred.merge(les, on="image_id", how="left", validate="one_to_one")
    ebl = errors_by_lesion(pt["grade"].to_numpy(), pt["pred"].to_numpy(), pt[LESIONS].to_numpy(int), ref)
    summary["errors_by_lesion"] = {"n_referable": ebl.attrs["n_referable"], "n_missed": ebl.attrs["n_missed"],
                                   "rows": ebl.to_dict(orient="records")}

    # ---- figure
    tr = [r for r in probes["trained"] if not r["skipped"]]
    im = {r["lesion"]: r for r in probes["imagenet_only"] if not r["skipped"]}
    if tr:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        x = np.arange(len(tr))
        for off, rows, col, lab in [(-0.2, tr, "#4B2FB5", "DR-trained features"),
                                    (0.2, [im[r["lesion"]] for r in tr], "#B0B0B0", "ImageNet-only features (floor)")]:
            pts = np.array([r["test_auroc"] for r in rows])
            err = np.array([[r["test_auroc"] - r["lo"] for r in rows], [r["hi"] - r["test_auroc"] for r in rows]])
            ax.bar(x + off, pts, 0.38, yerr=err, color=col, label=lab, capsize=2)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.set_xticks(x, [r["lesion"].replace("_", " ") for r in tr], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("Test AUROC (95% CI)")
        ax.set_title("Can each lesion be read from the model's features?", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        (run_dir / "figures").mkdir(exist_ok=True)
        fig.savefig(run_dir / "figures" / "lesion_probe.png", dpi=200)
        plt.close(fig)

    # ---- markdown
    md = ["# Lesion-flag analyses", "", "## Label consistency (ICDR-implied rules; flags for inspection, not proof of error)",
          "| Rule | Train | Val | Test | Total |", "|---|---|---|---|---|"]
    for rule, v in summary["consistency"].items():
        if rule != "any_rule":
            md.append(f"| {rule} | {v['train']} | {v['val']} | {v['test']} | {v['total']} |")
    a = summary["consistency"]["any_rule"]
    md += [f"\nImages breaking at least one rule: {a['total']} ({a['share']:.1%})", "",
           "## Lesion probe (test AUROC, 95% CI; C chosen on validation)",
           "| Lesion | DR-trained | ImageNet-only | Difference | Interval excludes 0 |", "|---|---|---|---|---|"]
    dmap = {d["lesion"]: d for d in diffs}
    for r in probes["trained"]:
        if r["skipped"]:
            md.append(f"| {r['lesion']} | skipped (positives {r['positives']}) | | | |")
            continue
        i, d = im[r["lesion"]], dmap[r["lesion"]]
        md.append(f"| {r['lesion']} | {r['test_auroc']:.3f} ({r['lo']:.3f} to {r['hi']:.3f}) | "
                  f"{i['test_auroc']:.3f} ({i['lo']:.3f} to {i['hi']:.3f}) | {d['difference']:+.3f} "
                  f"({d['lo']:+.3f} to {d['hi']:+.3f}) | {'yes' if d['interval_excludes_zero'] else 'no'} |")
    md += ["", f"## Errors by lesion (test; {ebl.attrs['n_missed']} missed of {ebl.attrs['n_referable']} referable)",
           "| Lesion | Referable with lesion | Missed with lesion | Miss rate when present | Share of misses |",
           "|---|---|---|---|---|"]
    for r in ebl.to_dict(orient="records"):
        md.append(f"| {r['lesion']} | {r['referable_with_lesion']} | {r['missed_with_lesion']} | "
                  f"{r['miss_rate_when_present']:.3f} | {r['share_of_misses_with_lesion']:.3f} |")
    (run_dir / "lesions_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_json(summary, run_dir / "lesions_report.json")
    print("\n".join(md))


if __name__ == "__main__":
    main()
