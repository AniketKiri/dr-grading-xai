"""Step 4. Grad-CAM explanations, a randomisation sanity check, and agreement with lesion annotations.

    python -m drxai.explain --run runs/resnet50_corn

Requires evaluate to have been run (it reads predictions_test.csv). Writes figures/gradcam_gallery.png,
figures/sanity_check.png, explain_summary.json and, if lesion annotations are configured,
lesion_agreement.csv.
"""
from __future__ import annotations

import argparse
import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

from .config import cache_dir, resolve, save_json
from .data import fov_mask
from .datasets import eval_transform
from .inference import device, load_manifest, load_run
from .lesions import LesionSource, energy_enrichment, geometry, pointing_game
from .metrics import bootstrap_ci
from .model import corn_expected_grade, gradcam_layer
from .plots import SHORT


class GradCAM:
    """Selvaraju et al. (2017). Channel weights are spatially averaged gradients of the target score.

    Target: for the ordinal head, the expected grade E[y] = sum_k P(y > k), so the map shows
    regions that push the predicted severity up. For softmax, the logit of the predicted grade.
    """

    def __init__(self, net, layer, head: str):
        self.net, self.head = net, head
        self.acts = self.grads = None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.net.zero_grad(set_to_none=True)
        out = self.net(x)
        if self.head == "corn":
            target = corn_expected_grade(out)
        else:
            target = out.gather(1, out.argmax(1, keepdim=True)).squeeze(1)
        target.sum().backward()   # per-sample gradients are independent in eval mode
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
        mx = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
        return cam / np.where(mx > 0, mx, 1)


def load_images(df: pd.DataFrame, cfg: dict):
    tf = eval_transform(cfg["data"]["img_size"])
    size = cfg["data"]["img_size"]
    raw, tens = [], []
    for p in df["cache_path"]:
        img = Image.open(cache_dir(cfg) / p).convert("RGB")
        raw.append(np.asarray(img.resize((size, size), Image.BICUBIC)))
        tens.append(tf(img))
    return raw, torch.stack(tens)


def cams_for(df: pd.DataFrame, cam: GradCAM, cfg: dict, batch: int = 16):
    raws, maps = [], []
    for s in range(0, len(df), batch):
        r, x = load_images(df.iloc[s:s + batch], cfg)
        raws += r
        maps.append(cam(x.to(device())))
    return raws, (np.concatenate(maps) if maps else np.zeros((0,)))


def randomised_copy(net, backbone: str):
    """Model-parameter randomisation test (Adebayo et al., 2018): re-initialise the explained block
    and the head. If the map barely changes, it reflects image structure, not what the model learned."""
    rnd = copy.deepcopy(net)
    torch.manual_seed(0)
    for m in list(gradcam_layer(rnd, backbone).modules()) + list(rnd.modules())[-3:]:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return rnd.eval()


def select_gallery(pred: pd.DataFrame, n: int, ref: int) -> list[tuple[str, pd.DataFrame]]:
    p = pred.sort_values("confidence", ascending=False)
    groups = [
        ("Correct, referable", p[(p.grade >= ref) & (p.pred == p.grade)]),
        ("Correct, no DR", p[(p.grade == 0) & (p.pred == 0)]),
        ("Missed referable (confident)", p[(p.grade >= ref) & (p.pred < ref)]),
        ("Over-graded", p[p.pred > p.grade]),
    ]
    return [(name, g.head(n)) for name, g in groups if len(g)]


def draw_gallery(items, path):
    """items: list of (group name, rows DataFrame, images, heatmaps, lesion masks or None)."""
    rows = sum(len(g) for _, g, *_ in items)
    fig, axes = plt.subplots(rows, 2, figsize=(5.0, 2.5 * rows), squeeze=False)
    r = 0
    for name, g, raws, cams, lesions in items:
        for (_, row), img, cam, les in zip(g.iterrows(), raws, cams, lesions):
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"{name}\n{row['image_id']}", fontsize=7)
            axes[r, 1].imshow(img)
            axes[r, 1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            if les is not None:
                axes[r, 1].contour(les.astype(float), levels=[0.5], colors="white", linewidths=0.6)
            axes[r, 1].set_title(f"ref {SHORT[int(row['grade'])]} | pred {SHORT[int(row['pred'])]}\n"
                                 f"confidence {row['confidence']:.2f}", fontsize=7)
            for a in axes[r]:
                a.axis("off")
            r += 1
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def select_cases(pred: pd.DataFrame, tau: float, ref: int) -> list[tuple[str, pd.Series]]:
    """Three cases that show what the referral rule can and cannot do."""
    auto = pred[pred.confidence >= tau]
    referred = pred[pred.confidence < tau]
    picks = [
        ("Not flagged, correct",auto[(auto.grade >= ref) & (auto.pred == auto.grade)]
         .sort_values("confidence", ascending=False)),
        ("Sent for review by low confidence", referred[referred.pred != referred.grade].sort_values("confidence")),
        ("Confident error the rule cannot catch", auto[(auto.grade >= ref) & (auto.pred < ref)]
         .sort_values("confidence", ascending=False)),
    ]
    return [(name, df.iloc[0]) for name, df in picks if len(df)]


def draw_case_cards(cases, raws, cams, lesions, tau: float, K: int, ref: int, path):
    """Clinician-facing layout that keeps prediction, explanation and judgement visibly separate."""
    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 3.3 * len(cases)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.1, 1.6]})
    for r, ((title, row), img, cam, les) in enumerate(zip(cases, raws, cams, lesions)):
        probs = np.array([row[f"p{k}"] for k in range(K)])
        g, p, c = int(row["grade"]), int(row["pred"]), float(row["confidence"])
        # "no flag" rather than "decide automatically": an unflagged eye still gets a routine examination
        action = "no flag: routine examination" if c >= tau else "flag for extended optometrist review"
        # Triage turns on referable vs not, which the top grade alone can hide when mass is split.
        p_ref = float(probs[ref:].sum())
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{title}\n{row['image_id']}", fontsize=8)
        axes[r, 1].imshow(img)
        axes[r, 1].imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        if les is not None:
            axes[r, 1].contour(les.astype(float), levels=[0.5], colors="white", linewidths=0.6)
        axes[r, 1].set_title("Grad-CAM (white: annotated lesions)" if les is not None else "Grad-CAM", fontsize=8)
        for a in axes[r, :2]:
            a.axis("off")
        bars = axes[r, 2].barh(range(K), probs, color=["#4B2FB5" if k == p else "#C9C3E6" for k in range(K)])
        axes[r, 2].set_yticks(range(K), SHORT[:K], fontsize=7)
        axes[r, 2].invert_yaxis()
        axes[r, 2].set_xlim(0, 1)
        axes[r, 2].set_xlabel("Calibrated probability", fontsize=7)
        axes[r, 2].set_title("Grade distribution", fontsize=8)
        txt = (f"1  MODEL PREDICTION\n   {SHORT[p]}  (confidence {c:.2f}; threshold {tau:.2f})\n"
               f"   P(referable DR, grade >= {ref}) = {p_ref:.2f}\n"
               f"   Action under the referral rule:\n   {action}\n\n"
               f"2  EXPLANATION\n   Regions that raise the predicted severity. Plausible location\n"
               f"   does not prove the model recognised a lesion.\n\n"
               f"3  CLINICAL JUDGEMENT\n   Reference grade: {SHORT[g]}. Decision remains with the\n"
               f"   optometrist or ophthalmologist reviewing the image.")
        axes[r, 3].text(0.0, 0.95, txt, va="top", ha="left", fontsize=7.5, family="monospace",
                        transform=axes[r, 3].transAxes)
        axes[r, 3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-agreement", type=int, default=None, help="cap images used for lesion agreement")
    args = ap.parse_args()
    run_dir = resolve(args.run)
    cfg, net = load_run(run_dir)
    head, backbone = cfg["model"]["head"], cfg["model"]["backbone"]
    ref, size = cfg["data"]["referable_min_grade"], cfg["data"]["img_size"]
    ex = cfg["explain"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    man = load_manifest(cfg).set_index("image_id")
    pred = pd.read_csv(run_dir / "predictions_test.csv", dtype={"image_id": str})
    pred = pred.join(man.drop(columns=["grade"]), on="image_id")
    lesions = LesionSource(ex["lesions"], size, cfg["data"]["cache_size"])
    rnd = randomised_copy(net, backbone)   # copy before any hooks are attached
    cam = GradCAM(net, gradcam_layer(net, backbone), head)
    summary: dict = {"target": "expected grade" if head == "corn" else "predicted-class logit"}

    # ---- gallery
    items = []
    for name, g in select_gallery(pred, ex["n_per_group"], ref):
        raws, maps = cams_for(g, cam, cfg)
        les = [lesions.mask(r["image_id"], geometry(r)) if lesions.enabled else None for _, r in g.iterrows()]
        items.append((name, g, raws, list(maps), les))
    draw_gallery(items, fig_dir / "gradcam_gallery.png")

    # ---- case cards: prediction, explanation and judgement kept separate (brief section 3.3)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    tau = metrics["referral_rule"]["threshold"]
    cases = select_cases(pred, tau, ref)
    if cases:
        cdf = pd.DataFrame([row for _, row in cases])
        raws, maps = cams_for(cdf, cam, cfg)
        les = [lesions.mask(r["image_id"], geometry(r)) if lesions.enabled else None for _, r in cdf.iterrows()]
        draw_case_cards(cases, raws, list(maps), les, tau, cfg["data"]["num_classes"], ref,
                        fig_dir / "case_cards.png")

    # ---- randomisation sanity check
    sub = pred.sample(min(64, len(pred)), random_state=cfg["seed"])
    _, m_trained = cams_for(sub, cam, cfg)
    _, m_rand = cams_for(sub, GradCAM(rnd, gradcam_layer(rnd, backbone), head), cfg)
    small = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((32, 32))).ravel()
    rhos = [spearmanr(small(a), small(b))[0] for a, b in zip(m_trained, m_rand)]
    summary["sanity_check"] = {"n": len(rhos), "median_spearman_trained_vs_randomised": float(np.nanmedian(rhos)),
                               "note": "values near 1 would mean the map does not depend on learned weights"}
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(rhos, bins=20, color="#4B2FB5")
    ax.set_xlabel("Spearman rho, trained vs randomised Grad-CAM")
    ax.set_ylabel("Images")
    ax.set_title("Model-randomisation sanity check", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "sanity_check.png", dpi=160)
    plt.close(fig)

    # ---- lesion agreement
    if lesions.enabled:
        rows = []
        cand = pred if args.max_agreement is None else pred.head(args.max_agreement)
        masks = {r["image_id"]: lesions.mask(r["image_id"], geometry(r)) for _, r in cand.iterrows()}
        ann = cand[cand["image_id"].map(lambda i: masks[i] is not None)]
        raws, maps = cams_for(ann, cam, cfg)
        for (_, r), raw, m in zip(ann.iterrows(), raws, maps):
            les, fov = masks[r["image_id"]], fov_mask(raw)
            pg = pointing_game(m, les, fov, ex["pointing_tolerance_px"])
            rows.append({"image_id": r["image_id"], "grade": r["grade"], "pred": r["pred"],
                         "correct": r["grade"] == r["pred"], "group": r["group"], "hit": pg["hit"],
                         "chance": pg["chance"], "enrichment": energy_enrichment(m, les, fov)})
        la = pd.DataFrame(rows)
        la.to_csv(run_dir / "lesion_agreement.csv", index=False)
        if len(la):
            hit, ch = la["hit"].to_numpy(float), la["chance"].to_numpy()
            diff = bootstrap_ci(lambda i: float(hit[i].mean() - ch[i].mean()), la["group"].to_numpy(),
                                cfg["eval"]["bootstrap"], cfg["seed"])
            summary["lesion_agreement"] = {
                "n_annotated_test_images": len(la),
                "pointing_hit_rate": float(hit.mean()),
                "chance_hit_rate": float(ch.mean()),
                "hit_minus_chance_95ci": diff,
                "median_energy_enrichment": float(la["enrichment"].median()),
                "hit_rate_correct": float(la.loc[la.correct, "hit"].mean()) if la.correct.any() else None,
                "hit_rate_incorrect": float(la.loc[~la.correct, "hit"].mean()) if (~la.correct).any() else None,
            }
    else:
        summary["lesion_agreement"] = "not configured: set explain.lesions in the config"

    save_json(summary, run_dir / "explain_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
