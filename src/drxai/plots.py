"""Figures for the report. Matplotlib only, no seaborn."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import GRADE_NAMES

SHORT = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion(cm, path: Path, title="Test confusion matrix"):
    cm = np.asarray(cm)
    row = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    K = cm.shape[0]
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]}\n{row[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if row[i, j] > 0.55 else "black")
    ax.set_xticks(range(K), SHORT[:K], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(K), SHORT[:K], fontsize=8)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Reference grade")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row proportion (recall)")
    _save(fig, path)


def reliability(before: dict, after: dict, path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Perfect calibration")
    for res, lab, c in [(before, "Uncalibrated", "#B5450A"), (after, "Temperature-scaled", "#4B2FB5")]:
        xs = [b["conf"] for b in res["bins"]]
        ys = [b["acc"] for b in res["bins"]]
        ax.plot(xs, ys, marker="o", ms=4, c=c, label=f"{lab} (ECE {res['ece']:.3f})")
    ax.set_xlabel("Mean confidence in predicted grade")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Reliability on the test split", fontsize=10)
    _save(fig, path)


def referral(rows: list[dict], path: Path):
    cov = [r["coverage"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a1.plot(cov, [r["qwk"] for r in rows], marker="o", c="#4B2FB5", label="QWK")
    a1.plot(cov, [r["accuracy"] for r in rows], marker="s", c="#0B5E4F", label="Accuracy")
    a1.invert_xaxis()
    a1.set_xlabel("Coverage (fraction decided automatically)")
    a1.set_ylabel("Performance on retained cases")
    a1.legend(fontsize=8)
    a1.set_title("Selective prediction", fontsize=10)
    a2.plot(cov, [r["missed_referable_caught_by_review"] for r in rows], marker="o", c="#4B2FB5",
            label="Referred by confidence")
    a2.plot(cov, [r["caught_if_referred_at_random"] for r in rows], ls="--", c="grey",
            label="Referred at random (floor)")
    a2.invert_xaxis()
    a2.set_xlabel("Coverage")
    a2.set_ylabel("Missed referable-DR cases sent to review")
    a2.legend(fontsize=8)
    a2.set_title("Does low confidence find the dangerous errors?", fontsize=10)
    _save(fig, path)


def quality(rows: list[dict], path: Path):
    labels = [r["condition"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.2, [r["qwk"] for r in rows], 0.4, color="#4B2FB5", label="QWK")
    ax.bar(x + 0.2, [r["mean_confidence"] for r in rows], 0.4, color="#E0A82E", label="Mean confidence")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Sensitivity to simulated image-quality loss (test split)", fontsize=10)
    _save(fig, path)
