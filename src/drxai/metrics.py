"""Performance, error-direction and floor metrics, plus a group-aware bootstrap. NumPy only."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def qwk(y, pred, K: int) -> float:
    return float(cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(K))))


def classification_metrics(y, pred, K: int) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0)),
        "qwk": qwk(y, pred, K),
        "per_class": {int(k): {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]),
                               "support": int(s[k])} for k in range(K)},
        "confusion": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
    }


def error_direction(y, pred, referable_min: int) -> dict:
    """Under-grading is the clinically dangerous direction: it delays referral."""
    y, pred = np.asarray(y), np.asarray(pred)
    err = pred != y
    d = pred - y
    true_ref, pred_ref = y >= referable_min, pred >= referable_min
    n_err = int(err.sum())
    return {
        "n_errors": n_err,
        "share_of_errors_under": float((d < 0).sum() / n_err) if n_err else 0.0,
        "share_of_errors_over": float((d > 0).sum() / n_err) if n_err else 0.0,
        "errors_two_or_more_grades": int((np.abs(d) >= 2).sum()),
        "missed_referable": int((true_ref & ~pred_ref).sum()),
        "referable_sensitivity": float(pred_ref[true_ref].mean()) if true_ref.any() else float("nan"),
        "referable_specificity": float((~pred_ref[~true_ref]).mean()) if (~true_ref).any() else float("nan"),
    }


def referable_auc(y, probs, referable_min: int) -> float:
    y_ref = (np.asarray(y) >= referable_min).astype(int)
    if y_ref.min() == y_ref.max():
        return float("nan")
    return float(roc_auc_score(y_ref, probs[:, referable_min:].sum(axis=1)))


def majority_floor(y_train, y_eval, K: int) -> dict:
    """What a model with no image access achieves by always predicting the commonest training grade.
    Results are reported as margins above this, never as raw scores alone."""
    majority = int(np.bincount(np.asarray(y_train), minlength=K).argmax())
    pred = np.full(len(y_eval), majority)
    m = classification_metrics(y_eval, pred, K)
    return {"majority_grade": majority, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "qwk": m["qwk"]}


def bootstrap_ci(stat, groups, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI resampling *groups* with replacement (images within a group move together).

    stat: callable(index_array) -> float
    groups: group label per item (near-duplicate cluster, or patient if available)
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(groups)))
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        vals[b] = stat(idx)
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


def quality_strata(y, pred, conf, quality, cuts, K: int) -> dict:
    """Performance and confidence within bands of an image-quality measure.

    `cuts` come from the training images, so band boundaries are fixed before looking at test.
    The question is twofold: does performance fall on poorer images, and does confidence fall with it?
    A model that is less accurate but no less confident on poor images is the dangerous case.
    """
    from scipy.stats import spearmanr

    y, pred, conf, q = map(np.asarray, (y, pred, conf, quality))
    band = np.where(np.isfinite(q), np.digitize(np.nan_to_num(q), cuts), -1)   # NaN quality: no band
    names = ["low", "middle", "high"][: len(cuts) + 1]
    rows = []
    for b, name in enumerate(names):
        m = band == b
        if m.sum() == 0:
            continue
        rows.append({"band": name, "n": int(m.sum()), "accuracy": float((y[m] == pred[m]).mean()),
                     "qwk": qwk(y[m], pred[m], K) if len(np.unique(y[m])) > 1 else float("nan"),
                     "mean_confidence": float(conf[m].mean())})
    ok = np.isfinite(q)
    return {"cuts": [float(c) for c in cuts], "bands": rows,
            "spearman_quality_vs_confidence": float(spearmanr(q[ok], conf[ok])[0]),
            "spearman_quality_vs_correct": float(spearmanr(q[ok], (y == pred)[ok])[0])}
