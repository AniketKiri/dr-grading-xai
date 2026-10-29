"""Temperature scaling, expected calibration error and the referral (selective prediction) curve."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import qwk
from .ordinal import outputs_to_probs


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


def fit_temperature(logits: np.ndarray, y: np.ndarray, head: str) -> float:
    """Single scalar T minimising validation NLL. Fitted on validation, evaluated on test."""
    res = minimize_scalar(lambda lt: nll(outputs_to_probs(logits, head, float(np.exp(lt))), y),
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def confidence(probs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Probability the model assigns to the grade it actually predicted."""
    return probs[np.arange(len(pred)), pred]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    """Expected calibration error with equal-width bins; also returns per-bin data for plotting."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            total += m.mean() * gap
            bins.append({"lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return {"ece": float(total), "bins": bins}


def referral_curve(y, pred, conf, coverages, K: int, referable_min: int) -> list[dict]:
    """Keep the most confident fraction of cases, refer the rest for human review.

    A useful uncertainty signal makes retained-case performance rise as coverage falls,
    and should send a disproportionate share of the model's referable-DR misses to review.
    """
    y, pred, conf = map(np.asarray, (y, pred, conf))
    order = np.argsort(-conf, kind="stable")
    true_ref = y >= referable_min
    missed = true_ref & (pred < referable_min)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        keep = np.zeros(len(y), bool)
        keep[order[:k]] = True
        yk, pk = y[keep], pred[keep]
        rows.append({
            "coverage": float(c), "n_kept": int(k), "n_referred": int(len(y) - k),
            "accuracy": float((yk == pk).mean()),
            "qwk": qwk(yk, pk, K) if len(np.unique(yk)) > 1 else float("nan"),
            "missed_referable_kept": int(missed[keep].sum()),
            "missed_referable_caught_by_review": int(missed[~keep].sum()),
            # floor: referring the same number of cases at random catches this many misses on average
            "caught_if_referred_at_random": float(missed.sum() * (len(y) - k) / len(y)),
        })
    return rows


def threshold_for_coverage(conf_val: np.ndarray, coverage: float) -> float:
    """Confidence cut-off chosen on validation so that `coverage` of cases are decided automatically.
    Choosing it on validation, not test, keeps the test estimate of the deployed rule honest."""
    return float(np.quantile(np.asarray(conf_val), 1.0 - coverage))


def apply_referral_rule(y, pred, conf, tau: float, K: int, referable_min: int) -> dict:
    """Evaluate the fixed rule 'refer if confidence < tau' on held-out data."""
    y, pred, conf = map(np.asarray, (y, pred, conf))
    keep = conf >= tau
    missed = (y >= referable_min) & (pred < referable_min)
    yk, pk = y[keep], pred[keep]
    return {
        "threshold": float(tau),
        "achieved_coverage": float(keep.mean()),
        "n_referred": int((~keep).sum()),
        "accuracy_kept": float((yk == pk).mean()) if keep.any() else float("nan"),
        "qwk_kept": qwk(yk, pk, K) if len(np.unique(yk)) > 1 else float("nan"),
        "accuracy_referred": float((y[~keep] == pred[~keep]).mean()) if (~keep).any() else float("nan"),
        "missed_referable_total": int(missed.sum()),
        "missed_referable_caught_by_review": int(missed[~keep].sum()),
        "caught_if_referred_at_random": float(missed.sum() * (~keep).mean()),
    }
