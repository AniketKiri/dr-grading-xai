"""Analyses that use MMRDR's image-level lesion flags. NumPy and scikit-learn only, so all are unit-tested.

MMRDR records, for each image, seven present/absent flags but no lesion locations. That rules out
spatial checks of the heatmaps (pointing game, enrichment) and supports three others instead:
grade-lesion consistency (label quality), a linear probe of lesion concepts in the model's features,
and the lesion make-up of the model's errors.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .metrics import bootstrap_ci

LESIONS = ["microaneurysm", "hard_exudate", "intraretinal_haemorrhage", "venous_beading_irma",
           "neovascularisation", "vitreous_haemorrhage", "retinal_detachment"]


def parse_lesions(series: pd.Series) -> np.ndarray:
    """'[1, 0, 1, 0, 0, 0, 0]' strings -> (N, 7) int array. Refuses anything malformed."""
    rows = [ast.literal_eval(s) if isinstance(s, str) else s for s in series]
    arr = np.asarray(rows, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != len(LESIONS) or not np.isin(arr, (0, 1)).all():
        raise ValueError(f"lesion column must hold {len(LESIONS)} binary flags per image; got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------- 1. grade and lesion consistency
def consistency_flags(grade: np.ndarray, les: np.ndarray) -> pd.DataFrame:
    """Combinations that the ICDR scale makes implausible. These flag images for inspection;
    they are not proof of error, because graders may apply the scale differently."""
    grade = np.asarray(grade)
    any_lesion = les.any(axis=1)
    # MMRDR (Tang et al., 2026, p.4) defines PDR by neovascularisation, vitreous haemorrhage or retinal detachment
    pdr_sign = les[:, 4] | les[:, 5] | les[:, 6]
    beyond_ma = les[:, 1:].any(axis=1)
    return pd.DataFrame({
        "grade0_with_lesion": (grade == 0) & any_lesion,
        "graded_dr_without_lesion": (grade >= 1) & ~any_lesion,
        "mild_with_more_than_microaneurysms": (grade == 1) & beyond_ma,
        "pdr_without_nv_vh_rd": (grade == 4) & ~pdr_sign,
        "nv_vh_rd_below_pdr": (grade < 4) & pdr_sign,
    })


def consistency_summary(flags: pd.DataFrame, split: np.ndarray) -> dict:
    split = np.asarray(split)
    out = {}
    for rule in flags.columns:
        out[rule] = {s: int(flags.loc[split == s, rule].sum()) for s in ("train", "val", "test")}
        out[rule]["total"] = int(flags[rule].sum())
    out["any_rule"] = {"total": int(flags.any(axis=1).sum()), "share": float(flags.any(axis=1).mean())}
    return out


# ---------------------------------------------------------------- 2. lesion probe
def fit_probe(Xtr, ytr, Xva, yva, Cs=(0.01, 0.1, 1.0)):
    """Logistic regression on standardised features. C chosen on validation AUROC; test is not seen."""
    scaler = StandardScaler().fit(Xtr)
    a, b = scaler.transform(Xtr), scaler.transform(Xva)
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(a, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(b)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return {"val_auroc": float(best[0]), "C": best[1], "model": best[2], "scaler": scaler}


def probe_lesions(feats: dict, les: dict, groups_test, B: int, seed: int, min_pos: int = 20) -> list[dict]:
    """For each lesion: test AUROC of a probe on these features. feats/les are dicts keyed train/val/test.
    Lesions with too few positives in any split are skipped, not reported with a meaningless interval."""
    rows = []
    for k, name in enumerate(LESIONS):
        pos = {s: int(les[s][:, k].sum()) for s in ("train", "val", "test")}
        neg = {s: int(len(les[s]) - pos[s]) for s in pos}
        if min(min(pos.values()), min(neg.values())) < min_pos:
            rows.append({"lesion": name, "skipped": True, "positives": pos})
            continue
        p = fit_probe(feats["train"], les["train"][:, k], feats["val"], les["val"][:, k])
        score = p["model"].predict_proba(p["scaler"].transform(feats["test"]))[:, 1]
        y = les["test"][:, k]

        def auc(i, y=y, score=score):
            return roc_auc_score(y[i], score[i]) if 0 < y[i].sum() < len(i) else np.nan

        ci = bootstrap_ci(auc, groups_test, B, seed)
        rows.append({"lesion": name, "skipped": False, "positives": pos, "C": p["C"],
                     "val_auroc": p["val_auroc"], "test_auroc": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                     "_score": score})
    return rows


def probe_difference(rows_a: list[dict], rows_b: list[dict], les_test, groups_test, B: int, seed: int) -> list[dict]:
    """Paired test AUROC difference (a minus b) per lesion, resampling the same groups for both."""
    out = []
    for ra, rb in zip(rows_a, rows_b):
        if ra["skipped"] or rb["skipped"]:
            continue
        k = LESIONS.index(ra["lesion"])
        y, sa, sb = les_test[:, k], ra["_score"], rb["_score"]

        def d(i):
            if not 0 < y[i].sum() < len(i):
                return np.nan
            return roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i])

        ci = bootstrap_ci(d, groups_test, B, seed)
        out.append({"lesion": ra["lesion"], "difference": ci["point"], "lo": ci["lo"], "hi": ci["hi"],
                    "interval_excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0)})
    return out


# ---------------------------------------------------------------- 3. errors by lesion
def errors_by_lesion(grade, pred, les: np.ndarray, referable_min: int) -> pd.DataFrame:
    """Among truly referable images: how often each lesion is present, and the miss rate when it is.
    A lesion whose presence comes with a high miss rate points at what the model fails to use."""
    grade, pred = np.asarray(grade), np.asarray(pred)
    ref = grade >= referable_min
    missed = ref & (pred < referable_min)
    rows = []
    for k, name in enumerate(LESIONS):
        has = les[:, k].astype(bool) & ref
        rows.append({"lesion": name, "referable_with_lesion": int(has.sum()),
                     "missed_with_lesion": int((has & missed).sum()),
                     "miss_rate_when_present": float((has & missed).sum() / has.sum()) if has.any() else np.nan,
                     "share_of_misses_with_lesion": float((les[:, k].astype(bool) & missed).sum() / missed.sum())
                     if missed.any() else np.nan})
    df = pd.DataFrame(rows)
    df.attrs["n_referable"], df.attrs["n_missed"] = int(ref.sum()), int(missed.sum())
    return df
