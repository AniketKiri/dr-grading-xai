"""Hand-checked cases for the lesion-flag analyses."""
import numpy as np
import pandas as pd
import pytest

from drxai.lesion_analysis import (LESIONS, consistency_flags, errors_by_lesion, parse_lesions, probe_difference,
                                   probe_lesions)


def test_parse_lesions_accepts_mmrdr_strings_and_rejects_bad_ones():
    arr = parse_lesions(pd.Series(["[0, 0, 0, 0, 0, 0, 0]", "[1, 1, 1, 0, 0, 0, 0]"]))
    assert arr.shape == (2, 7) and arr[1, :3].tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        parse_lesions(pd.Series(["[1, 0]"]))
    with pytest.raises(ValueError):
        parse_lesions(pd.Series(["[2, 0, 0, 0, 0, 0, 0]"]))


def test_consistency_rules_hand_cases():
    les = np.array([[1, 0, 0, 0, 0, 0, 0],    # grade 0 with a microaneurysm      -> grade0_with_lesion
                    [0, 0, 0, 0, 0, 0, 0],    # grade 2 with nothing flagged      -> graded_dr_without_lesion
                    [1, 1, 0, 0, 0, 0, 0],    # grade 1 with exudate              -> mild_with_more_than_microaneurysms
                    [1, 1, 1, 1, 0, 0, 0],    # grade 4 without NV, VH or RD      -> pdr_without_nv_vh_rd
                    [1, 0, 0, 0, 1, 0, 0],    # grade 2 with NV                   -> nv_vh_rd_below_pdr
                    [1, 0, 0, 0, 0, 0, 0],    # grade 1, microaneurysm only       -> consistent
                    [1, 0, 0, 0, 0, 0, 1],    # grade 4 with RD only              -> consistent (RD is a PDR sign)
                    [1, 1, 1, 1, 0, 0, 1]])   # grade 3 with RD                   -> nv_vh_rd_below_pdr
    f = consistency_flags(np.array([0, 2, 1, 4, 2, 1, 4, 3]), les)
    assert list(f.columns) == ["grade0_with_lesion", "graded_dr_without_lesion", "mild_with_more_than_microaneurysms",
                               "pdr_without_nv_vh_rd", "nv_vh_rd_below_pdr"]
    assert f.sum().tolist() == [1, 1, 1, 1, 2]
    assert not f.iloc[5].any() and not f.iloc[6].any()


def test_errors_by_lesion_hand_case():
    grade = np.array([2, 3, 2, 0])
    pred = np.array([0, 3, 1, 0])          # two missed referable: idx 0 and 2
    les = np.zeros((4, 7), int)
    les[[0, 1, 2], 0] = 1                  # microaneurysm in all three referable
    les[0, 3] = 1                          # venous beading only in a missed case
    df = errors_by_lesion(grade, pred, les, 2).set_index("lesion")
    assert df.attrs["n_missed"] == 2 and df.attrs["n_referable"] == 3
    assert df.loc["microaneurysm", "miss_rate_when_present"] == pytest.approx(2 / 3)
    assert df.loc["venous_beading_irma", "share_of_misses_with_lesion"] == pytest.approx(0.5)


def test_probe_finds_an_encoded_lesion_and_not_a_random_one():
    rng = np.random.default_rng(0)
    n = {"train": 600, "val": 200, "test": 300}
    les = {s: (rng.uniform(size=(k, 7)) < 0.3).astype(int) for s, k in n.items()}
    # feature 0 carries lesion 0 (microaneurysm); the rest is noise
    feats = {s: np.c_[les[s][:, 0] * 2.0 + rng.normal(0, 0.5, n[s]), rng.normal(size=(n[s], 9))] for s in n}
    noise = {s: rng.normal(size=(n[s], 10)) for s in n}
    groups = np.arange(n["test"])
    a = probe_lesions(feats, les, groups, B=200, seed=0)
    b = probe_lesions(noise, les, groups, B=200, seed=0)
    assert a[0]["test_auroc"] > 0.9 and abs(b[0]["test_auroc"] - 0.5) < 0.1
    assert abs(a[1]["test_auroc"] - 0.5) < 0.1                    # lesion 1 is not in the features
    d = probe_difference(a, b, les["test"], groups, 200, 0)
    assert d[0]["lesion"] == LESIONS[0] and d[0]["interval_excludes_zero"]
