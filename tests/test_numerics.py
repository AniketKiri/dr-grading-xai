"""Every number in the report passes through these functions, so each is checked against a value
worked out by hand. Two scaling bugs in earlier analysis code produced plausible-looking numbers;
these tests exist so that cannot happen silently here."""
import numpy as np
import pytest
from PIL import Image

from drxai.calibration import ece, fit_temperature, referral_curve
from drxai.data import (connected_groups, crop_to_square, fingerprint, index_images, near_duplicate_pairs,
                        resolve_ids, surround_threshold)
from drxai.evaluate import analyse, results_markdown
from drxai.lesions import energy_enrichment, pointing_game
from drxai.metrics import bootstrap_ci, error_direction, majority_floor, qwk
from drxai.ordinal import corn_class_probs, corn_predict, outputs_to_probs, softmax

K = 5


# ---------------------------------------------------------------- ordinal head
def test_corn_probs_are_a_distribution():
    z = np.random.default_rng(0).normal(0, 3, (200, K - 1))
    p = corn_class_probs(z)
    assert p.shape == (200, K)
    assert np.all(p >= 0)
    assert np.allclose(p.sum(1), 1)


def test_corn_hand_example():
    # sigmoid(+inf)=1, sigmoid(-inf)=0  -> P(y>0)=1, P(y>1)=1, P(y>2)=0 -> grade 2 with certainty
    z = np.array([[50.0, 50.0, -50.0, -50.0]])
    assert corn_predict(z)[0] == 2
    assert np.allclose(corn_class_probs(z)[0], [0, 0, 1, 0, 0], atol=1e-9)
    # all thresholds at 0 -> S = 0.5, 0.25, 0.125, 0.0625 -> median: none exceed 0.5 -> grade 0
    z0 = np.zeros((1, 4))
    assert corn_predict(z0)[0] == 0
    assert np.allclose(corn_class_probs(z0)[0], [0.5, 0.25, 0.125, 0.0625, 0.0625])


# ---------------------------------------------------------------- metrics
def test_qwk_hand_value():
    assert qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], K) == pytest.approx(1.0)
    # y=[0,1], pred=[1,0], 2-class quadratic kappa: observed disagreement equals chance -> -1
    assert qwk([0, 1], [1, 0], 2) == pytest.approx(-1.0)


def test_majority_floor_has_zero_kappa():
    f = majority_floor([0, 0, 0, 1, 2], [0, 1, 2, 3, 4, 0], K)
    assert f["majority_grade"] == 0
    assert f["qwk"] == pytest.approx(0.0)
    assert f["accuracy"] == pytest.approx(2 / 6)


def test_error_direction_hand_example():
    y = np.array([0, 2, 3, 4, 1])
    p = np.array([0, 1, 3, 2, 2])
    e = error_direction(y, p, referable_min=2)
    assert e["n_errors"] == 3
    assert e["share_of_errors_under"] == pytest.approx(2 / 3)   # 2->1 and 4->2
    assert e["errors_two_or_more_grades"] == 1                  # 4->2
    assert e["missed_referable"] == 1                           # 2->1 crosses below referable
    assert e["referable_sensitivity"] == pytest.approx(2 / 3)
    assert e["referable_specificity"] == pytest.approx(1 / 2)   # grade 1 called 2


def test_bootstrap_moves_groups_together():
    vals = np.array([1.0, 1.0, 0.0, 0.0])
    ci = bootstrap_ci(lambda i: vals[i].mean(), ["a", "a", "b", "b"], B=500, seed=0)
    assert ci["point"] == pytest.approx(0.5)
    # with only two whole groups the resampled mean can only be 0, 0.5 or 1
    assert ci["lo"] in (0.0, 0.5) and ci["hi"] in (0.5, 1.0)


# ---------------------------------------------------------------- calibration
def test_temperature_recovers_known_overconfidence():
    rng = np.random.default_rng(1)
    true = rng.normal(0, 1.5, (4000, K))
    y = np.array([rng.choice(K, p=p) for p in softmax(true)])
    T = fit_temperature(true * 3.0, y, "ce")
    assert T == pytest.approx(3.0, rel=0.15)


def test_ece_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.2, 1.0, 20000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    assert ece(conf, correct)["ece"] < 0.02
    assert ece(np.full(1000, 0.95), np.r_[np.ones(500), np.zeros(500)])["ece"] == pytest.approx(0.45)


def test_referral_curve_bookkeeping():
    y = np.array([3, 3, 0, 0, 2, 0])
    p = np.array([0, 3, 0, 0, 0, 1])          # two missed referable cases: idx 0 and 4
    c = np.array([0.1, 0.9, 0.9, 0.8, 0.2, 0.7])
    rows = referral_curve(y, p, c, [1.0, 0.5], K, 2)
    assert rows[0]["n_referred"] == 0 and rows[0]["missed_referable_caught_by_review"] == 0
    assert rows[1]["n_referred"] == 3
    assert rows[1]["missed_referable_caught_by_review"] == 2   # both misses were least confident
    assert rows[1]["caught_if_referred_at_random"] == pytest.approx(1.0)


# ---------------------------------------------------------------- explanation agreement
def test_pointing_game_and_chance():
    cam = np.zeros((100, 100))
    cam[10, 10] = 1
    les = np.zeros((100, 100), bool)
    les[5:15, 5:15] = True
    fov = np.ones((100, 100), bool)
    r = pointing_game(cam, les, fov, tol_px=0)
    assert r["hit"] is True
    assert r["chance"] == pytest.approx(100 / 10000)
    cam2 = np.zeros((100, 100))
    cam2[80, 80] = 1
    assert pointing_game(cam2, les, fov, 0)["hit"] is False


def test_uniform_heatmap_has_enrichment_one():
    les = np.zeros((50, 50), bool)
    les[:10] = True
    assert energy_enrichment(np.ones((50, 50)), les, np.ones((50, 50), bool)) == pytest.approx(1.0)


# ---------------------------------------------------------------- data integrity
def test_crop_geometry_maps_points_and_masks_consistently():
    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[50:250, 100:300] = 120          # field of view: x 100..299, y 50..249
    rgb[100, 150] = 255                 # a marked pixel
    out, g = crop_to_square(Image.fromarray(rgb), 100)
    assert (g.crop_x0, g.crop_y0, g.crop_x1, g.crop_y1) == (100, 50, 300, 250)
    x, y = g.to_cached(150, 100)
    assert (float(x), float(y)) == pytest.approx((25.0, 25.0))
    mask = np.zeros((300, 400), bool)
    mask[100:110, 150:160] = True
    w = g.warp_mask(mask, 100)
    ys, xs = np.nonzero(w)
    assert ys.min() == 25 and xs.min() == 25


def test_grey_surround_is_blacked_out_and_cropped_like_its_black_twin():
    from PIL import ImageDraw
    black = Image.new("RGB", (400, 300))
    d = ImageDraw.Draw(black)
    d.ellipse([60, 20, 340, 280], fill=(170, 80, 40))
    d.ellipse([190, 140, 205, 155], fill=(5, 5, 5))        # dark spot inside the disc: retina, not surround
    a = np.asarray(black).copy()
    grey = a.copy()
    grey[a.max(axis=2) == 0] = 20                          # same photograph, grey camera surround
    assert surround_threshold(a) == (10, False)
    assert surround_threshold(grey) == (30, True)
    out_b, g_b = crop_to_square(Image.fromarray(a), 100)
    out_g, g_g = crop_to_square(Image.fromarray(grey), 100)
    assert g_g.as_dict() == g_b.as_dict()
    assert np.array_equal(np.asarray(out_g), np.asarray(out_b))
    assert np.asarray(out_g)[0, 0].tolist() == [0, 0, 0]
    # a thin bright frame line round the edge must not shield the surround from correction
    framed = grey.copy()
    framed[0], framed[-1], framed[:, 0], framed[:, -1] = 100, 100, 100, 100
    out_f, g_f = crop_to_square(Image.fromarray(framed), 100)
    assert g_f.as_dict() == g_b.as_dict()
    assert np.array_equal(np.asarray(out_f), np.asarray(out_b))
    # a white canvas around a black ring is not a camera surround: fixed threshold, no correction
    canvas = a.copy()
    canvas[:15], canvas[-15:] = 255, 255
    assert surround_threshold(canvas) == (10, False)


def _fake_fundus(rng, size=400):
    """Shared dark border and bright disc, like every real fundus photograph; random vessels on top."""
    from PIL import ImageDraw
    im = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(im)
    d.ellipse([20, 20, size - 20, size - 20], fill=(170, 80, 40))
    for _ in range(12):
        pts = [tuple(int(v) for v in rng.integers(60, size - 60, 2)) for _ in range(4)]
        d.line(pts, fill=(110, 30, 20), width=int(rng.integers(2, 5)))
    return im


def test_fingerprint_separates_lookalike_fundus_images_but_catches_copies():
    import io
    rng = np.random.default_rng(3)
    imgs = [_fake_fundus(rng) for _ in range(30)]
    buf = io.BytesIO()
    imgs[0].resize((520, 520)).save(buf, "JPEG", quality=85)   # same photo, resized and re-encoded
    copy = Image.open(io.BytesIO(buf.getvalue())).resize((400, 400))
    fps = np.stack([fingerprint(i) for i in imgs + [copy]])
    pairs = near_duplicate_pairs(fps, 0.9)
    assert [tuple(p[:2].astype(int)) for p in pairs] == [(0, 30)]
    g = connected_groups(len(fps), pairs)
    assert g[0] == g[30] and len(set(g)) == 30


def test_resolver_refuses_ambiguity_and_missing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "a" / "x1.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "b" / "x2.png")
    idx = index_images(tmp_path)
    assert resolve_ids(["x1", "x2"], idx)[0].stem == "x1"
    with pytest.raises(RuntimeError):
        resolve_ids(["x3"], idx)
    Image.new("RGB", (4, 4)).save(tmp_path / "b" / "x1.jpg")
    with pytest.raises(RuntimeError):
        index_images(tmp_path)


# ---------------------------------------------------------------- the whole test-set analysis
def test_analyse_margins_are_model_minus_floor():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, K, n)
    logits = np.stack([np.where(y > k, 3.0, -3.0) for k in range(K - 1)], 1) + rng.normal(0, 1, (n, K - 1))
    cfg = {"data": {"num_classes": K, "referable_min_grade": 2}, "model": {"head": "corn"},
           "eval": {"bootstrap": 200, "coverage_grid": [1.0, 0.8], "referral_target_coverage": 0.8}, "seed": 0}
    q = rng.uniform(0, 1, n)
    grey = rng.uniform(size=n) < 0.1
    r, arr = analyse(y, logits, y, logits, y, np.arange(n), cfg, quality_test=q, quality_train=q, grey_test=grey)
    md = results_markdown(r)   # formatting of every section, including the new subgroup and grade-mix tables
    assert "Grey camera surround subgroup" in md and "Grades 0/1/2/3/4" in md and "Severe disease" in md
    s = r["severe_sensitivity"]
    assert s["n_severe"] == int((y >= 3).sum()) and s["lo"] <= s["point"] <= s["hi"]
    assert r["grey_surround_subgroup"]["subsets"]["in_group"]["n"] == int(grey.sum())
    assert r["referral_rule"]["achieved_coverage"] == pytest.approx(0.8, abs=0.01)   # val == test here
    assert sum(b["n"] for b in r["quality_strata_sharpness"]["bands"]) == n
    for k in ("qwk", "accuracy", "macro_f1"):
        assert r["margin_above_floor"][k] == pytest.approx(r["performance"][k] - r["majority_floor"][k])
        assert r["bootstrap_95ci"][k]["lo"] <= r["performance"][k] <= r["bootstrap_95ci"][k]["hi"]
    assert np.allclose(arr["probs"].sum(1), 1)
    assert r["performance"]["qwk"] > 0.8


def test_subgroup_report_hand_case_and_blank_small_cells():
    from drxai.metrics import subgroup_report
    # grey: grade 1 x10 (5 right, 5 called grade 2), grade 2 x6 (all right)          -> accuracy 11/16
    # black: grade 0 x20 right, grade 1 x10 right, grade 2 x5 called 1, grade 3 x5 right -> accuracy 35/40
    y = np.r_[[1] * 10, [2] * 6, [0] * 20, [1] * 10, [2] * 5, [3] * 5]
    pred = np.r_[[1] * 5, [2] * 5, [2] * 6, [0] * 20, [1] * 10, [1] * 5, [3] * 5]
    grey = np.r_[np.ones(16, bool), np.zeros(40, bool)]
    s = subgroup_report(y, pred, grey, np.arange(len(y)), K, 2, B=300, seed=0)
    gi, go = s["subsets"]["in_group"], s["subsets"]["out_group"]
    assert (gi["n"], go["n"]) == (16, 40)
    assert gi["grade_mix"] == {0: 0, 1: 10, 2: 6, 3: 0, 4: 0} and go["grade_mix"] == {0: 20, 1: 10, 2: 5, 3: 5, 4: 0}
    assert gi["accuracy"]["point"] == pytest.approx(11 / 16) and go["accuracy"]["point"] == pytest.approx(35 / 40)
    d = s["difference_in_minus_out"]["accuracy"]
    assert d["point"] == pytest.approx(11 / 16 - 35 / 40) and d["lo"] <= d["point"] <= d["hi"]
    rows = {r["true_grade"]: r for r in s["predicted_referable_by_true_grade"]}
    assert rows[1]["share_predicted_referable_in_group"] == pytest.approx(0.5)
    assert rows[1]["share_predicted_referable_out_group"] == pytest.approx(0.0)
    assert rows[2]["n_in_group"] == 6 and rows[2]["share_predicted_referable_in_group"] is None    # below 10
    assert rows[2]["n_out_group"] == 5 and rows[2]["share_predicted_referable_out_group"] is None
    # a subgroup below 10 keeps its count but gets no metric and no difference
    tiny = np.zeros(len(y), bool)
    tiny[:5] = True
    t = subgroup_report(y, pred, tiny, np.arange(len(y)), K, 2, B=50, seed=0)
    assert t["subsets"]["in_group"]["n"] == 5 and t["subsets"]["in_group"]["accuracy"] is None
    assert t["difference_in_minus_out"]["accuracy"] is None and t["difference_in_minus_out"]["qwk"] is None


def test_severe_sensitivity_hand_case():
    from drxai.metrics import severe_sensitivity
    y = np.array([3, 3, 4, 4, 2, 0])
    pred = np.array([3, 2, 4, 3, 3, 0])     # severe: 3->3, 3->2 (missed), 4->4, 4->3; the 2->3 does not count
    assert severe_sensitivity(y, pred) == pytest.approx(3 / 4)
    assert np.isnan(severe_sensitivity(np.array([0, 1, 2]), np.array([0, 1, 2])))


def test_quality_strata_reports_grade_mix_per_band():
    from drxai.metrics import quality_strata
    y = np.array([0, 0, 0, 1, 4, 4, 2, 3, 4])
    q = np.array([9, 8, 7, 5, 1, 2, 5, 1, 9.0])          # cuts 3 and 6: low = q < 3, middle 3..6, high >= 6
    pred = y.copy()
    pred[0] = 1                                            # one error, so the Spearman terms are defined
    r = quality_strata(y, pred, np.linspace(0.5, 0.9, 9), q, [3.0, 6.0], K)
    mix = {b["band"]: b["grade_mix"] for b in r["bands"]}
    assert mix["low"] == {0: 0, 1: 0, 2: 0, 3: 1, 4: 2}
    assert mix["middle"] == {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}
    assert mix["high"] == {0: 3, 1: 0, 2: 0, 3: 0, 4: 1}


# ---------------------------------------------------------------- protocol: choices made on validation only
def test_referral_threshold_from_validation_and_rule_bookkeeping():
    from drxai.calibration import apply_referral_rule, threshold_for_coverage
    conf_val = np.linspace(0.0, 1.0, 101)
    tau = threshold_for_coverage(conf_val, 0.8)
    assert tau == pytest.approx(0.2)
    y = np.array([2, 2, 0, 0, 3])
    p = np.array([0, 2, 0, 1, 3])      # one missed referable (idx 0)
    c = np.array([0.1, 0.9, 0.9, 0.15, 0.95])
    r = apply_referral_rule(y, p, c, tau, K, 2)
    assert r["n_referred"] == 2 and r["achieved_coverage"] == pytest.approx(0.6)
    assert r["accuracy_kept"] == pytest.approx(1.0)
    assert r["missed_referable_caught_by_review"] == 1
    assert r["caught_if_referred_at_random"] == pytest.approx(0.4)


def test_quality_bands_use_given_cuts_and_skip_nan():
    from drxai.metrics import quality_strata
    y = np.array([0, 1, 2, 0, 1, 2, 0])
    p = np.array([0, 1, 2, 1, 1, 0, 0])
    c = np.array([.9, .9, .9, .5, .6, .4, .7])
    q = np.array([10., 11., 12., 1., 2., 3., np.nan])
    r = quality_strata(y, p, c, q, cuts=[5.0, 9.0], K=K)
    names = {b["band"]: b for b in r["bands"]}
    assert names["low"]["n"] == 3 and names["high"]["n"] == 3     # the NaN image sits in no band
    assert names["high"]["accuracy"] == pytest.approx(1.0)
    assert names["low"]["accuracy"] == pytest.approx(1 / 3)


def test_sharpness_drops_when_image_is_blurred():
    from PIL import ImageFilter
    from drxai.data import image_quality
    img = _fake_fundus(np.random.default_rng(5))
    sharp = image_quality(img)["q_sharpness"]
    blurred = image_quality(img.filter(ImageFilter.GaussianBlur(4)))["q_sharpness"]
    assert blurred < 0.5 * sharp


def test_paired_difference_is_zero_for_identical_models():
    from drxai.compare import paired_difference
    rng = np.random.default_rng(6)
    y = rng.integers(0, K, 200)
    p = np.where(rng.uniform(size=200) < 0.7, y, rng.integers(0, K, 200))
    d = paired_difference(y, p, p, np.arange(200), K, 200, 0)
    assert d["point"] == 0 and d["lo"] == 0 and d["hi"] == 0
