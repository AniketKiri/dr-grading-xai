"""Step 1. Build the cached dataset, the splits and the integrity report.

    python -m drxai.prepare [--workers 4] [--force]

Writes cache/images/<id>.png, cache/manifest.csv, cache/near_duplicates.csv and
cache/integrity_report.json. Fails loudly on any check that would invalidate results.
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .config import (apply_overrides, base_parser, cache_dir, load_config, manifest_path,
                     resolve, save_json, seed_everything)
from .data import (QUALITY_COLS, connected_groups, crop_to_square, fingerprint, image_quality, index_images,
                   near_duplicate_pairs, resolve_ids, surround_threshold)
from .labels import detect_columns, normalise_id, read_labels

GEOM_COLS = ["orig_w", "orig_h", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "pad_x", "pad_y", "scale"]


def _process_one(task):
    src, dst, size, force = task
    dst = Path(dst)
    img = Image.open(src).convert("RGB")
    grey = surround_threshold(np.asarray(img))[1]
    out, geom = crop_to_square(img, size)
    if force or not dst.exists():
        out.save(dst)
    return geom.as_dict(), fingerprint(out), image_quality(out), grey


def _assign_official_split(df: pd.DataFrame, cfg: dict, cols: dict) -> pd.Series | None:
    if cols.get("split"):
        s = df[cols["split"]].astype(str).str.lower().str.strip()
        s = s.replace({"training": "train", "testing": "test", "validation": "val", "valid": "val"})
        return s
    prefix_map = cfg.get("split_from_id_prefix") or {}
    if prefix_map:
        pref = df["image_id"].map(lambda i: re.match(r"[A-Za-z]*", i).group(0))
        unknown = sorted(set(pref) - set(prefix_map))
        if unknown:
            raise RuntimeError(f"id prefixes {unknown} not covered by split_from_id_prefix")
        return pref.map(prefix_map)
    return None


def _grouped_holdout(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of a stratified hold-out in which no group is split across the boundary."""
    n_splits = max(2, int(round(1 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, hold = next(sgkf.split(df, df["grade"], groups=df["group"]))
    mask = np.zeros(len(df), dtype=bool)
    mask[hold] = True
    return mask


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rewrite cached images")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    seed_everything(cfg["seed"])
    K = cfg["data"]["num_classes"]
    checks: dict[str, dict] = {}

    def check(name, ok, detail="", critical=True):
        checks[name] = {"passed": bool(ok), "detail": detail, "critical": critical}
        flag = "PASS" if ok else ("FAIL" if critical else "WARN")
        print(f"  [{flag}] {name}" + (f": {detail}" if detail else ""))

    # ---- labels
    raw = read_labels(cfg)
    cols = detect_columns(raw, cfg["columns"])
    print(f"column mapping: {cols}")
    df = pd.DataFrame({"image_id": raw[cols["image_id"]].map(normalise_id)})
    df["grade"] = pd.to_numeric(raw[cols["grade"]], errors="coerce")
    df["eye"] = raw[cols["eye"]].values if cols.get("eye") else np.nan
    df["patient_id"] = raw[cols["patient_id"]].astype(str).values if cols.get("patient_id") else None

    print("\nintegrity checks")
    n_bad = int(df["grade"].isna().sum())
    check("grades parse as numbers", n_bad == 0, f"{n_bad} unparseable rows dropped", critical=False)
    df = df[df["grade"].notna()].copy()
    df["grade"] = df["grade"].astype(int)
    check("grades within 0..K-1", df["grade"].between(0, K - 1).all(),
          f"observed {sorted(int(g) for g in df['grade'].unique())}")
    check("image ids unique", df["image_id"].is_unique,
          f"{df['image_id'].duplicated().sum()} duplicate ids")

    # ---- strict id -> file resolution
    root = resolve(cfg["paths"]["data_root"])
    index = index_images(root / cfg["paths"]["image_dir"])
    src_paths = resolve_ids(df["image_id"].tolist(), index)
    check("every id resolves to exactly one file with a matching stem", True, f"{len(src_paths)} files")

    # ---- crop, cache, hash
    out_dir = cache_dir(cfg) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["data"]["cache_size"]
    tasks = [(str(p), str(out_dir / f"{i}.png"), size, args.force) for i, p in zip(df["image_id"], src_paths)]
    print(f"\ncaching {len(tasks)} images at {size}px with {args.workers} workers ...")
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = list(ex.map(_process_one, tasks, chunksize=32))
    else:
        results = [_process_one(t) for t in tasks]
    geoms, fps, quals, greys = zip(*results)
    for c in GEOM_COLS:
        df[c] = [g[c] for g in geoms]
    for c in QUALITY_COLS:
        df[c] = [q[c] for q in quals]
    # kept so evaluate can report test performance on these images as a subgroup (measurement only)
    df["grey_surround"] = list(greys)
    qs = df[QUALITY_COLS].describe(percentiles=[0.01, 0.1, 0.5, 0.9]).T
    print("\nimage quality (cached images, inside the field of view)\n" + qs.round(3).to_string())
    low = df["q_sharpness"] < df["q_sharpness"].quantile(0.02)
    df.loc[low, ["image_id", "grade"] + QUALITY_COLS].to_csv(cache_dir(cfg) / "lowest_sharpness_2pct.csv", index=False)
    df["cache_path"] = [f"images/{i}.png" for i in df["image_id"]]

    # ---- near-duplicates: same photograph appearing under two ids
    pairs = near_duplicate_pairs(np.stack(fps), cfg["data"]["near_duplicate_min_corr"])
    dup_group = connected_groups(len(df), pairs)
    if df["patient_id"].notna().all() and cols.get("patient_id"):
        # patients are the true unit; merge duplicate clusters into them as well
        df["group"] = df["patient_id"]
    else:
        df["group"] = [f"g{g}" for g in dup_group]
    pi, pj = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    dup = pd.DataFrame({"id_a": df["image_id"].values[pi], "id_b": df["image_id"].values[pj],
                        "corr": pairs[:, 2].round(4), "grade_a": df["grade"].values[pi],
                        "grade_b": df["grade"].values[pj]})
    dup.to_csv(cache_dir(cfg) / "near_duplicates.csv", index=False)
    disagree = int((dup["grade_a"] != dup["grade_b"]).sum())
    check("near-duplicate pairs found", True, f"{len(dup)} pairs, {disagree} with different grades", critical=False)
    involved = len(set(dup["id_a"]) | set(dup["id_b"]))
    plausible = involved <= 0.05 * len(df)
    check("duplicate detector flags a plausible share of images", plausible,
          f"{involved} of {len(df)} images in a duplicate pair"
          + ("" if plausible else "; above 5% points at the detector, not the data: inspect "
             "near_duplicates.csv and raise data.near_duplicate_min_corr"))

    # ---- splits
    official = _assign_official_split(df, cfg, cols)
    seed = cfg["seed"]
    if official is not None:
        df["split"] = official.values
        split_source = "official"
    else:
        test_mask = _grouped_holdout(df, cfg["data"]["test_fraction"], seed)
        df["split"] = np.where(test_mask, "test", "train")
        split_source = "stratified grouped hold-out (no official split found)"
    check("split source", True, split_source, critical=False)

    # remove training images that are near-copies of a test image; the test set is never altered
    test_ids = set(df.loc[df["split"] == "test", "image_id"])
    train_ids = set(df.loc[df["split"] != "test", "image_id"])
    cross = dup[(dup["id_a"].isin(test_ids) & dup["id_b"].isin(train_ids)) |
                (dup["id_b"].isin(test_ids) & dup["id_a"].isin(train_ids))]
    leaked = set(cross["id_a"]).union(cross["id_b"]) & train_ids
    if leaked and cfg["data"]["drop_train_duplicates_of_test"]:
        df = df[~df["image_id"].isin(leaked)].copy()
        check("train near-copies of test images removed", True, f"{len(leaked)} training images dropped",
              critical=False)
    else:
        check("no train image is a near-copy of a test image", not leaked, f"{len(leaked)} found")

    if "val" not in set(df["split"]):
        pool = df[df["split"] == "train"]
        val_mask = _grouped_holdout(pool, cfg["data"]["val_fraction"], seed + 1)
        df.loc[pool.index[val_mask], "split"] = "val"

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ga = set(df.loc[df["split"] == a, "group"])
        gb = set(df.loc[df["split"] == b, "group"])
        crit = not (split_source == "official" and "test" in (a, b))
        check(f"no group spans {a}/{b}", not (ga & gb), f"{len(ga & gb)} shared groups", critical=crit)
    for s in ("train", "val", "test"):
        present = set(df.loc[df["split"] == s, "grade"])
        check(f"all grades present in {s}", present == set(range(K)), f"present {sorted(present)}",
              critical=False)

    # ---- cached files open at the expected size
    sample = df.sample(min(50, len(df)), random_state=seed)
    sizes_ok = all(Image.open(cache_dir(cfg) / p).size == (size, size) for p in sample["cache_path"])
    check("cached images readable at expected size (50 sampled)", sizes_ok)

    # ---- write and round-trip
    mpath = manifest_path(cfg)
    df.to_csv(mpath, index=False)
    back = pd.read_csv(mpath, dtype={"image_id": str})
    check("manifest round-trips", back["image_id"].tolist() == df["image_id"].tolist()
          and back["grade"].tolist() == df["grade"].tolist())

    table = pd.crosstab(df["grade"], df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    print("\ngrade x split\n" + table.to_string())
    # train and val only: this table is the evidence for the surround correction, so test must not inform it
    dev = df[df["split"] != "test"]
    grey = pd.crosstab(dev["grade"], dev["grey_surround"]).reindex(columns=[False, True], fill_value=0)
    grey.columns = ["black_surround", "grey_surround"]
    grey["grey_share_pct"] = (100 * grey["grey_surround"] / grey.sum(axis=1)).round(1)
    print(f"\ngrey-surround images by grade (train and val only; {int(df['grey_surround'].sum())} corrected "
          f"in all splits)\n" + grey.to_string())
    save_json({"column_mapping": cols, "split_source": split_source, "checks": checks,
               "grade_by_split": table.to_dict(), "grey_surround_by_grade_train_val": grey.to_dict(),
              "n_grey_surround_corrected": int(df["grey_surround"].sum()), "n_near_duplicate_pairs": len(dup),
               "n_near_duplicate_grade_disagreements": disagree,
               "n_train_dropped_as_test_duplicates": len(leaked) if cfg["data"]["drop_train_duplicates_of_test"] else 0},
              cache_dir(cfg) / "integrity_report.json")
    failed = [k for k, v in checks.items() if v["critical"] and not v["passed"]]
    if failed:
        raise SystemExit(f"\nCRITICAL CHECKS FAILED: {failed}")
    print(f"\nmanifest written: {mpath}  ({len(df)} images)")


if __name__ == "__main__":
    main()
