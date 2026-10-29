"""Step 0. Inspect the raw dataset before trusting any assumption about it.

    python -m drxai.inspect_data

Prints the folder layout, the annotation CSV's columns, the grade distribution, id prefixes
(which may encode an official split), and whether every labelled id has exactly one image.
Nothing is written. Paste the output into the config review before running prepare.
"""
from __future__ import annotations

import re
from collections import Counter

from .config import apply_overrides, base_parser, load_config, resolve
from .data import index_images
from .labels import detect_columns, find_label_csv, normalise_id, read_labels


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    root = resolve(cfg["paths"]["data_root"])
    print(f"data_root: {root}  exists={root.exists()}")
    if not root.exists():
        raise SystemExit("data_root not found; download MMRDR and set paths.data_root (see README)")

    print("\n== top-level contents")
    for p in sorted(root.iterdir()):
        n = sum(1 for _ in p.rglob("*")) if p.is_dir() else None
        print(f"  {'[dir] ' if p.is_dir() else '      '}{p.name}" + (f"  ({n} entries)" if n is not None else ""))

    csv = find_label_csv(cfg)
    df = read_labels(cfg)
    print(f"\n== label file: {csv.name}  rows={len(df)}")
    print(df.dtypes.to_string())
    print("\nfirst rows:")
    print(df.head(8).to_string())

    cols = detect_columns(df, cfg["columns"])
    print(f"\n== column mapping used: {cols}")
    for key, name in cols.items():
        if name and key != "image_id":
            print(f"\n{key} ({name}) value counts:")
            print(df[name].value_counts(dropna=False).sort_index().to_string())

    ids = df[cols["image_id"]].map(normalise_id)
    print(f"\nunique ids: {ids.nunique()} of {len(ids)} rows"
          + ("" if ids.is_unique else "  <-- DUPLICATE IDS: one image may carry two labels"))
    prefixes = Counter(re.match(r"[A-Za-z]*", i).group(0) for i in ids)
    print(f"id prefixes (may encode the official split): {dict(prefixes)}")

    others = [c for c in df.columns if c not in cols.values()]
    if others:
        print(f"\nunmapped columns (check for lesion, quality or patient fields): {others}")

    img_dir = root / cfg["paths"]["image_dir"]
    index = index_images(img_dir)
    print(f"\n== images in {img_dir.name}/: {len(index)}")
    missing = [i for i in ids if i not in index]
    orphans = set(index) - set(ids)
    print(f"labelled ids without an image: {len(missing)}  e.g. {missing[:5]}")
    print(f"images without a label: {len(orphans)}  e.g. {sorted(orphans)[:5]}")

    print("\nNext: confirm the mapping and split rule in the config, then run python -m drxai.prepare")


if __name__ == "__main__":
    main()
