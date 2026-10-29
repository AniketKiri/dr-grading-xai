"""Reading the MMRDR annotation CSV and mapping its columns to internal names."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import resolve

CANDIDATES = {
    "image_id": ["image_id", "imageid", "id", "image", "img", "img_id", "name", "image_name",
                 "filename", "file_name", "file"],
    "grade": ["dr_grade", "drgrade", "grade", "dr", "dr_level", "level", "label", "severity",
              "diagnosis"],
}


def find_label_csv(cfg: dict) -> Path:
    root = resolve(cfg["paths"]["data_root"])
    if cfg["paths"].get("label_csv"):
        return resolve(cfg["paths"]["label_csv"])
    csvs = sorted(root.glob("*.csv"))
    if len(csvs) != 1:
        raise RuntimeError(f"expected exactly one CSV in {root}, found {[c.name for c in csvs]}; "
                           "set paths.label_csv in the config")
    return csvs[0]


def read_labels(cfg: dict) -> pd.DataFrame:
    path = find_label_csv(cfg)
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def detect_columns(df: pd.DataFrame, configured: dict) -> dict:
    """Configured names win. Otherwise take the first candidate present (case-insensitive)."""
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for key in ("image_id", "grade", "eye", "patient_id", "split"):
        name = configured.get(key)
        if name:
            if name not in df.columns:
                raise KeyError(f"configured column {key}={name!r} not in CSV columns {list(df.columns)}")
            out[key] = name
            continue
        out[key] = None
        for cand in CANDIDATES.get(key, []):
            if cand in lower:
                out[key] = lower[cand]
                break
    for key in ("image_id", "grade"):
        if out[key] is None:
            raise KeyError(f"could not find a {key} column in {list(df.columns)}; set columns.{key}")
    return out


def normalise_id(x) -> str:
    """Strip whitespace and any image extension so ids compare against file stems."""
    s = str(x).strip()
    p = Path(s)
    return p.stem if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"} else s
