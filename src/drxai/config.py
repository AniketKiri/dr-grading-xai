"""Configuration loading, path resolution and seeding."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | os.PathLike | None) -> Path | None:
    """Resolve a config path: absolute paths unchanged, relative ones against the repo root."""
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data-root", default=None, help="override paths.data_root")
    return ap


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "data_root", None):
        cfg["paths"]["data_root"] = args.data_root
    return cfg


def cache_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["cache_dir"])


def manifest_path(cfg: dict) -> Path:
    return cache_dir(cfg) / "manifest.csv"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
