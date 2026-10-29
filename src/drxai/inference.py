"""Shared run loading and batched inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import cache_dir, manifest_path
from .datasets import FundusDataset, eval_transform
from .model import build_model


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(manifest_path(cfg), dtype={"image_id": str, "group": str})


def load_run(run_dir: Path):
    cfg = json.loads((run_dir / "config_used.json").read_text(encoding="utf-8"))
    m = cfg["model"]
    net = build_model(m["backbone"], m["head"], cfg["data"]["num_classes"], pretrained=False,
                      dropout=m["dropout"])
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    return cfg, net.to(device()).eval()


@torch.no_grad()
def infer(net, df: pd.DataFrame, cfg: dict, blur_sigma=0.0, gamma=1.0, batch_size=32) -> np.ndarray:
    """Raw logits for every row of df, in df order."""
    ds = FundusDataset(df, cache_dir(cfg), eval_transform(cfg["data"]["img_size"]), blur_sigma, gamma)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg["train"]["num_workers"])
    dev = device()
    use_amp = cfg["train"]["amp"] and dev.type == "cuda"
    out = np.zeros((len(ds), net_out_dim(net)), dtype=np.float32)
    for x, _, idx in dl:
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            z = net(x.to(dev, non_blocking=True))
        out[idx.numpy()] = z.float().cpu().numpy()
    return out


def net_out_dim(net) -> int:
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    return last.out_features
