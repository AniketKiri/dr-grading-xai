"""Step 2. Train on the training split, select the epoch on the validation split.

    python -m drxai.train --run-name resnet50_corn
    python -m drxai.train --run-name resnet50_ce --head ce        # softmax baseline

The test split is never loaded here. Model selection uses validation QWK only.
"""
from __future__ import annotations

import math
import platform
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from .config import apply_overrides, base_parser, cache_dir, load_config, resolve, save_json, seed_everything
from .datasets import FundusDataset, class_balanced_sampler, train_transform
from .inference import device, infer, load_manifest
from .model import build_model, loss_fn
from .ordinal import outputs_to_pred


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--head", choices=["corn", "ce"], default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    for key, sect, name in [("head", "model", "head"), ("backbone", "model", "backbone"),
                            ("epochs", "train", "epochs"), ("seed", None, "seed"), ("lr", "train", "lr")]:
        v = getattr(args, key)
        if v is not None:
            (cfg[sect] if sect else cfg)[name] = v
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    run_dir = resolve(cfg["paths"]["runs_dir"]) / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    df = load_manifest(cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    assert not (set(train_df["group"]) & set(val_df["group"])), "a group spans train and val"
    print(f"train {len(train_df)}  val {len(val_df)}  (test not loaded)")

    tc, mc, K = cfg["train"], cfg["model"], cfg["data"]["num_classes"]
    dev = device()
    use_amp = tc["amp"] and dev.type == "cuda"
    net = build_model(mc["backbone"], mc["head"], K, mc["pretrained"], mc["dropout"]).to(dev)
    if dev.type == "cuda":
        net = net.to(memory_format=torch.channels_last)

    ds = FundusDataset(train_df, cache_dir(cfg), train_transform(cfg["data"]["img_size"]))
    sampler = class_balanced_sampler(train_df["grade"].to_numpy(), tc["sampler"], cfg["seed"])
    # Not persistent: the training workers exit before validation starts its own, so at most num_workers
    # worker processes exist at once. On Windows each re-imports torch (~1.4 GB committed), and keeping
    # both sets alive crashed a 16 GB laptop at the end of an epoch.
    dl = DataLoader(ds, batch_size=tc["batch_size"], sampler=sampler, shuffle=sampler is None,
                    num_workers=tc["num_workers"], pin_memory=dev.type == "cuda",
                    persistent_workers=False, drop_last=True)

    opt = torch.optim.AdamW(net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = tc["epochs"] * len(dl)
    warm = tc["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = loss_fn(mc["head"])

    save_json(cfg, run_dir / "config_used.json")
    history, best, bad = [], -np.inf, 0
    for epoch in range(1, tc["epochs"] + 1):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, _ in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if dev.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = crit(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)

        net.eval()
        logits = infer(net, val_df, cfg)
        pred = outputs_to_pred(logits, mc["head"])
        qwk = float(cohen_kappa_score(val_df["grade"], pred, weights="quadratic"))
        acc = float((pred == val_df["grade"].to_numpy()).mean())
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_qwk": qwk, "val_acc": acc,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch:2d}  loss {rec['train_loss']:.4f}  val QWK {qwk:.4f}  acc {acc:.4f}  "
              f"{rec['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if qwk > best:
            best, bad = qwk, 0
            torch.save({"model": net.state_dict(), "epoch": int(epoch), "val_qwk": float(qwk)},
                       run_dir / "checkpoints" / "best.pt")
        else:
            bad += 1
            if bad >= tc["patience"]:
                print(f"early stop: no validation gain for {bad} epochs")
                break

    save_json({"best_val_qwk": best, "best_epoch": int(max(history, key=lambda r: r["val_qwk"])["epoch"]),
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               "python": platform.python_version(), "seed": cfg["seed"],
               "n_train": len(train_df), "n_val": len(val_df)}, run_dir / "train_summary.json")
    print(f"best validation QWK {best:.4f}; run saved to {run_dir}")


if __name__ == "__main__":
    main()
