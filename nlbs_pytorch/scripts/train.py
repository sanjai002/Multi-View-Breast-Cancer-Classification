#!/usr/bin/env python3
"""Train the dual-view NLBS classifier, then evaluate on the held-out test split.

Usage:
    python scripts/train.py                       # uses configs/config.yaml
    python scripts/train.py model.backbone=densenet121 train.epochs=25
    python scripts/train.py --smoke               # tiny fast end-to-end check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders import build_loaders
from src.models.fusion import build_model
from src.losses.losses import build_loss
from src.engine.trainer import Trainer
from src.utils.metrics import compute_metrics, save_eval_plots


def patient_level(y, p, pids):
    """Aggregate breast predictions to patient level (max prob / any-cancer)."""
    df = pd.DataFrame({"pid": pids, "y": y, "p": p})
    g = df.groupby("pid").agg(y=("y", "max"), p=("p", "max"))
    return g["y"].to_numpy(), g["p"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.smoke:                                   # fast path for validation
        cfg.train.epochs = 2
        cfg.train.freeze_backbone_epochs = 1
        cfg.preprocess.img_size = 224
        cfg.preprocess.progressive_sizes = []
        cfg.data.neg_per_pos = 1
        cfg.data.max_train = 48                       # tiny subset -> minutes on CPU
        cfg.data.max_eval = 24
        cfg.data.num_workers = 2
        cfg.model.pretrained = True

    set_seed(cfg.seed)
    device = get_device()
    print(f"[train] device={device} backbone={cfg.model.backbone} fusion={cfg.model.fusion}")
    out = Path(cfg.paths.output_root) / f"run_{cfg.model.backbone}_{cfg.model.fusion}"

    train_loader_fn, val_loader, test_loader, df, cw = build_loaders(cfg)
    model = build_model(cfg.model)
    loss_fn = build_loss(cfg.loss, cw.to(device))
    trainer = Trainer(model, loss_fn, cfg, device, out)

    sched = ({int(e): int(s) for e, s in cfg.preprocess.progressive_sizes}
             if cfg.preprocess.get("progressive_sizes") else None)
    trainer.fit(train_loader_fn, val_loader, sched)

    # Load best checkpoint and evaluate on TEST with TTA.
    ckpt = torch.load(trainer.best_path, map_location=device, weights_only=False)
    use_ema = ckpt.get("ema") is not None
    if use_ema:
        trainer.ema.ema.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    y, p, pids = trainer.evaluate(test_loader, use_ema=use_ema, tta=cfg.train.tta)

    breast_m = compute_metrics(y, p)
    py, pp = patient_level(y, p, pids)
    patient_m = compute_metrics(py, pp)
    result = {"breast_level": breast_m, "patient_level": patient_m,
              "backbone": cfg.model.backbone, "fusion": cfg.model.fusion}

    (out / "reports").mkdir(parents=True, exist_ok=True)
    with open(out / "reports" / "test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    save_eval_plots(y, p, out / "reports")

    print("\n=== TEST (breast-level) ===")
    for k in ("accuracy", "precision", "recall_sensitivity", "specificity",
              "f1", "roc_auc", "avg_precision"):
        print(f"  {k:20s}: {breast_m[k]:.4f}")
    print("=== TEST (patient-level) ===")
    for k in ("accuracy", "f1", "roc_auc"):
        print(f"  {k:20s}: {patient_m[k]:.4f}")
    print(f"\nsaved -> {out/'reports'}")


if __name__ == "__main__":
    main()
