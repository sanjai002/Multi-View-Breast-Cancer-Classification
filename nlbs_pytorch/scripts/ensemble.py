#!/usr/bin/env python3
"""
Multi-backbone ensemble: ResNet50 / DenseNet121 / EfficientNetV2-S / ConvNeXt-Tiny.

Trains (or reuses) one model per backbone, then combines test-set probabilities:
  soft_vote           : unweighted mean of P(cancer)
  weighted_soft_vote  : mean weighted by each model's validation AUC  ***RECOMMENDED***
  stacking            : logistic-regression meta-learner on stacked probs

Weighted soft voting is the best default here: robust, needs no extra held-out
data (unlike stacking, which can overfit a meta-learner on a small set) and lets
stronger models contribute more (unlike plain soft voting).

Usage:
    python scripts/ensemble.py --train      # train all backbones then ensemble
    python scripts/ensemble.py              # reuse existing run_*/checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders import build_loaders
from src.models.fusion import build_model
from src.losses.losses import build_loss
from src.engine.trainer import Trainer
from src.utils.metrics import compute_metrics, save_eval_plots


def get_probs(cfg, backbone, device, train_loader_fn, val_loader, test_loader, cw, do_train):
    cfg = load_config(str(ROOT / "configs/config.yaml"))
    cfg.model.backbone = backbone
    out = Path(cfg.paths.output_root) / f"run_{backbone}_{cfg.model.fusion}"
    model = build_model(cfg.model)
    trainer = Trainer(model, build_loss(cfg.loss, cw.to(device)), cfg, device, out)
    if do_train or not trainer.best_path.exists():
        trainer.fit(train_loader_fn, val_loader,
                    {int(e): int(s) for e, s in cfg.preprocess.progressive_sizes}
                    if cfg.preprocess.get("progressive_sizes") else None)
    ckpt = torch.load(trainer.best_path, map_location=device, weights_only=False)
    use_ema = ckpt.get("ema") is not None
    (trainer.ema.ema if use_ema else model).load_state_dict(
        ckpt["ema"] if use_ema else ckpt["model"])
    vy, vp, _ = trainer.evaluate(val_loader, use_ema=use_ema, tta=True)
    ty, tp, tpid = trainer.evaluate(test_loader, use_ema=use_ema, tta=True)
    val_auc = compute_metrics(vy, vp)["roc_auc"]
    return val_auc, ty, tp, tpid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()
    train_loader_fn, val_loader, test_loader, _, cw = build_loaders(cfg)

    probs, weights, ty = [], [], None
    for bb in cfg.ensemble.backbones:
        print(f"\n=== ensemble member: {bb} ===")
        val_auc, ty, tp, _ = get_probs(cfg, bb, device, train_loader_fn,
                                       val_loader, test_loader, cw, args.train)
        probs.append(tp); weights.append(max(val_auc, 1e-3))
        print(f"  {bb}: val_auc={val_auc:.4f} test_auc={compute_metrics(ty, tp)['roc_auc']:.4f}")

    P = np.stack(probs, axis=1)                       # (N, M)
    method = cfg.ensemble.method
    if method == "soft_vote":
        ens = P.mean(1)
    elif method == "stacking":
        meta = LogisticRegression(max_iter=1000).fit(P, ty)
        ens = meta.predict_proba(P)[:, 1]
    else:                                             # weighted_soft_vote
        w = np.array(weights) / np.sum(weights)
        ens = (P * w).sum(1)

    m = compute_metrics(ty, ens)
    out = Path(cfg.paths.output_root) / "ensemble"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"method": method, "metrics": m,
               "members": list(cfg.ensemble.backbones), "weights": weights},
              open(out / "ensemble_metrics.json", "w"), indent=2)
    save_eval_plots(ty, ens, out)
    print(f"\n=== ENSEMBLE ({method}) ===")
    for k in ("accuracy", "f1", "roc_auc", "recall_sensitivity", "specificity"):
        print(f"  {k:20s}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
