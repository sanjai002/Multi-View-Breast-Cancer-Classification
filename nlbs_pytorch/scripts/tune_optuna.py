#!/usr/bin/env python3
"""
Optuna hyperparameter search (maximizes validation ROC-AUC).

Tunes: learning rate, weight decay, batch size, dropout, focal gamma, image
size, optimizer, scheduler. Each trial runs a short training budget.

Usage:
    python scripts/tune_optuna.py --trials 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders import build_loaders
from src.models.fusion import build_model
from src.losses.losses import build_loss
from src.engine.trainer import Trainer
from src.utils.metrics import compute_metrics


def objective(trial, base_cfg, device):
    cfg = load_config(str(ROOT / "configs/config.yaml"))     # fresh copy
    cfg.train.base_lr = trial.suggest_float("base_lr", 1e-5, 3e-3, log=True)
    cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    cfg.train.batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.6)
    cfg.loss.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    cfg.preprocess.img_size = trial.suggest_categorical("img_size", [256, 384, 512])
    cfg.train.optimizer = trial.suggest_categorical("optimizer", ["adamw", "sgd"])
    cfg.train.scheduler = trial.suggest_categorical("scheduler", ["onecycle", "cosine_warmup"])
    cfg.preprocess.progressive_sizes = []
    cfg.train.epochs = 6                                       # short budget/trial
    cfg.train.freeze_backbone_epochs = 1

    train_loader_fn, val_loader, _, _, cw = build_loaders(cfg)
    model = build_model(cfg.model)
    trainer = Trainer(model, build_loss(cfg.loss, cw.to(device)), cfg, device,
                      Path(cfg.paths.output_root) / f"optuna/trial_{trial.number}")
    trainer.fit(train_loader_fn, val_loader, None)
    y, p, _ = trainer.evaluate(val_loader, use_ema=True, tta=False)
    return compute_metrics(y, p)["roc_auc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()
    n = args.trials or cfg.optuna.n_trials
    study = optuna.create_study(direction="maximize", study_name=cfg.optuna.study_name)
    study.optimize(lambda t: objective(t, cfg, device), n_trials=n,
                   timeout=cfg.optuna.timeout_min * 60)
    print("best value (val AUC):", study.best_value)
    print("best params:", study.best_params)
    out = Path(cfg.paths.output_root) / "optuna"
    out.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out / "optuna_trials.csv", index=False)


if __name__ == "__main__":
    main()
