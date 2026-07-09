#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split (breast- and patient-level)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils.seed import set_seed, get_device
from src.data.loaders import build_loaders
from src.models.fusion import build_model
from src.engine.trainer import Trainer
from src.losses.losses import build_loss
from src.utils.metrics import compute_metrics, save_eval_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    device = get_device()
    _, _, test_loader, _, cw = build_loaders(cfg)
    model = build_model(cfg.model)
    trainer = Trainer(model, build_loss(cfg.loss, cw.to(device)), cfg, device,
                      Path(cfg.paths.output_root) / "eval_tmp")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    use_ema = ckpt.get("ema") is not None
    (trainer.ema.ema if use_ema else model).load_state_dict(
        ckpt["ema"] if use_ema else ckpt["model"])

    y, p, pids = trainer.evaluate(test_loader, use_ema=use_ema, tta=cfg.train.tta)
    m = compute_metrics(y, p)
    df = pd.DataFrame({"pid": pids, "y": y, "p": p}).groupby("pid").agg(
        y=("y", "max"), p=("p", "max"))
    pm = compute_metrics(df["y"].to_numpy(), df["p"].to_numpy())
    out = Path(args.checkpoint).parent.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"breast_level": m, "patient_level": pm}, open(out / "test_metrics.json", "w"),
              indent=2)
    save_eval_plots(y, p, out)
    print("breast-level:", {k: round(m[k], 4) for k in
                            ("accuracy", "f1", "roc_auc", "recall_sensitivity", "specificity")})
    print("patient-level:", {k: round(pm[k], 4) for k in ("accuracy", "f1", "roc_auc")})


if __name__ == "__main__":
    main()
