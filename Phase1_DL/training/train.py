"""Training entry point for Phase 1.

Wires together the whole pipeline: patient-level splitting, on-demand data
loading, the dual-branch fusion model, mixed-precision optimisation with SAM,
EMA, differential learning rates, progressive unfreezing, cosine / plateau
scheduling, TensorBoard logging, early stopping and checkpointing.

Run from the project root::

    python -m training.train                     # full run with defaults
    python -m training.train --epochs 2 --limit 40   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from augmentation import (build_eval_transforms, build_train_transforms,
                          mix_criterion, mixup_cutmix)
from config import Config, get_config
from dataset import build_dataloaders
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import compute_metrics, format_metrics
from models.fusion import build_model
from training.callbacks import EMA, SAM, EarlyStopping, ModelCheckpoint
from training.validate import evaluate
from utils import (AverageMeter, amp_settings, autocast_context, build_loss,
                   compute_class_weights, count_parameters, get_device,
                   get_logger, load_metadata, build_patient_table,
                   patient_level_split, save_checkpoint, set_seed)


class Trainer:
    """Owns all training state and runs the fit loop."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.create_dirs()
        self.logger = get_logger("phase1", cfg.paths.log_dir)
        set_seed(cfg.data.seed)
        self.device = get_device()
        self.logger.info("Device: %s", self.device)

        self._setup_data()
        self._setup_model()
        self._setup_optimisation()

        self.writer = SummaryWriter(cfg.paths.tensorboard_dir)
        self.history: list = []                      # per-epoch metric rows
        self.start_epoch = 0                         # overwritten when resuming
        self._epoch_plots_dir = os.path.join(cfg.paths.output_dir, "epoch_plots")
        os.makedirs(self._epoch_plots_dir, exist_ok=True)
        self.use_amp, self.amp_dtype, self.use_scaler = amp_settings(cfg, self.device)
        try:  # torch >= 2.3 preferred API
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)
        except (AttributeError, TypeError):  # older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)
        self.logger.info(
            "AMP=%s dtype=%s scaler=%s | SAM=%s EMA=%s",
            self.use_amp, self.amp_dtype, self.use_scaler,
            cfg.train.use_sam, cfg.train.use_ema,
        )

    # ------------------------------------------------------------------ #
    def _setup_data(self) -> None:
        cfg = self.cfg
        df = load_metadata(cfg, self.logger)
        table = build_patient_table(df, cfg, self.logger)

        # Reuse an existing split manifest if one covers the same patients,
        # rather than re-deriving it. This guarantees the train/val/test patient
        # assignment is IDENTICAL across machines/sessions (e.g. a checkpoint
        # trained locally and resumed on Colab, or vice versa) instead of
        # relying on train_test_split's determinism holding across environments.
        reused = False
        if os.path.isfile(cfg.paths.manifest_csv):
            cached = pd.read_csv(cfg.paths.manifest_csv)
            if "split" in cached.columns and set(cached["Patient_ID"].astype(str)) == set(
                table["Patient_ID"].astype(str)
            ):
                table = cached
                reused = True
                self.logger.info("Reusing existing split manifest: %s", cfg.paths.manifest_csv)
        if not reused:
            table = patient_level_split(table, cfg, self.logger)

        if getattr(cfg, "_limit", None):
            # Subsample patients per split for a fast smoke test.
            table = (
                table.groupby("split", group_keys=False)
                .apply(lambda g: g.head(max(4, cfg._limit // 3)))
                .reset_index(drop=True)
            )
            self.logger.warning("LIMIT active: using %d patients", len(table))

        os.makedirs(os.path.dirname(cfg.paths.manifest_csv), exist_ok=True)
        table.to_csv(cfg.paths.manifest_csv, index=False)
        self.table = table

        self.train_table = table[table["split"] == "train"].reset_index(drop=True)
        self.val_table = table[table["split"] == "val"].reset_index(drop=True)
        self.test_table = table[table["split"] == "test"].reset_index(drop=True)

        self.loaders = build_dataloaders(
            self.train_table, self.val_table, self.test_table, cfg,
            build_train_transforms(cfg), build_eval_transforms(cfg),
        )
        self.class_weights = compute_class_weights(
            self.train_table["label"].values, cfg.data.num_classes,
            power=cfg.train.class_weight_power,
        ).to(self.device)
        self.logger.info("Class weights: %s", self.class_weights.tolist())

    def _setup_model(self) -> None:
        self.model = build_model(self.cfg).to(self.device)
        trainable, total = count_parameters(self.model)
        self.logger.info("Model params: %.2fM trainable / %.2fM total",
                         trainable / 1e6, total / 1e6)
        self.loss_fn = build_loss(self.cfg, self.class_weights).to(self.device)
        self.ema = EMA(self.model, self.cfg.train.ema_decay) if self.cfg.train.use_ema else None

    def _setup_optimisation(self) -> None:
        cfg = self.cfg
        param_groups = self.model.get_param_groups(cfg)
        adam_kwargs = dict(lr=cfg.train.head_lr, weight_decay=cfg.train.weight_decay,
                           betas=(0.9, 0.999), eps=1e-8)
        if cfg.train.use_sam:
            self.optimizer = SAM(param_groups, torch.optim.AdamW,
                                 rho=cfg.train.sam_rho, adaptive=cfg.train.sam_adaptive,
                                 **adam_kwargs)
            self.base_optimizer = self.optimizer.base_optimizer
        else:
            self.optimizer = torch.optim.AdamW(param_groups, **adam_kwargs)
            self.base_optimizer = self.optimizer

        self.base_lrs = [g["lr"] for g in self.base_optimizer.param_groups]
        self.plateau = None
        if cfg.train.scheduler == "plateau":
            self.plateau = ReduceLROnPlateau(
                self.base_optimizer, mode=cfg.train.monitor_mode,
                factor=cfg.train.plateau_factor, patience=cfg.train.plateau_patience,
                min_lr=cfg.train.min_lr,
            )

        self.early_stopping = EarlyStopping(
            cfg.train.early_stopping_patience, cfg.train.monitor_mode,
            cfg.train.early_stopping_min_delta,
        )
        self.checkpoint = ModelCheckpoint(cfg.train.monitor_mode)

    # ------------------------------------------------------------------ #
    def _set_lr(self, epoch: int) -> None:
        """Warm-up then cosine anneal (plateau path only warms up)."""
        cfg = self.cfg
        groups = self.base_optimizer.param_groups
        if epoch < cfg.train.warmup_epochs:
            factor = (epoch + 1) / max(1, cfg.train.warmup_epochs)
            for g, base in zip(groups, self.base_lrs):
                g["lr"] = base * factor
        elif cfg.train.scheduler == "cosine":
            progress = (epoch - cfg.train.warmup_epochs) / max(
                1, cfg.train.epochs - cfg.train.warmup_epochs
            )
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            for g, base in zip(groups, self.base_lrs):
                g["lr"] = cfg.train.min_lr + (base - cfg.train.min_lr) * cos

    def _monitor_key(self) -> str:
        return self.cfg.train.monitor.replace("val_", "").replace("train_", "")

    # ------------------------------------------------------------------ #
    def _clip_grads(self) -> None:
        if self.cfg.train.grad_clip_norm and self.cfg.train.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)

    def _compute_loss(self, views, mask, y_a, y_b, lam):
        out = self.model(views, mask)
        logits = out["logits"].float()
        loss = mix_criterion(self.loss_fn, logits, y_a, y_b, lam)
        return loss, logits

    def _train_step(self, views, mask, labels) -> float:
        cfg = self.cfg
        views, mask, y_a, y_b, lam = mixup_cutmix(views, mask, labels, cfg)

        if cfg.train.use_sam:
            # ---- SAM: ascent step then descent step (no grad scaler) ----
            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                loss, _ = self._compute_loss(views, mask, y_a, y_b, lam)
            loss.backward()
            self._clip_grads()
            self.optimizer.first_step(zero_grad=True)

            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                loss2, _ = self._compute_loss(views, mask, y_a, y_b, lam)
            loss2.backward()
            self._clip_grads()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.use_amp, self.amp_dtype):
                loss, _ = self._compute_loss(views, mask, y_a, y_b, lam)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._clip_grads()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self._clip_grads()
                self.optimizer.step()

        if self.ema is not None:
            self.ema.update(self.model)
        return float(loss.item())

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        changed = self.model.apply_unfreeze_schedule(epoch, self.cfg)
        if changed:
            trainable, total = count_parameters(self.model)
            self.logger.info("Epoch %d: unfroze backbone -> %.2fM trainable params",
                             epoch, trainable / 1e6)
        self._set_lr(epoch)

        meter = AverageMeter()
        loader = self.loaders["train"]
        for step, batch in enumerate(loader):
            views = batch["views"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            loss = self._train_step(views, mask, labels)
            meter.update(loss, n=views.size(0))

            if step % self.cfg.train.log_every_n_steps == 0:
                gstep = epoch * len(loader) + step
                self.writer.add_scalar("train/loss_step", loss, gstep)
                self.writer.add_scalar("train/lr_head",
                                       self.base_optimizer.param_groups[-1]["lr"], gstep)
                self.logger.info("Epoch %d [%d/%d] loss=%.4f",
                                 epoch, step, len(loader), loss)
        self.writer.add_scalar("train/loss_epoch", meter.avg, epoch)
        return meter.avg

    def validate_epoch(self, epoch: int) -> dict:
        # Evaluate with EMA weights when available.
        ctx = self.ema.average_parameters(self.model) if self.ema else _null_ctx()
        with ctx:
            res = evaluate(self.model, self.loaders["val"], self.device, self.cfg,
                           loss_fn=self.loss_fn)
        metrics = compute_metrics(res.y_true, res.y_prob, self.cfg.data.class_names)
        metrics["loss"] = res.loss

        self.writer.add_scalar("val/loss", res.loss, epoch)
        for k in ("accuracy", "macro_f1", "macro_auc", "balanced_accuracy",
                  "macro_sensitivity", "macro_specificity"):
            self.writer.add_scalar(f"val/{k}", metrics[k], epoch)
        self.logger.info("Epoch %d | val_loss=%.4f | %s",
                         epoch, res.loss, format_metrics(metrics))

        # Per-epoch validation confusion matrix (row-normalised) - directly
        # exposes per-class behaviour / imbalance as training progresses.
        try:
            plot_confusion_matrix(
                res.y_true, res.y_pred, self.cfg.data.class_names,
                os.path.join(self._epoch_plots_dir, f"epoch_{epoch:03d}_val_confusion.png"),
                normalize=True, title=f"Val confusion - epoch {epoch}",
            )
        except Exception as exc:  # never let a plot crash training
            self.logger.warning("epoch confusion plot failed: %s", exc)
        return metrics

    # ------------------------------------------------------------------ #
    def _maybe_resume(self) -> None:
        """Resume model/optimizer/EMA/history from checkpoints/last.pth."""
        last = os.path.join(self.cfg.paths.checkpoint_dir, "last.pth")
        if not (self.cfg.train.resume and os.path.isfile(last)):
            return
        ck = torch.load(last, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            try:
                self.base_optimizer.load_state_dict(ck["optimizer_state"])
            except Exception as exc:
                self.logger.warning("could not restore optimizer state: %s", exc)
        if self.ema is not None and ck.get("ema_state"):
            self.ema.load_state_dict(ck["ema_state"])
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        best = ck.get("best_metric", None)
        if best is not None:
            self.checkpoint.best = best
            self.early_stopping.best = best

        # Reload metrics history so curves/CSV continue seamlessly.
        csvp = os.path.join(self.cfg.paths.output_dir, "metrics_history.csv")
        if os.path.isfile(csvp):
            with open(csvp) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                for k, v in list(r.items()):
                    try:
                        r[k] = int(v) if k == "epoch" else float(v)
                    except (ValueError, TypeError):
                        pass
            self.history = [r for r in rows if int(r["epoch"]) < self.start_epoch]
        self.logger.info("RESUMED from checkpoint -> starting at epoch %d (best %s=%.4f)",
                         self.start_epoch, self._monitor_key(), best if best else 0.0)

    def fit(self) -> None:
        cfg = self.cfg
        monitor_key = self._monitor_key()
        self._maybe_resume()
        best_metric = self.checkpoint.best
        self.logger.info("Training epochs %d..%d", self.start_epoch, cfg.train.epochs - 1)

        for epoch in range(self.start_epoch, cfg.train.epochs):
            train_loss = self.train_one_epoch(epoch)
            metrics = self.validate_epoch(epoch)
            monitored = metrics[monitor_key]

            if self.plateau is not None and epoch >= cfg.train.warmup_epochs:
                self.plateau.step(monitored)

            self._record_epoch(epoch, train_loss, metrics)

            improved = self.checkpoint.is_improvement(monitored)
            self._save_last(epoch, monitored)
            if cfg.train.save_every_epoch:
                self._save_epoch_checkpoint(epoch)
            if improved:
                best_metric = monitored
                self._save_best(epoch, monitored)
                self.logger.info("  -> new best %s=%.4f (checkpoint saved)",
                                 monitor_key, monitored)

            self.early_stopping.step(monitored)
            if self.early_stopping.should_stop:
                self.logger.info("Early stopping at epoch %d (best %s=%.4f)",
                                 epoch, monitor_key, best_metric)
                break

        self._finalise()
        self.writer.close()
        self.logger.info("Training complete. Best %s=%.4f", monitor_key, best_metric)

    # ------------------------------------------------------------------ #
    def _state_for_save(self):
        """Return the state dict to persist (EMA weights when enabled)."""
        if self.ema is not None:
            with self.ema.average_parameters(self.model):
                return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _save_best(self, epoch: int, metric: float) -> None:
        path = os.path.join(self.cfg.paths.checkpoint_dir, "best_model.pth")
        torch.save(
            {"model_state": self._state_for_save(), "epoch": epoch,
             "best_metric": metric, "config": self.cfg.to_dict()}, path,
        )
        torch.save(self._state_for_save(),
                   os.path.join(self.cfg.paths.checkpoint_dir, "best_weights.pth"))

    def _save_last(self, epoch: int, metric: float) -> None:
        save_checkpoint(
            os.path.join(self.cfg.paths.checkpoint_dir, "last.pth"),
            self.model, self.base_optimizer, None, epoch, metric,
            extra={"ema_state": self.ema.state_dict() if self.ema else None},
        )

    def _save_epoch_checkpoint(self, epoch: int) -> None:
        """Persist per-epoch weights (EMA if enabled) to epoch_XXX.pth."""
        path = os.path.join(self.cfg.paths.checkpoint_dir, f"epoch_{epoch:03d}.pth")
        torch.save(self._state_for_save(), path)

    def _record_epoch(self, epoch: int, train_loss: float, metrics: dict) -> None:
        """Append this epoch's metrics to history, rewrite the CSV and redraw curves."""
        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(metrics["loss"]), 6),
            "val_macro_f1": round(float(metrics["macro_f1"]), 6),
            "val_macro_auc": round(float(metrics["macro_auc"]), 6),
            "val_balanced_acc": round(float(metrics["balanced_accuracy"]), 6),
            "val_accuracy": round(float(metrics["accuracy"]), 6),
            "val_macro_sensitivity": round(float(metrics["macro_sensitivity"]), 6),
            "val_macro_specificity": round(float(metrics["macro_specificity"]), 6),
        }
        for name in self.cfg.data.class_names:
            pc = metrics["per_class"][name]
            row[f"recall_{name}"] = round(float(pc["recall"]), 6)
            row[f"precision_{name}"] = round(float(pc["precision"]), 6)
        self.history.append(row)

        csv_path = os.path.join(self.cfg.paths.output_dir, "metrics_history.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        self._plot_training_curves()

    def _plot_training_curves(self) -> None:
        """Redraw outputs/training_curves.png from the running history."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = self.history
        ep = [r["epoch"] for r in h]
        names = list(self.cfg.data.class_names)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))

        ax[0, 0].plot(ep, [r["train_loss"] for r in h], "-o", label="train", ms=3)
        ax[0, 0].plot(ep, [r["val_loss"] for r in h], "-o", label="val", ms=3)
        ax[0, 0].set(title="Loss", xlabel="epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

        for key, lab in [("val_macro_f1", "macro-F1"), ("val_macro_auc", "macro-AUC"),
                         ("val_balanced_acc", "balanced-acc")]:
            ax[0, 1].plot(ep, [r[key] for r in h], "-o", label=lab, ms=3)
        ax[0, 1].set(title="Validation headline metrics", xlabel="epoch", ylim=(0, 1))
        ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

        for n in names:
            ax[1, 0].plot(ep, [r[f"recall_{n}"] for r in h], "-o", label=n, ms=3)
        ax[1, 0].set(title="Per-class recall (sensitivity)", xlabel="epoch", ylim=(0, 1))
        ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

        for n in names:
            ax[1, 1].plot(ep, [r[f"precision_{n}"] for r in h], "-o", label=n, ms=3)
        ax[1, 1].set(title="Per-class precision", xlabel="epoch", ylim=(0, 1))
        ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.cfg.paths.output_dir, "training_curves.png"), dpi=130)
        plt.close(fig)

    def _finalise(self) -> None:
        """Save the feature extractor (encoder) from the best weights."""
        best_path = os.path.join(self.cfg.paths.checkpoint_dir, "best_weights.pth")
        if os.path.isfile(best_path):
            state = torch.load(best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
        fe_state = self.model.feature_extractor_state_dict()
        torch.save(
            {"encoder_state": {k: v.cpu() for k, v in fe_state.items()},
             "config": self.cfg.to_dict()},
            os.path.join(self.cfg.paths.checkpoint_dir, "feature_extractor.pth"),
        )
        self.logger.info("Saved feature_extractor.pth")


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--metadata-csv", type=str, default=None)
    p.add_argument("--loss", type=str, default=None, choices=["focal", "weighted_ce"])
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Subsample patients for a quick smoke test.")
    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    cfg = get_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.data_root is not None:
        cfg.paths.data_root = args.data_root
    if args.metadata_csv is not None:
        cfg.paths.metadata_csv = args.metadata_csv
    if args.loss is not None:
        cfg.train.loss = args.loss
    if args.no_sam:
        cfg.train.use_sam = False
    if args.limit is not None:
        cfg._limit = args.limit
    cfg.validate()
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    Trainer(cfg).fit()


if __name__ == "__main__":
    main()
