# train.py
"""
train.py

Minimal training loop for the NLBS four-view cache-only pipeline.

Assumptions:
- patient_manifest.csv already exists
- preprocess.py has already generated the .npy cache
- dataset.py never reads DICOM files
- model.py provides a shared ResNet50 four-view classifier

Outputs:
- checkpoints/best_model.pt
- checkpoints/last_model.pt
- outputs/history.csv
- outputs/predictions_val.csv
- outputs/predictions_test.csv
- outputs/confusion_matrix.png
- outputs/roc_curve.png
- outputs/training_curves.png
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config as C
from dataset import (
    NLBSDataset,
    load_patient_manifest,
    split_manifest,
    verify_cache_for_manifest,
)
from model import build_model
from utils import (
    EarlyStopping,
    build_class_weights,
    compute_binary_metrics,
    get_device,
    get_logger,
    load_checkpoint,
    plot_confusion_matrix,
    plot_history,
    plot_roc_curve,
    save_checkpoint,
    save_predictions_csv,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _cfg(name: str, default):
    return getattr(C, name, default)


OUTPUT_DIR = Path(_cfg("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
CHECKPOINT_DIR = Path(_cfg("CHECKPOINT_DIR", OUTPUT_DIR / "checkpoints"))
LOG_DIR = Path(_cfg("LOG_DIR", OUTPUT_DIR / "logs"))
TENSORBOARD_DIR = Path(_cfg("TENSORBOARD_DIR", OUTPUT_DIR / "tensorboard"))
GRADCAM_DIR = Path(_cfg("GRADCAM_DIR", OUTPUT_DIR / "gradcam"))

DATA_ROOT = Path(_cfg("DATA_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()
CACHE_DIR = Path(_cfg("CACHE_DIR", OUTPUT_DIR / "preproc_cache")).expanduser().resolve()
PATIENT_MANIFEST = Path(_cfg("PATIENT_MANIFEST", OUTPUT_DIR / "patient_manifest.csv"))

IMAGE_SIZE = int(_cfg("IMAGE_SIZE", 224))
VIEW_ORDER = tuple(_cfg("VIEW_ORDER", ("LCC", "LMLO", "RCC", "RMLO")))
BATCH_SIZE = int(_cfg("BATCH_SIZE", 8))
NUM_WORKERS = int(_cfg("NUM_WORKERS", 4))
EPOCHS = int(_cfg("EPOCHS", 40))
LEARNING_RATE = float(_cfg("LEARNING_RATE", 1e-4))
WEIGHT_DECAY = float(_cfg("WEIGHT_DECAY", 1e-4))
MIN_LR = float(_cfg("MIN_LR", 1e-6))
PATIENCE = int(_cfg("EARLY_STOPPING_PATIENCE", 10))
SEED = int(_cfg("RANDOM_SEED", 42))
USE_AMP = bool(_cfg("USE_AMP", True))
DEVICE_NAME = str(_cfg("DEVICE", "cuda"))
PRETRAINED = bool(_cfg("PRETRAINED", True))
FREEZE_BACKBONE = bool(_cfg("FREEZE_BACKBONE", False))
DROPOUT = float(_cfg("DROPOUT", 0.5))
HIDDEN_DIM = int(_cfg("HIDDEN_DIM", 512))
NUM_CLASSES = int(_cfg("NUM_CLASSES", 2))
CLASS_NAMES = tuple(_cfg("CLASS_NAMES", ("Normal", "Abnormal")))


def ensure_dirs() -> None:
    for p in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, TENSORBOARD_DIR, GRADCAM_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def build_loaders(
    manifest: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    train_df = split_manifest(manifest, "train")
    val_df = split_manifest(manifest, "val")
    test_df = split_manifest(manifest, "test")

    common_kwargs = dict(
        data_root=data_root,
        cache_dir=cache_dir,
        image_size=IMAGE_SIZE,
        strict_cache=True,
    )

    train_ds = NLBSDataset(train_df, **common_kwargs)
    val_ds = NLBSDataset(val_df, **common_kwargs)
    test_ds = NLBSDataset(test_df, **common_kwargs)

    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    model.eval()

    all_true: List[int] = []
    all_prob: List[float] = []
    all_pred: List[int] = []
    all_pid: List[str] = []
    all_age: List[float] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        out = model(views, mask)
        logits = out["logits"]
        probs = out["probs"]

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1

        pred = torch.argmax(probs, dim=1)
        pos_prob = probs[:, 1]

        all_true.extend(labels.cpu().tolist())
        all_prob.extend(pos_prob.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_pid.extend(list(batch["patient_id"]))
        all_age.extend([float(a) if torch.is_tensor(a) else float(a) for a in batch["age"]])

    metrics = compute_binary_metrics(np.asarray(all_true), np.asarray(all_prob), np.asarray(all_pred))
    metrics["loss"] = total_loss / max(n_batches, 1)

    pred_df = pd.DataFrame(
        {
            "patient_id": all_pid,
            "age": all_age,
            "true_label": all_true,
            "pred_label": all_pred,
            "prob_abnormal": all_prob,
            "prob_normal": [1.0 - p for p in all_prob],
            "correct": [int(t == p) for t, p in zip(all_true, all_pred)],
        }
    )
    return metrics["loss"], metrics, pred_df


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(views, mask)
                loss = criterion(out["logits"], labels)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(views, mask)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item())
        n_batches += 1

    return running_loss / max(n_batches, 1)


def fit(args: argparse.Namespace) -> None:
    ensure_dirs()
    logger = get_logger("phase1", str(LOG_DIR))
    device = get_device(DEVICE_NAME)
    set_seed(SEED)

    logger.info("Device: %s", device)
    logger.info("Reading manifest: %s", PATIENT_MANIFEST)

    if not PATIENT_MANIFEST.is_file():
        raise FileNotFoundError(f"patient manifest not found: {PATIENT_MANIFEST}")

    manifest = load_patient_manifest(PATIENT_MANIFEST)

    logger.info("Verifying cache...")
    missing = verify_cache_for_manifest(
        manifest,
        data_root=DATA_ROOT,
        cache_dir=CACHE_DIR,
        image_size=IMAGE_SIZE,
    )
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            "Cache verification failed. Missing cache files:\n"
            f"{preview}\n"
            f"Total missing: {len(missing)}"
        )

    loaders = build_loaders(
        manifest=manifest,
        data_root=DATA_ROOT,
        cache_dir=CACHE_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_labels = split_manifest(manifest, "train")["label"].to_numpy()
    class_weights = build_class_weights(train_labels, num_classes=NUM_CLASSES).to(device)

    model = build_model(
        pretrained=PRETRAINED,
        dropout=DROPOUT,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=MIN_LR)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")
    early_stopping = EarlyStopping(patience=PATIENCE, mode="max", min_delta=0.0)

    history: List[Dict[str, float]] = []
    best_score = -math.inf
    best_epoch = -1

    start_epoch = 0
    last_ckpt = CHECKPOINT_DIR / "last_model.pt"
    if args.resume and last_ckpt.is_file():
        ckpt = load_checkpoint(last_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        logger.info("Resumed from %s at epoch %d", last_ckpt, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=USE_AMP,
        )
        val_loss, val_metrics, val_pred_df = evaluate_loader(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(epoch_row)

        history_df = pd.DataFrame(history)
        history_df.to_csv(OUTPUT_DIR / "history.csv", index=False)
        plot_history(history_df, OUTPUT_DIR / "training_curves.png")

        save_predictions_csv(val_pred_df, OUTPUT_DIR / "predictions_val.csv")
        plot_confusion_matrix(
            val_pred_df["true_label"].to_numpy(),
            val_pred_df["pred_label"].to_numpy(),
            CLASS_NAMES,
            OUTPUT_DIR / "confusion_matrix.png",
        )
        plot_roc_curve(
            val_pred_df["true_label"].to_numpy(),
            val_pred_df["prob_abnormal"].to_numpy(),
            OUTPUT_DIR / "roc_curve.png",
        )

        score = float(val_metrics["balanced_accuracy"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                },
                CHECKPOINT_DIR / "best_model.pt",
            )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
            },
            last_ckpt,
        )

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_bal_acc=%.4f | best=%.4f | time=%.1fs",
            epoch + 1,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["balanced_accuracy"],
            best_score,
            time.time() - t0,
        )

        if early_stopping.step(score):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    best_ckpt = CHECKPOINT_DIR / "best_model.pt"
    if best_ckpt.is_file():
        ckpt = load_checkpoint(best_ckpt, device)
        model.load_state_dict(ckpt["model_state"])

    test_loss, test_metrics, test_pred_df = evaluate_loader(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
    )
    test_metrics["loss"] = test_loss
    save_predictions_csv(test_pred_df, OUTPUT_DIR / "predictions_test.csv")

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    logger.info("Test metrics: %s", json.dumps({k: float(v) for k, v in test_metrics.items()}, indent=2))
    logger.info("Best epoch: %d | best val balanced accuracy: %.4f", best_epoch, best_score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the minimal NLBS four-view model")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/last_model.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()