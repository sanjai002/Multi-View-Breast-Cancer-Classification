#!/usr/bin/env python3
"""Full training script for Colab: runs balanced dataset with live progress, checkpointing, and metrics."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

# Ensure Phase1_DL can be imported
ROOT = Path(__file__).resolve().parents[1]
PHASE1_DL = ROOT / "Phase1_DL"
if str(PHASE1_DL) not in sys.path:
    sys.path.insert(0, str(PHASE1_DL))

from config.base import get_config
from datasets.cached_dataset import CachedMammoDataset
from models.simple_fusion import SimpleFusionModel
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full training on Colab")
    parser.add_argument(
        "--manifest",
        type=str,
        default="balanced",
        help="Manifest key: 'balanced' or 'full' or custom path.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cuda', 'cpu'")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: outputs/checkpoints)",
    )
    return parser.parse_args()


def get_manifest_path(cfg, manifest_key: str) -> Path:
    """Resolve manifest CSV path by key."""
    if manifest_key == "balanced":
        path = cfg.output_dir / "patient_manifest_balanced.csv"
        if path.exists():
            return path
        raise FileNotFoundError(f"Balanced manifest not found: {path}")
    if manifest_key == "full":
        path = cfg.patient_manifest_csv
        if path.exists():
            return path
        raise FileNotFoundError(f"Full manifest not found: {path}")
    path = Path(manifest_key)
    if path.exists():
        return path
    raise FileNotFoundError(f"Manifest not found: {path}")


def setup_drive_save(logger) -> Path | None:
    """Mount Google Drive and setup save directory if running on Colab."""
    try:
        from google.colab import drive
        drive_mount = Path("/content/drive/MyDrive/project")
        if not drive_mount.exists():
            logger.info("Mounting Google Drive...")
            drive.mount("/content/drive")
        return drive_mount
    except ImportError:
        logger.info("Not running on Colab; outputs will stay local")
        return None


def save_to_drive(local_path: Path, drive_root: Path, logger) -> None:
    """Copy local file/dir to Google Drive."""
    if drive_root is None:
        return
    try:
        import shutil
        drive_dest = drive_root / local_path.name
        if local_path.is_dir():
            if drive_dest.exists():
                shutil.rmtree(drive_dest)
            shutil.copytree(local_path, drive_dest)
        else:
            drive_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, drive_dest)
        logger.info(f"Saved to Drive: {drive_dest}")
    except Exception as e:
        logger.warning(f"Failed to save to Drive: {e}")


def main() -> None:
    args = parse_args()
    cfg = get_config()
    logger = get_logger("train_colab")

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load manifest
    manifest_path = get_manifest_path(cfg, args.manifest)
    logger.info(f"Loading manifest from {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(manifest_df)} patients")

    # Create dataset
    logger.info("Creating cached dataset...")
    ds = CachedMammoDataset(manifest_path, cfg=cfg)
    logger.info(f"Dataset size: {len(ds)}")

    # Split into train/val
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # DataLoaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    logger.info("Initializing model...")
    model = SimpleFusionModel(num_views=4, feat_dim=64, num_classes=2).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.training.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(train_dl):
            imgs = batch["images"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(imgs, mask=mask)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * labels.size(0)
            train_count += labels.size(0)

            if (batch_idx + 1) % max(1, len(train_dl) // 10) == 0 or batch_idx + 1 == len(train_dl):
                avg_loss = train_loss / train_count
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx + 1}/{len(train_dl)} | Loss {avg_loss:.4f}"
                )

        train_loss_avg = train_loss / train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        correct = 0

        with torch.no_grad():
            for batch in val_dl:
                imgs = batch["images"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(imgs, mask=mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_count += labels.size(0)

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss_avg = val_loss / val_count
        val_acc = correct / val_count

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss {train_loss_avg:.4f} | "
            f"Val Loss {val_loss_avg:.4f} | Val Acc {val_acc:.4f}"
        )

        # Save checkpoint if best
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            ckpt_path = ckpt_dir / f"best_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
