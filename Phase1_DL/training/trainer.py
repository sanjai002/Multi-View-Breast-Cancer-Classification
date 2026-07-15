"""Training loop for multi-view mammography classifier."""

from typing import Dict, Tuple, Optional, Callable
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F

from config.base import Config
from utils.logging import get_logger


class Trainer:
    """Train multi-view mammography classifier with validation and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        cfg: Optional[Config] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Optional learning rate scheduler.
            cfg: Configuration object.
            device: Device to train on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg or Config()
        self.device = device

        self.logger = get_logger("trainer")

        # Tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.train_history = {
            "loss": [],
            "acc": [],
            "auc": [],
        }
        self.val_history = {
            "loss": [],
            "acc": [],
            "auc": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dictionary with epoch metrics: loss, accuracy, AUC.
        """
        self.model.train()
        losses = []
        preds_all = []
        labels_all = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            views = batch["views"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            logits, attention = self.model(views, mask)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Tracking
            losses.append(loss.item())
            preds_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

            if (batch_idx + 1) % 100 == 0:
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}"
                )

        # Compute metrics
        preds = np.concatenate(preds_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)

        avg_loss = np.mean(losses)
        accuracy = self._compute_accuracy(preds, labels)
        auc = self._compute_auc(preds, labels)

        return {
            "loss": avg_loss,
            "acc": accuracy,
            "auc": auc,
        }

    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary with validation metrics: loss, accuracy, AUC.
        """
        self.model.eval()
        losses = []
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                views = batch["views"].to(self.device)
                mask = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                logits, attention = self.model(views, mask)

                # Compute loss
                loss = self.criterion(logits, labels)

                # Tracking
                losses.append(loss.item())
                preds_all.append(logits.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        # Compute metrics
        preds = np.concatenate(preds_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)

        avg_loss = np.mean(losses)
        accuracy = self._compute_accuracy(preds, labels)
        auc = self._compute_auc(preds, labels)

        return {
            "loss": avg_loss,
            "acc": accuracy,
            "auc": auc,
        }

    def fit(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict:
        """Train model for specified epochs with early stopping.

        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Stop if validation AUC doesn't improve for N epochs.

        Returns:
            Dictionary with training history and best metrics.
        """
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()
            self.train_history["loss"].append(train_metrics["loss"])
            self.train_history["acc"].append(train_metrics["acc"])
            self.train_history["auc"].append(train_metrics["auc"])

            # Validate
            val_metrics = self.validate()
            self.val_history["loss"].append(val_metrics["loss"])
            self.val_history["acc"].append(val_metrics["acc"])
            self.val_history["auc"].append(val_metrics["auc"])

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["auc"])

            epoch_time = time.time() - epoch_start

            # Logging
            self.logger.info(
                f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)\n"
                f"  Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['acc']:.4f}, AUC: {train_metrics['auc']:.4f}\n"
                f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}"
            )

            # Best model checkpoint
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.best_epoch = epoch
                patience_counter = 0

                if self.cfg.paths.checkpoint_dir:
                    ckpt_path = self.cfg.paths.checkpoint_dir / f"best_model.pt"
                    self._save_checkpoint(ckpt_path)
                    self.logger.info(f"✓ New best model saved (AUC={self.best_val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(
                        f"Early stopping: no improvement for {early_stopping_patience} epochs"
                    )
                    break

        self.logger.info(
            f"\nTraining complete. Best validation AUC: {self.best_val_auc:.4f} "
            f"(epoch {self.best_epoch})"
        )

        return {
            "best_auc": self.best_val_auc,
            "best_epoch": self.best_epoch,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_auc": self.best_val_auc,
                "best_epoch": self.best_epoch,
                "cfg": self.cfg,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_auc = checkpoint["best_val_auc"]
        self.best_epoch = checkpoint["best_epoch"]
        self.logger.info(f"Loaded checkpoint from {path} (best AUC: {self.best_val_auc:.4f})")

    @staticmethod
    def _compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy.

        Args:
            logits: (N, 2) logits.
            labels: (N,) ground truth.

        Returns:
            Accuracy in [0, 1].
        """
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == labels))

    @staticmethod
    def _compute_auc(logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC (binary classification).

        Args:
            logits: (N, 2) logits.
            labels: (N,) ground truth {0, 1}.

        Returns:
            AUC in [0, 1].
        """
        try:
            from sklearn.metrics import roc_auc_score

            # Probability of positive class
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
            auc = roc_auc_score(labels, probs)
            return float(auc)
        except Exception:
            return 0.0  # Return 0 if AUC computation fails (e.g., all samples same class)
