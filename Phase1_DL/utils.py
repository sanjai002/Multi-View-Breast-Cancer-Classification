# utils.py
"""
utils.py

Small shared helpers for the minimal NLBS four-view pipeline:
- logging
- reproducibility
- class weighting
- metrics
- checkpoint IO
- plotting
- simple Grad-CAM
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config as C

LOGGER_NAME = "nlbs"


def _cfg(name: str, default):
    return getattr(C, name, default)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = Path(_cfg("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
CHECKPOINT_DIR = Path(_cfg("CHECKPOINT_DIR", OUTPUT_DIR / "checkpoints"))
LOG_DIR = Path(_cfg("LOG_DIR", OUTPUT_DIR / "logs"))
VIEW_ORDER = tuple(_cfg("VIEW_ORDER", ("LCC", "LMLO", "RCC", "RMLO")))
CLASS_NAMES = tuple(_cfg("CLASS_NAMES", ("Normal", "Abnormal")))


# ---------------------------------------------------------------------
# Logging / reproducibility
# ---------------------------------------------------------------------
def get_logger(name: str = LOGGER_NAME, log_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# Class weights / early stopping
# ---------------------------------------------------------------------
def build_class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------
def save_checkpoint(state: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(Path(path), map_location=map_location)


def save_predictions_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    out_path: str | Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: Sequence[int], y_prob: Sequence[float], out_path: str | Path) -> None:
    from sklearn.metrics import roc_curve, auc

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: pd.DataFrame, out_path: str | Path) -> None:
    if history.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "balanced_accuracy" in history.columns:
        axes[1].plot(history["epoch"], history["balanced_accuracy"], label="balanced_acc")
    if "accuracy" in history.columns:
        axes[1].plot(history["epoch"], history["accuracy"], label="accuracy")
    if "f1" in history.columns:
        axes[1].plot(history["epoch"], history["f1"], label="f1")
    axes[1].set(title="Validation metrics", xlabel="Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------
def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam -= cam.min()
    cam /= (cam.max() - cam.min() + 1e-8)
    return cam


@torch.no_grad()
def denormalize_image(img: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    x = img.detach().cpu().numpy()
    if x.ndim == 3:
        x = x[0]
    x = x * std + mean
    return np.clip(x, 0.0, 1.0)


def generate_gradcam(
    model: torch.nn.Module,
    views: torch.Tensor,
    mask: torch.Tensor,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Return Grad-CAM maps with shape [B, V, H, W] for the model's last conv block.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    target_layer = model.backbone.get_target_layer()

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_module, _inp, output):
        activations.append(output)

        def bwd_hook(grad):
            gradients.append(grad)

        output.register_hook(bwd_hook)

    handle = target_layer.register_forward_hook(fwd_hook)

    try:
        views = views.to(device)
        mask = mask.to(device)

        logits = model(views, mask)["logits"]
        if target_class is None:
            target_idx = logits.argmax(dim=1)
        else:
            target_idx = torch.full((logits.size(0),), int(target_class), device=device, dtype=torch.long)

        score = logits.gather(1, target_idx.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))  # [B*V, h, w]
        cam = F.interpolate(cam.unsqueeze(1), size=views.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        cam_np = cam.detach().cpu().numpy()
        b, v = views.shape[:2]
        cam_np = cam_np.reshape(b, v, cam_np.shape[-2], cam_np.shape[-1])
        return np.stack([_normalize_map(c) for c in cam_np.reshape(-1, cam_np.shape[-2], cam_np.shape[-1])]).reshape(
            b, v, cam_np.shape[-2], cam_np.shape[-1]
        )
    finally:
        handle.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a grayscale image and a heatmap into an RGB overlay.
    """
    image = _normalize_map(image)
    heatmap = _normalize_map(heatmap)

    img_u8 = (image * 255).astype(np.uint8)
    heat_u8 = (heatmap * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    gray_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(gray_rgb, 1.0 - alpha, colored, alpha, 0)
    return overlay


def save_gradcam_grid(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    out_path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    """
    Save a simple Grad-CAM grid for one batch.
    """
    views = batch["views"]
    mask = batch["mask"]
    labels = batch["label"]

    cams = generate_gradcam(model, views, mask)

    b = views.shape[0]
    fig, axes = plt.subplots(b, len(VIEW_ORDER) + 1, figsize=(3 * (len(VIEW_ORDER) + 1), 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(b):
        for j, view_name in enumerate(VIEW_ORDER):
            img = denormalize_image(views[i, j])
            cam = cams[i, j]
            overlay = overlay_heatmap(img, cam)
            axes[i, j].imshow(overlay)
            axes[i, j].set_title(view_name)
            axes[i, j].axis("off")

        axes[i, -1].text(
            0.5,
            0.5,
            f"Label: {class_names[int(labels[i])]}",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[i, -1].axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)