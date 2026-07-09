"""Multi-class ROC curve plotting (one-vs-rest, with micro/macro averages)."""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_roc_curves(y_true: Sequence[int], y_prob: np.ndarray,
                    class_names: Sequence[str], save_path: str) -> str:
    """Save one-vs-rest ROC curves plus micro/macro averages."""
    num_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0,
                           posinf=1.0, neginf=0.0)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    fpr_grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    valid_classes = 0

    for i, name in enumerate(class_names):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
        valid_classes += 1

    if valid_classes > 0:
        mean_tpr /= valid_classes
        macro_auc = auc(fpr_grid, mean_tpr)
        ax.plot(fpr_grid, mean_tpr, "k--", lw=2,
                label=f"macro-average (AUC={macro_auc:.3f})")

    # Micro-average over all one-vs-rest decisions.
    if len(np.unique(y_bin.ravel())) >= 2:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_prob.ravel())
        ax.plot(fpr_micro, tpr_micro, ":", color="gray", lw=2,
                label=f"micro-average (AUC={auc(fpr_micro, tpr_micro):.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", alpha=0.6)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title="ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
