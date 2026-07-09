"""Confusion-matrix plotting (raw counts and row-normalised)."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int],
                          class_names: Sequence[str], save_path: str,
                          normalize: bool = False,
                          title: Optional[str] = None) -> str:
    """Save a confusion-matrix heatmap to ``save_path``."""
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_display = cm.astype(np.float64)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(num_classes), yticks=np.arange(num_classes),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True label", xlabel="Predicted label",
        title=title or ("Normalised confusion matrix" if normalize else "Confusion matrix"),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_display.max() / 2.0 if cm_display.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            txt = f"{cm_display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
