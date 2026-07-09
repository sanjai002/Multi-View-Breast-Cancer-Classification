"""Multi-class precision-recall curve plotting (one-vs-rest)."""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_precision_recall_curves(y_true: Sequence[int], y_prob: np.ndarray,
                                 class_names: Sequence[str], save_path: str) -> str:
    """Save one-vs-rest precision-recall curves with average precision."""
    num_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0,
                           posinf=1.0, neginf=0.0)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    ap_values = []
    for i, name in enumerate(class_names):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ap_values.append(ap)
        ax.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")
        # Baseline = class prevalence.
        prevalence = y_bin[:, i].mean()
        ax.hlines(prevalence, 0, 1, colors="gray", linestyles=":", alpha=0.4)

    if ap_values:
        ax.set_title(f"Precision-Recall curves (mAP={np.mean(ap_values):.3f})")
    else:
        ax.set_title("Precision-Recall curves")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
