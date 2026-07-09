"""Calibration (reliability) diagrams and Expected Calibration Error.

Well-calibrated probabilities matter for the downstream RL phase, whose reward
can depend on predicted confidence. This module plots one-vs-rest reliability
curves and reports per-class and overall ECE.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize


def expected_calibration_error(y_true_bin: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE for a single one-vs-rest problem."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true_bin[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def plot_calibration_curves(y_true: Sequence[int], y_prob: np.ndarray,
                            class_names: Sequence[str], save_path: str,
                            n_bins: int = 10) -> Tuple[str, Dict[str, float]]:
    """Save reliability diagrams; return path and per-class ECE dict."""
    num_classes = len(class_names)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0,
                           posinf=1.0, neginf=0.0)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")

    ece_by_class: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        if len(np.unique(y_bin[:, i])) < 2:
            ece_by_class[name] = float("nan")
            continue
        frac_pos, mean_pred = calibration_curve(
            y_bin[:, i], y_prob[:, i], n_bins=n_bins, strategy="uniform"
        )
        ece = expected_calibration_error(y_bin[:, i], y_prob[:, i], n_bins)
        ece_by_class[name] = ece
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=f"{name} (ECE={ece:.3f})")

    valid_ece = [v for v in ece_by_class.values() if not np.isnan(v)]
    ece_by_class["macro"] = float(np.mean(valid_ece)) if valid_ece else float("nan")

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
           ylabel="Fraction of positives",
           title=f"Calibration curves (macro ECE={ece_by_class['macro']:.3f})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path, ece_by_class
