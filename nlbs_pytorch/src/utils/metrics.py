"""Classification metrics + evaluation plots (binary, positive class = cancer)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve,
                             average_precision_score)


def compute_metrics(y_true, y_prob, thresh: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)          # P(cancer)
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_eval_plots(y_true, y_prob, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["normal", "cancer"], yticklabels=["normal", "cancer"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(rec, prec, label=f"AP={average_precision_score(y_true, y_prob):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=150); plt.close()

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.xlabel("Mean predicted prob"); plt.ylabel("Fraction positive")
    plt.title("Calibration"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "calibration_curve.png", dpi=150); plt.close()
