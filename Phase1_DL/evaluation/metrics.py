"""Classification metrics and the PDF classification report.

``compute_metrics`` returns a flat dictionary of scalars (for logging / model
selection) plus per-class breakdowns and the confusion matrix. Sensitivity and
specificity are computed one-vs-rest from the confusion matrix. ROC AUC falls
back to per-class computation when a class is missing from ``y_true`` so a rare
class in a small validation split never crashes training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict:
    """Macro one-vs-rest ROC AUC that tolerates absent classes."""
    labels = list(range(num_classes))
    per_class: Dict[int, float] = {}
    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.shape[1] == 1:  # binary edge-case from label_binarize
        y_bin = np.hstack([1 - y_bin, y_bin])
    for c in labels:
        if len(np.unique(y_bin[:, c])) < 2:
            per_class[c] = float("nan")
            continue
        try:
            per_class[c] = float(roc_auc_score(y_bin[:, c], y_prob[:, c]))
        except ValueError:
            per_class[c] = float("nan")
    valid = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float("nan")
    return {"macro_auc": macro, "per_class_auc": per_class}


def compute_metrics(y_true: Sequence[int], y_prob: np.ndarray,
                    class_names: Sequence[str]) -> Dict:
    """Compute the full metric suite from labels and class probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.nan_to_num(np.asarray(y_prob, dtype=np.float64), nan=0.0,
                           posinf=1.0, neginf=0.0)
    num_classes = len(class_names)
    y_pred = y_prob.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # One-vs-rest sensitivity / specificity from the confusion matrix.
    sensitivity, specificity = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    auc = _safe_auc(y_true, y_prob, num_classes)

    metrics: Dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_auc": auc["macro_auc"],
        "macro_sensitivity": float(np.mean(sensitivity)),
        "macro_specificity": float(np.mean(specificity)),
        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }

    prec = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    for i, name in enumerate(class_names):
        metrics["per_class"][name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]),
            "auc": float(auc["per_class_auc"][i]),
            "support": int((y_true == i).sum()),
        }
    return metrics


def format_metrics(metrics: Dict) -> str:
    """One-line human-readable summary for logging."""
    return (
        f"acc={metrics['accuracy']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | "
        f"macroF1={metrics['macro_f1']:.4f} | AUC={metrics['macro_auc']:.4f} | "
        f"sens={metrics['macro_sensitivity']:.4f} | spec={metrics['macro_specificity']:.4f}"
    )


def save_classification_report_pdf(y_true: Sequence[int], y_pred: Sequence[int],
                                   metrics: Dict, class_names: Sequence[str],
                                   save_path: str,
                                   title: str = "Classification Report") -> str:
    """Render a multi-section classification report to a PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.metrics import classification_report

    report_txt = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=list(class_names), zero_division=0,
    )

    with PdfPages(save_path) as pdf:
        # Page 1: headline metrics + text report.
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        lines = [
            title, "",
            f"Accuracy            : {metrics['accuracy']:.4f}",
            f"Balanced accuracy   : {metrics['balanced_accuracy']:.4f}",
            f"Macro F1            : {metrics['macro_f1']:.4f}",
            f"Weighted F1         : {metrics['weighted_f1']:.4f}",
            f"Macro ROC AUC       : {metrics['macro_auc']:.4f}",
            f"Macro sensitivity   : {metrics['macro_sensitivity']:.4f}",
            f"Macro specificity   : {metrics['macro_specificity']:.4f}",
            f"Cohen's kappa       : {metrics['cohen_kappa']:.4f}",
            "", "Per-class report (sklearn):", "",
        ]
        ax.text(0.05, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
        ax.text(0.05, 0.55, report_txt, va="top", family="monospace", fontsize=10)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: per-class table with sensitivity / specificity / AUC.
        fig, ax = plt.subplots(figsize=(8.27, 6))
        ax.axis("off")
        col_labels = ["Class", "Precision", "Recall", "F1", "Sensitivity", "Specificity", "AUC", "Support"]
        rows = []
        for name in class_names:
            pc = metrics["per_class"][name]
            rows.append([
                name, f"{pc['precision']:.3f}", f"{pc['recall']:.3f}", f"{pc['f1']:.3f}",
                f"{pc['sensitivity']:.3f}", f"{pc['specificity']:.3f}",
                f"{pc['auc']:.3f}", str(pc["support"]),
            ])
        table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        ax.set_title("Per-class metrics", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return save_path
