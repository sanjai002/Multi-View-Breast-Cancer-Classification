"""Evaluation metrics and diagnostic plots."""

from evaluation.calibration_curve import plot_calibration_curves
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import (compute_metrics, format_metrics,
                                save_classification_report_pdf)
from evaluation.precision_recall import plot_precision_recall_curves
from evaluation.roc_curve import plot_roc_curves

__all__ = [
    "compute_metrics",
    "format_metrics",
    "save_classification_report_pdf",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_calibration_curves",
]
