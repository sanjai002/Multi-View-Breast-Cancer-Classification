"""Test-set evaluation and full artefact export for Phase 1.

Loads the best checkpoint and produces every deliverable required downstream:

* Metrics + plots on the held-out test split (confusion matrix, ROC, PR,
  calibration, PDF report).
* Per-patient prediction CSVs for **all** splits.
* CNN feature vectors for **all** splits (``image_features.npy`` per view and
  ``patient_features.npy`` per patient) with index CSVs -- the Phase 2 inputs.
* ``feature_extractor.pth`` (encoder weights).
* Grad-CAM / Grad-CAM++ / Score-CAM / Integrated-Gradients overlays for a sample
  of test patients under ``outputs/gradcam_images/``.

Run from the project root::

    python -m training.test
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from augmentation import build_eval_transforms
from config import VIEW_ORDER, Config, get_config
from dataset import MultiViewMammographyDataset
from evaluation.calibration_curve import plot_calibration_curves
from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import (compute_metrics, format_metrics,
                                save_classification_report_pdf)
from evaluation.precision_recall import plot_precision_recall_curves
from evaluation.roc_curve import plot_roc_curves
from explainability.gradcam import GradCAM
from explainability.gradcam_plus import GradCAMPlusPlus
from explainability.integrated_gradients import IntegratedGradients
from explainability.scorecam import ScoreCAM
from models.fusion import build_model
from training.validate import evaluate
from utils import (build_patient_table, get_device, get_logger, load_metadata,
                   overlay_heatmap, patient_level_split)


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def load_trained_model(cfg: Config, device: torch.device, logger) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    weights_path = os.path.join(cfg.paths.checkpoint_dir, "best_weights.pth")
    model_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pth")
    if os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        logger.info("Loaded weights from %s", weights_path)
    elif os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Loaded model from %s", model_path)
    else:
        logger.warning("No checkpoint found; evaluating randomly-initialised model.")
    model.eval()
    return model


def load_tables(cfg: Config, logger) -> pd.DataFrame:
    """Reuse the training manifest if present, else rebuild the split."""
    if os.path.isfile(cfg.paths.manifest_csv):
        logger.info("Loading split manifest: %s", cfg.paths.manifest_csv)
        return pd.read_csv(cfg.paths.manifest_csv)
    logger.warning("Manifest missing; rebuilding patient split.")
    df = load_metadata(cfg, logger)
    table = build_patient_table(df, cfg, logger)
    return patient_level_split(table, cfg, logger)


def make_loader(table: pd.DataFrame, cfg: Config) -> DataLoader:
    ds = MultiViewMammographyDataset(table, cfg, build_eval_transforms(cfg), train=False)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory)


# --------------------------------------------------------------------------- #
# Test-set metrics & plots
# --------------------------------------------------------------------------- #
def evaluate_test_split(model, cfg, table, device, logger) -> dict:
    test_table = table[table["split"] == "test"].reset_index(drop=True)
    loader = make_loader(test_table, cfg)
    res = evaluate(model, loader, device, cfg, loss_fn=None)
    metrics = compute_metrics(res.y_true, res.y_prob, cfg.data.class_names)
    logger.info("TEST | %s", format_metrics(metrics))

    out = cfg.paths.output_dir
    class_names = list(cfg.data.class_names)
    plot_confusion_matrix(res.y_true, res.y_pred, class_names,
                          os.path.join(out, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(res.y_true, res.y_pred, class_names,
                          os.path.join(out, "confusion_matrix_normalized.png"), normalize=True)
    plot_roc_curves(res.y_true, res.y_prob, class_names, os.path.join(out, "roc_curve.png"))
    plot_precision_recall_curves(res.y_true, res.y_prob, class_names,
                                 os.path.join(out, "precision_recall.png"))
    _, ece = plot_calibration_curves(res.y_true, res.y_prob, class_names,
                                     os.path.join(out, "calibration_curve.png"))
    metrics["ece"] = ece
    save_classification_report_pdf(res.y_true, res.y_pred, metrics, class_names,
                                   os.path.join(out, "classification_report.pdf"),
                                   title="NLBS Multi-View Test Report")
    logger.info("Saved confusion matrix / ROC / PR / calibration / PDF report.")
    return metrics


# --------------------------------------------------------------------------- #
# Feature & prediction export (all splits)
# --------------------------------------------------------------------------- #
def export_features_and_predictions(model, cfg, table, device, logger) -> None:
    class_names = list(cfg.data.class_names)
    patient_feats, patient_index = [], []
    image_feats, image_index = [], []
    pred_rows = []

    for split in ("train", "val", "test"):
        sub = table[table["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            continue
        loader = make_loader(sub, cfg)
        res = evaluate(model, loader, device, cfg, loss_fn=None, collect_features=True)
        preds = res.y_pred

        for i, pid in enumerate(res.patient_ids):
            label = int(res.y_true[i])
            pred = int(preds[i])
            probs = res.y_prob[i]
            patient_feats.append(res.patient_embeddings[i])
            patient_index.append({
                "row": len(patient_feats) - 1, "patient_id": pid, "split": split,
                "label": label, "label_name": class_names[label],
                "pred": pred, "pred_name": class_names[pred],
                "age": res.ages[i],
            })
            pred_rows.append({
                "patient_id": pid, "split": split,
                "true_label": label, "true_name": class_names[label],
                "pred_label": pred, "pred_name": class_names[pred],
                "correct": int(label == pred),
                "confidence": float(probs[pred]),
                **{f"prob_{class_names[c]}": float(probs[c]) for c in range(len(class_names))},
            })
            for v_idx, view in enumerate(VIEW_ORDER):
                image_feats.append(res.view_embeddings[i, v_idx])
                image_index.append({
                    "row": len(image_feats) - 1, "patient_id": pid, "view": view,
                    "split": split, "present": int(res.view_masks[i, v_idx] > 0.5),
                    "label": label,
                })
        logger.info("Extracted features for split=%s (%d patients)", split, len(sub))

    out = cfg.paths.output_dir
    np.save(os.path.join(out, "patient_features.npy"), np.stack(patient_feats).astype(np.float32))
    np.save(os.path.join(out, "image_features.npy"), np.stack(image_feats).astype(np.float32))
    pd.DataFrame(patient_index).to_csv(os.path.join(out, "patient_feature_index.csv"), index=False)
    pd.DataFrame(image_index).to_csv(os.path.join(out, "image_feature_index.csv"), index=False)

    pred_df = pd.DataFrame(pred_rows)
    prob_cols = ["patient_id", "split", "true_label"] + [f"prob_{c}" for c in class_names] + ["pred_label"]
    pred_df[prob_cols].to_csv(os.path.join(out, "prediction_probabilities.csv"), index=False)
    pred_df[["patient_id", "split", "true_label", "true_name", "pred_label",
             "pred_name", "correct", "confidence"]].to_csv(
        os.path.join(out, "patient_predictions.csv"), index=False)

    # Encoder-only weights for downstream re-use.
    fe_state = model.feature_extractor_state_dict()
    torch.save({"encoder_state": {k: v.cpu() for k, v in fe_state.items()},
                "config": cfg.to_dict()},
               os.path.join(cfg.paths.checkpoint_dir, "feature_extractor.pth"))

    logger.info(
        "Saved patient_features.npy %s, image_features.npy %s, prediction CSVs, feature_extractor.pth",
        np.stack(patient_feats).shape, np.stack(image_feats).shape,
    )


# --------------------------------------------------------------------------- #
# Explainability overlays
# --------------------------------------------------------------------------- #
def _denormalize(view_tensor: torch.Tensor, cfg: Config) -> np.ndarray:
    mean = float(cfg.data.normalize_mean[0])
    std = float(cfg.data.normalize_std[0])
    img = view_tensor.detach().cpu().numpy()
    if img.ndim == 3:
        img = img[0]
    return np.clip(img * std + mean, 0.0, 1.0)


def generate_explanations(model, cfg, table, device, logger) -> None:
    test_table = table[table["split"] == "test"].reset_index(drop=True)
    if len(test_table) == 0:
        logger.warning("No test patients; skipping explainability.")
        return
    sample = test_table.head(cfg.explain.num_samples).reset_index(drop=True)
    loader = DataLoader(
        MultiViewMammographyDataset(sample, cfg, build_eval_transforms(cfg), train=False),
        batch_size=min(4, cfg.train.batch_size), shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    class_names = list(cfg.data.class_names)
    method_names = ["GradCAM", "GradCAM++", "ScoreCAM", "IntegratedGradients"]
    saved = 0

    for batch in loader:
        views = batch["views"].to(device)
        mask = batch["mask"].to(device)

        with GradCAM(model, cfg) as gc:
            m_gc, targets, probs = gc.attribute(views, mask)
        with GradCAMPlusPlus(model, cfg) as gpp:
            m_gpp, _, _ = gpp.attribute(views, mask, target=torch.as_tensor(targets, device=device))
        with ScoreCAM(model, cfg) as sc:
            m_sc, _, _ = sc.attribute(views, mask, target=torch.as_tensor(targets, device=device))
        ig = IntegratedGradients(model, cfg)
        m_ig, _, _ = ig.attribute(views, mask, target=torch.as_tensor(targets, device=device))

        method_maps = [m_gc, m_gpp, m_sc, m_ig]
        for bi in range(views.size(0)):
            pid = str(batch["patient_id"][bi])
            true_label = int(batch["label"][bi])
            pred_label = int(targets[bi])
            _save_patient_overlays(
                views[bi], mask[bi], method_maps, bi, method_names, cfg,
                pid, true_label, pred_label, float(probs[bi, pred_label]),
                class_names,
            )
            saved += 1
    logger.info("Saved Grad-CAM style overlays for %d patients -> %s",
                saved, cfg.paths.gradcam_dir)


def _save_patient_overlays(view_tensor, view_mask, method_maps, bi, method_names,
                           cfg, pid, true_label, pred_label, confidence, class_names):
    present = [j for j in range(len(VIEW_ORDER)) if view_mask[j] > 0.5]
    if not present:
        return
    n_cols = 1 + len(method_names)
    fig, axes = plt.subplots(len(present), n_cols,
                             figsize=(3 * n_cols, 3 * len(present)), squeeze=False)
    for r, v_idx in enumerate(present):
        base = _denormalize(view_tensor[v_idx], cfg)
        axes[r][0].imshow(base, cmap="gray")
        axes[r][0].set_ylabel(VIEW_ORDER[v_idx], fontsize=11)
        axes[r][0].set_title("Original" if r == 0 else "")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for c, (name, maps) in enumerate(zip(method_names, method_maps), start=1):
            overlay = overlay_heatmap(base, maps[bi, v_idx],
                                      alpha=cfg.explain.overlay_alpha,
                                      colormap=cfg.explain.colormap)
            axes[r][c].imshow(overlay)
            axes[r][c].set_title(name if r == 0 else "")
            axes[r][c].axis("off")
    fig.suptitle(
        f"Patient {pid} | true={class_names[true_label]} | "
        f"pred={class_names[pred_label]} ({confidence:.2f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(cfg.paths.gradcam_dir, f"{pid}.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 test / export")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--metadata-csv", type=str, default=None)
    p.add_argument("--no-explain", action="store_true", help="Skip Grad-CAM export.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    cfg.create_dirs()
    if args.data_root:
        cfg.paths.data_root = args.data_root
    if args.metadata_csv:
        cfg.paths.metadata_csv = args.metadata_csv

    logger = get_logger("phase1", cfg.paths.log_dir)
    device = get_device()
    logger.info("Device: %s", device)

    model = load_trained_model(cfg, device, logger)
    table = load_tables(cfg, logger)

    metrics = evaluate_test_split(model, cfg, table, device, logger)
    export_features_and_predictions(model, cfg, table, device, logger)
    if not args.no_explain:
        generate_explanations(model, cfg, table, device, logger)

    import json
    with open(os.path.join(cfg.paths.output_dir, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                  f, indent=2, default=float)
    logger.info("Export complete. Artefacts in %s", cfg.paths.output_dir)


if __name__ == "__main__":
    main()
