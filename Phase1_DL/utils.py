"""Shared utilities for Phase 1.

Contains logging setup, reproducibility helpers, metadata construction and
normalisation, patient-level splitting, loss functions, checkpoint IO and small
numeric helpers. Kept dependency-light so it can be imported from any module.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

LOGGER_NAME = "phase1"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    # MONAI shares this project's RNGs; align its determinism state too.
    try:
        from monai.utils import set_determinism

        set_determinism(seed=seed, use_deterministic_algorithms=False)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = LOGGER_NAME, log_dir: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stdout and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Metadata construction & normalisation
# --------------------------------------------------------------------------- #
# Tolerant mapping from a lower-cased/stripped column name to its canonical key.
_COLUMN_ALIASES: Dict[str, str] = {
    "patient id": "Patient_ID", "patient_id": "Patient_ID", "patientid": "Patient_ID",
    "id": "Patient_ID",
    "age": "Age", "patient age": "Age",
    "image laterality": "Image_Laterality", "image_laterality": "Image_Laterality",
    "laterality": "Image_Laterality", "side": "Image_Laterality",
    "view position": "View_Position", "view_position": "View_Position",
    "view": "View_Position", "viewposition": "View_Position",
    "cancer": "Cancer", "malignant": "Cancer",
    "false positive": "False_Positive", "false_positive": "False_Positive",
    "falsepositive": "False_Positive", "fp": "False_Positive",
    "image path": "Image_Path", "image_path": "Image_Path", "path": "Image_Path",
    "filename": "Image_Path", "file": "Image_Path",
}


def normalise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names and normalise laterality/view values."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename).copy()

    # Normalise laterality to {L, R}.
    if "Image_Laterality" in df.columns:
        df["Image_Laterality"] = (
            df["Image_Laterality"].astype(str).str.strip().str.upper().str[0]
        )
    # Normalise view to {CC, MLO}.
    if "View_Position" in df.columns:
        df["View_Position"] = df["View_Position"].astype(str).str.strip().str.upper()
        df["View_Position"] = df["View_Position"].replace(
            {"CRANIOCAUDAL": "CC", "MEDIOLATERALOBLIQUE": "MLO", "ML": "MLO"}
        )
    # Coerce binary flags.
    for c in ("Cancer", "False_Positive"):
        if c in df.columns:
            df[c] = (
                pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
    return df


def build_metadata_from_directory(data_root: str,
                                  logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Scan ``data_root`` recursively and build a metadata table from DICOM tags.

    Used as a fallback when no metadata CSV is provided. Reads only headers
    (``stop_before_pixels=True``) so it is fast and memory efficient. Cancer /
    False_Positive default to 0 and should be corrected with ground-truth labels.
    """
    import pydicom

    log = logger or get_logger()
    rows: List[Dict] = []
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            if not fn.lower().endswith((".dcm", ".dicom")) and "." in fn:
                continue
            fpath = os.path.join(dirpath, fn)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            lat = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")) or "")
            view = str(getattr(ds, "ViewPosition", "") or "")
            rows.append(
                {
                    "Patient_ID": str(getattr(ds, "PatientID", os.path.basename(dirpath))),
                    "Age": _parse_age(getattr(ds, "PatientAge", None)),
                    "Image_Laterality": lat,
                    "View_Position": view,
                    "Cancer": 0,
                    "False_Positive": 0,
                    "Image_Path": fpath,
                }
            )
    log.info("Scanned %s -> %d DICOM files", data_root, len(rows))
    if not rows:
        raise FileNotFoundError(f"No DICOM files found under {data_root}")
    return normalise_metadata(pd.DataFrame(rows))


def _parse_age(age_str) -> float:
    """Parse a DICOM Age String like '045Y' into a float number of years."""
    if age_str is None:
        return float("nan")
    s = str(age_str).strip().upper()
    try:
        if s.endswith("Y"):
            return float(s[:-1])
        return float(s)
    except ValueError:
        return float("nan")


def load_metadata(cfg: Config, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load metadata from CSV if present, otherwise build it from DICOM headers."""
    log = logger or get_logger()
    if os.path.isfile(cfg.paths.metadata_csv):
        log.info("Loading metadata CSV: %s", cfg.paths.metadata_csv)
        df = pd.read_csv(cfg.paths.metadata_csv)
        df = normalise_metadata(df)
    else:
        log.warning(
            "Metadata CSV not found at %s; scanning DICOM directory instead.",
            cfg.paths.metadata_csv,
        )
        df = build_metadata_from_directory(cfg.paths.data_root, log)

    required = ["Patient_ID", "Image_Laterality", "View_Position", "Image_Path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    for c in ("Cancer", "False_Positive"):
        if c not in df.columns:
            df[c] = 0
    if "Age" not in df.columns:
        df["Age"] = float("nan")
    return df


# --------------------------------------------------------------------------- #
# Patient-level table & labels
# --------------------------------------------------------------------------- #
def patient_label(cancer_any: int, fp_any: int) -> int:
    """Aggregate per-image flags into a single patient class.

    Priority: Cancer (1) > False Positive (2) > Normal (0). Cancer dominates so
    that a screening error never masks a true malignancy.
    """
    if cancer_any:
        return 1
    if fp_any:
        return 2
    return 0


def build_patient_table(df: pd.DataFrame,
                        cfg: Config,
                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Collapse the per-image metadata into one row per patient.

    Produces columns: ``Patient_ID``, ``Age``, ``label`` and one path column per
    view in ``VIEW_ORDER`` (``path_LCC`` ...). Missing views are ``NaN`` and are
    masked out downstream.
    """
    log = logger or get_logger()
    view_key = df["Image_Laterality"].astype(str) + df["View_Position"].astype(str)
    df = df.assign(_view_key=view_key)

    records: List[Dict] = []
    for pid, group in df.groupby("Patient_ID"):
        rec: Dict = {"Patient_ID": str(pid)}
        rec["Age"] = float(pd.to_numeric(group["Age"], errors="coerce").mean())
        cancer_any = int(group["Cancer"].max()) if "Cancer" in group else 0
        fp_any = int(group["False_Positive"].max()) if "False_Positive" in group else 0
        rec["label"] = patient_label(cancer_any, fp_any)
        for view in cfg.data.view_order:
            matches = group.loc[group["_view_key"] == view, "Image_Path"].tolist()
            rec[f"path_{view}"] = matches[0] if matches else np.nan
            rec[f"n_{view}"] = len(matches)
        records.append(rec)

    table = pd.DataFrame.from_records(records)
    n_views_avail = table[[f"n_{v}" for v in cfg.data.view_order]].gt(0).sum(axis=1)
    table = table[n_views_avail > 0].reset_index(drop=True)
    log.info(
        "Built patient table: %d patients | class counts %s",
        len(table), table["label"].value_counts().to_dict(),
    )
    return table


def patient_level_split(table: pd.DataFrame,
                        cfg: Config,
                        logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Split *by patient* into train/val/test with label stratification.

    Adds a ``split`` column. Guarantees each ``Patient_ID`` lives in exactly one
    split (the table already has one row per patient).
    """
    from sklearn.model_selection import train_test_split

    log = logger or get_logger()
    seed = cfg.data.seed
    labels = table["label"].values

    # Fall back to non-stratified split when a class is too small to stratify.
    def _can_stratify(y) -> bool:
        _, counts = np.unique(y, return_counts=True)
        return counts.min() >= 2

    strat = labels if _can_stratify(labels) else None
    train_df, temp_df = train_test_split(
        table, test_size=(cfg.data.val_ratio + cfg.data.test_ratio),
        random_state=seed, stratify=strat,
    )
    rel_test = cfg.data.test_ratio / (cfg.data.val_ratio + cfg.data.test_ratio)
    strat_temp = temp_df["label"].values if _can_stratify(temp_df["label"].values) else None
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=seed, stratify=strat_temp,
    )

    table = table.copy()
    table["split"] = "train"
    table.loc[table["Patient_ID"].isin(val_df["Patient_ID"]), "split"] = "val"
    table.loc[table["Patient_ID"].isin(test_df["Patient_ID"]), "split"] = "test"

    for name in ("train", "val", "test"):
        sub = table[table["split"] == name]
        log.info("Split %-5s: %4d patients | classes %s",
                 name, len(sub), sub["label"].value_counts().sort_index().to_dict())
    return table


def compute_class_weights(labels: Sequence[int], num_classes: int,
                          power: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights normalised to mean 1.0.

    ``power`` softens extreme ratios: 1.0 is full inverse-frequency, 0.5 is
    the (inverse-frequency)**0.5 ("square-root") variant. Full inverse-
    frequency weighting stacked on top of focal loss's own dynamic
    down-weighting of easy examples over-corrects for imbalance and can
    collapse the model to predicting a single class (confirmed empirically on
    NLBS: 29x weight ratio between Cancer/Normal caused exactly this).
    ``power=0.5`` is the recommended default for this project.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = (counts.sum() / (num_classes * counts)) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``. Down-weights easy
    examples, which is well suited to the imbalanced NLBS three-class problem.
    """

    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        # Cross entropy with optional per-class (alpha) weighting.
        ce = F.nll_loss(
            log_probs, target, weight=self.weight, reduction="none",
        )
        # True-class probability p_t, computed directly for the modulating term.
        probs = log_probs.exp()
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce
        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=1) * self.label_smoothing
            loss = (1 - self.label_smoothing) * loss + smooth
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: Config, class_weights: Optional[torch.Tensor]) -> nn.Module:
    """Instantiate the configured loss (focal or weighted cross entropy)."""
    weight = class_weights if cfg.train.class_weighting else None
    if cfg.train.loss == "focal":
        return FocalLoss(
            gamma=cfg.train.focal_gamma,
            weight=weight,
            label_smoothing=cfg.train.label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)


# --------------------------------------------------------------------------- #
# Checkpoint IO
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None,
                    epoch: int = 0, best_metric: float = 0.0,
                    extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None,
                    map_location: str = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# --------------------------------------------------------------------------- #
# Misc numeric helpers
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Running average of a scalar quantity."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(cfg: Config, device: torch.device) -> Tuple[bool, torch.dtype, bool]:
    """Resolve mixed-precision settings for the current device.

    Returns ``(use_amp, autocast_dtype, use_grad_scaler)``. bf16 is preferred
    (no gradient scaler and composes cleanly with SAM); fp16 falls back to a
    ``GradScaler`` but only when SAM is *not* in use.
    """
    use_amp = cfg.train.use_amp and device.type == "cuda"
    dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
    if use_amp and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    use_scaler = use_amp and dtype == torch.float16 and not cfg.train.use_sam
    return use_amp, dtype, use_scaler


def autocast_context(device: torch.device, use_amp: bool, dtype: torch.dtype):
    """Return an autocast context manager (or a no-op when AMP is disabled)."""
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    """Overlay a normalised [0,1] heatmap on a grayscale [0,1] image.

    Returns an RGB uint8 image suitable for saving with matplotlib / cv2.
    """
    import cv2

    cmap_id = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    hm = np.clip(heatmap, 0, 1)
    hm_uint8 = (hm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cmap_id)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    img = np.clip(image, 0, 1)
    img_rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
    img_rgb = (img_rgb * 255).astype(np.uint8)

    overlay = (alpha * colored + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay
