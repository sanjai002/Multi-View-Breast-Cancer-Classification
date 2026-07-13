"""Central configuration for Phase 1 (Deep Learning) of the NLBS breast cancer
classification project.

All hyper-parameters, paths and runtime options live here so that every other
module imports a single, immutable-ish ``Config`` object. Nested dataclasses keep
related options grouped (``cfg.data.image_size``, ``cfg.train.epochs`` ...).

The configuration is intentionally environment agnostic: paths default to
locations relative to this file so the project runs unchanged on a local Ubuntu
workstation or inside Google Colab. Override any field from the command line
entry points or by editing :func:`get_config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Absolute path to the Phase1_DL project root (directory containing this file).
PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))

# Canonical ordering of the four standard screening mammography views. This
# ordering is used *everywhere* (dataset stacking, model forward, feature
# export, Grad-CAM row mapping) so it MUST remain stable.
VIEW_ORDER: Tuple[str, ...] = ("LCC", "LMLO", "RCC", "RMLO")


# --------------------------------------------------------------------------- #
# Path configuration
# --------------------------------------------------------------------------- #
@dataclass
class PathConfig:
    """Filesystem locations for inputs and generated artefacts."""

    project_root: str = PROJECT_ROOT

    # Root directory that contains the NLBS DICOM files (searched recursively
    # when auto-building metadata). Override to point at your dataset.
    data_root: str = os.environ.get(
        "NLBS_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "NLBS")
    )

    # Optional pre-built metadata CSV. When it does not exist the pipeline will
    # scan ``data_root`` and build one from DICOM headers.
    metadata_csv: str = os.environ.get(
        "NLBS_METADATA_CSV", os.path.join(PROJECT_ROOT, "data", "metadata.csv")
    )

    # Where the patient-level split manifest is cached.
    manifest_csv: str = os.path.join(PROJECT_ROOT, "outputs", "patient_manifest.csv")

    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    tensorboard_dir: str = os.path.join(PROJECT_ROOT, "tensorboard")
    log_dir: str = os.path.join(PROJECT_ROOT, "logs")
    gradcam_dir: str = os.path.join(PROJECT_ROOT, "outputs", "gradcam_images")

    def all_dirs(self) -> List[str]:
        return [
            self.output_dir,
            self.checkpoint_dir,
            self.tensorboard_dir,
            self.log_dir,
            self.gradcam_dir,
        ]


# --------------------------------------------------------------------------- #
# Data / preprocessing configuration
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Dataset, preprocessing and splitting options."""

    # Column names expected in the metadata table. The loader is tolerant to a
    # handful of common spelling variants (see ``utils.normalise_metadata``).
    col_patient_id: str = "Patient_ID"
    col_age: str = "Age"
    col_laterality: str = "Image_Laterality"
    col_view: str = "View_Position"
    col_cancer: str = "Cancer"
    col_path: str = "Image_Path"

    # Binary problem: 0=Normal, 1=Abnormal/Cancer.
    num_classes: int = 2
    class_names: Tuple[str, ...] = ("Normal", "Abnormal")

    image_size: int = 512          # square side length fed to the network
    in_channels: int = 1           # mammograms are grayscale; conv1 is adapted

    # Preprocessing knobs.
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    breast_threshold_method: str = "otsu"   # "otsu" | "triangle"
    min_breast_area_ratio: float = 0.005    # drop segmentation smaller than this
    flip_to: str = "left"          # orient every breast so chest wall is on the "left"

    # Normalisation applied after scaling to [0, 1] (single channel).
    normalize_mean: Tuple[float, ...] = (0.5,)
    normalize_std: Tuple[float, ...] = (0.5,)

    # Patient-level split ratios (must sum to 1.0).
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # Optionally oversample minority-class examples on top of the class-weighted
    # loss. The patient table is already balanced before splitting, so this is
    # usually unnecessary.
    use_balanced_sampler: bool = False

    # Optional on-disk cache of preprocessed views (npy). Off by default to
    # honour the "never load the entire dataset into RAM" requirement.
    cache_preprocessed: bool = False
    cache_dir: str = os.path.join(PROJECT_ROOT, "outputs", "preproc_cache")

    view_order: Tuple[str, ...] = VIEW_ORDER


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Backbone and multi-view fusion architecture options."""

    backbone: str = "resnet50"
    pretrained: bool = True
    feature_dim: int = 2048        # ResNet-50 penultimate dimension
    embed_dim: int = 512           # per-view / patient embedding size
    dropout: float = 0.4
    se_reduction: int = 16         # squeeze-and-excitation reduction ratio
    attention_hidden: int = 256    # hidden size of the gated attention module
    freeze_backbone: bool = True   # start with a frozen backbone (transfer learning)


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation, scheduling and regularisation options."""

    epochs: int = 60
    batch_size: int = 8            # patients per batch (each = 4 views)

    # Differential learning rates for transfer learning.
    head_lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 1e-4

    # Sharpness-Aware Minimisation.
    use_sam: bool = True
    sam_rho: float = 0.05
    sam_adaptive: bool = False

    # Learning-rate schedule. "cosine" uses cosine annealing with warm-up;
    # "plateau" uses ReduceLROnPlateau on the monitored validation metric.
    scheduler: str = "cosine"      # "cosine" | "plateau"
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    plateau_patience: int = 4
    plateau_factor: float = 0.5

    # Mixed precision. bf16 needs no gradient scaler and composes cleanly with
    # SAM; fp16 falls back to a GradScaler.
    use_amp: bool = True
    amp_dtype: str = "bf16"        # "bf16" | "fp16"

    grad_clip_norm: float = 5.0

    # Exponential moving average of weights.
    use_ema: bool = True
    ema_decay: float = 0.999

    # Loss. Both are implemented; "focal" is the recommended default for the
    # binary NLBS problem.
    loss: str = "focal"            # "focal" | "weighted_ce"
    focal_gamma: float = 2.0
    class_weighting: bool = True   # weight the loss by inverse class frequency
    class_weight_power: float = 0.5  # soften extremes: 1.0=full inverse-freq, 0.5=sqrt (recommended)
    label_smoothing: float = 0.0

    # Batch-level mixing augmentation.
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.3          # probability a batch is mixed at all

    # Progressive unfreezing schedule: epoch -> ResNet stage to unfreeze.
    # Stages are unfrozen from the top (closest to the head) downwards.
    unfreeze_schedule: Dict[int, str] = field(
        default_factory=lambda: {5: "layer4", 12: "layer3", 20: "layer2", 30: "layer1"}
    )

    # Early stopping / checkpoint monitoring.
    save_every_epoch: bool = True  # keep a per-epoch checkpoint (checkpoints/epoch_XXX.pth)
    resume: bool = False           # resume from checkpoints/last.pth if it exists
    monitor: str = "val_macro_f1"  # metric key produced by evaluation.metrics
    monitor_mode: str = "max"      # "max" | "min"
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4

    log_every_n_steps: int = 10


# --------------------------------------------------------------------------- #
# Explainability configuration
# --------------------------------------------------------------------------- #
@dataclass
class ExplainConfig:
    """Options for Grad-CAM / attribution generation."""

    num_samples: int = 24          # test patients to visualise
    target_layer: str = "layer4"   # ResNet stage hooked for CAMs
    scorecam_batch: int = 32       # channels forwarded per ScoreCAM batch
    scorecam_channels: int = 128   # top-K activation channels used by ScoreCAM
    ig_steps: int = 32             # integration steps for Integrated Gradients
    overlay_alpha: float = 0.45
    colormap: str = "jet"


# --------------------------------------------------------------------------- #
# Root configuration object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)

    experiment_name: str = "nlbs_multiview_resnet50"

    def create_dirs(self) -> None:
        """Create every output directory. Safe to call repeatedly."""
        for d in self.paths.all_dirs():
            os.makedirs(d, exist_ok=True)
        if self.data.cache_preprocessed:
            os.makedirs(self.data.cache_dir, exist_ok=True)

    def validate(self) -> None:
        total = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        if self.train.amp_dtype not in ("bf16", "fp16"):
            raise ValueError("amp_dtype must be 'bf16' or 'fp16'")
        if self.train.scheduler not in ("cosine", "plateau"):
            raise ValueError("scheduler must be 'cosine' or 'plateau'")
        if self.train.loss not in ("focal", "weighted_ce"):
            raise ValueError("loss must be 'focal' or 'weighted_ce'")
        if self.model.freeze_backbone and self.train.use_sam and self.train.amp_dtype == "fp16":
            # fp16 + SAM needs a scaler and two backward passes; supported but
            # bf16 is strongly recommended.
            pass

    def to_dict(self) -> Dict:
        return asdict(self)


def get_config() -> Config:
    """Factory returning a validated default configuration."""
    cfg = Config()
    cfg.validate()
    return cfg


if __name__ == "__main__":
    import json

    cfg = get_config()
    cfg.create_dirs()
    print(json.dumps(cfg.to_dict(), indent=2))
