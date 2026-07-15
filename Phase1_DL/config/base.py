"""Base configuration dataclasses with validation."""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PreprocessingConfig:
    """DICOM preprocessing parameters."""

    image_size: int = 512
    """Target image size (H=W)."""

    clahe_clip_limit: float = 2.0
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) clip limit."""

    clahe_grid_size: int = 8
    """CLAHE grid size."""

    breast_threshold_method: str = "otsu"
    """Breast segmentation method: 'otsu' or 'triangle'."""

    min_breast_area_ratio: float = 0.01
    """Drop segmentation if breast area < this fraction of total."""

    normalize_method: str = "zscore"
    """Intensity normalization: 'zscore', 'minmax', or 'none'."""

    cache_dir: Optional[Path] = None
    """Cache directory for preprocessed images."""

    def __post_init__(self) -> None:
        """Validate preprocessing config."""
        if self.image_size not in (224, 512):
            raise ValueError(f"image_size must be 224 or 512, got {self.image_size}")
        if self.breast_threshold_method not in ("otsu", "triangle"):
            raise ValueError(
                f"breast_threshold_method must be 'otsu' or 'triangle', "
                f"got {self.breast_threshold_method}"
            )
        if self.cache_dir is None:
            self.cache_dir = PROJECT_ROOT / "outputs" / "cache"


@dataclass
class DataConfig:
    """Dataset and splitting parameters."""

    train_ratio: float = 0.70
    """Fraction of patients for training."""

    val_ratio: float = 0.15
    """Fraction of patients for validation."""

    test_ratio: float = 0.15
    """Fraction of patients for testing."""

    seed: int = 42
    """Random seed for reproducibility."""

    num_workers: int = 4
    """DataLoader workers."""

    pin_memory: bool = True
    """Pin memory for faster GPU transfer."""

    batch_size: int = 8
    """Batch size (number of patients)."""

    def __post_init__(self) -> None:
        """Validate data config."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    backbone: str = "convnext_large"
    """Backbone architecture: 'convnext_large', 'resnet50', 'efficientnetv2_l', etc."""

    pretrained: bool = True
    """Use ImageNet-1K pretrained weights."""

    fusion_strategy: str = "attention"
    """Multi-view fusion: 'early', 'late', 'attention'."""

    missing_view_strategy: str = "mask"
    """Handle missing views: 'mask', 'pad', 'drop'."""

    num_classes: int = 2
    """Number of output classes (binary: Normal=0, Abnormal=1)."""

    dropout: float = 0.3
    """Dropout rate in classification head."""

    feature_dim: Optional[int] = None
    """Feature dimension (auto-detected from backbone)."""

    def __post_init__(self) -> None:
        """Validate model config."""
        if self.num_classes != 2:
            raise ValueError(f"Only binary classification supported, got {self.num_classes}")
        if self.fusion_strategy not in ("early", "late", "attention"):
            raise ValueError(
                f"fusion_strategy must be 'early', 'late', or 'attention', "
                f"got {self.fusion_strategy}"
            )


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 60
    """Number of training epochs."""

    learning_rate: float = 1e-4
    """Initial learning rate."""

    weight_decay: float = 1e-5
    """L2 regularization coefficient."""

    grad_clip: float = 1.0
    """Gradient clipping norm."""

    loss_fn: str = "focal"
    """Loss function: 'focal', 'weighted_ce', 'ce'."""

    focal_gamma: float = 2.0
    """Focal loss gamma parameter."""

    label_smoothing: float = 0.1
    """Label smoothing amount."""

    use_amp: bool = True
    """Use automatic mixed precision (AMP)."""

    amp_dtype: str = "fp16"
    """AMP data type: 'fp16' or 'bf16'."""

    scheduler: str = "cosine"
    """LR scheduler: 'cosine', 'plateau', 'linear'."""

    warmup_epochs: int = 5
    """Warmup epochs for cosine scheduler."""

    early_stopping_patience: int = 10
    """Early stopping patience (epochs)."""

    early_stopping_min_delta: float = 1e-4
    """Minimum improvement for early stopping."""

    checkpoint_dir: Optional[Path] = None
    """Checkpoint directory."""

    tensorboard_dir: Optional[Path] = None
    """TensorBoard log directory."""

    def __post_init__(self) -> None:
        """Validate training config."""
        if self.loss_fn not in ("focal", "weighted_ce", "ce"):
            raise ValueError(f"loss_fn must be 'focal', 'weighted_ce', or 'ce', got {self.loss_fn}")
        if self.scheduler not in ("cosine", "plateau", "linear"):
            raise ValueError(
                f"scheduler must be 'cosine', 'plateau', or 'linear', got {self.scheduler}"
            )
        if self.amp_dtype not in ("fp16", "bf16"):
            raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got {self.amp_dtype}")
        if self.checkpoint_dir is None:
            self.checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
        if self.tensorboard_dir is None:
            self.tensorboard_dir = PROJECT_ROOT / "outputs" / "tensorboard"


@dataclass
class Config:
    """Root configuration object."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    metadata_csv: Optional[Path] = None
    """Path to metadata.csv."""

    patient_manifest_csv: Optional[Path] = None
    """Path to patient_manifest.csv."""

    output_dir: Optional[Path] = None
    """Root output directory."""

    def __post_init__(self) -> None:
        """Set defaults and create directories."""
        if self.metadata_csv is None:
            self.metadata_csv = PROJECT_ROOT / "data" / "metadata.csv"
        if self.patient_manifest_csv is None:
            self.patient_manifest_csv = PROJECT_ROOT / "outputs" / "patient_manifest.csv"
        if self.output_dir is None:
            self.output_dir = PROJECT_ROOT / "outputs"

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessing.cache_dir.mkdir(parents=True, exist_ok=True)
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        data = asdict(self)
        # Convert Path objects to strings
        return self._convert_paths_to_str(data)

    @staticmethod
    def _convert_paths_to_str(obj: Any) -> Any:
        """Recursively convert Path objects to strings."""
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: Config._convert_paths_to_str(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(Config._convert_paths_to_str(v) for v in obj)
        return obj

    def save_json(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def get_config() -> Config:
    """Get default config."""
    return Config()
