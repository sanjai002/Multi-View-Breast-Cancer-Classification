"""
config.py

Central configuration for the NLBS Multi-View Breast Cancer Classification project.
All project settings are defined here.

"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# Raw Dataset
# -------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/NewVolume/projects/breast cancer detection")

NORMAL_DIR = DATA_ROOT / "normal"
ABNORMAL_DIR = DATA_ROOT / "abnormal"

# -------------------------------------------------------------------------
# Generated Data
# -------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CACHE_DIR = OUTPUT_DIR / "preproc_cache"

METADATA_CSV = OUTPUT_DIR / "metadata.csv"

PATIENT_MANIFEST = OUTPUT_DIR / "patient_manifest.csv"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

LOG_DIR = OUTPUT_DIR / "logs"

TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# =============================================================================
# DATASET
# =============================================================================

CLASS_NAMES = [
    "normal",
    "abnormal",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224

NUM_VIEWS = 4

VIEW_ORDER = [
    "LCC",
    "LMLO",
    "RCC",
    "RMLO",
]

# Ignore False Positive cases
USE_FALSE_POSITIVE = False

# =============================================================================
# SPLIT
# =============================================================================

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42

# =============================================================================
# TRAINING
# =============================================================================

MODEL_NAME = "resnet50"

PRETRAINED = True

BATCH_SIZE = 8

NUM_WORKERS = 4

EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.0

DROPOUT = 0.5

# =============================================================================
# OPTIMIZER
# =============================================================================

OPTIMIZER = "adamw"

# =============================================================================
# LR Scheduler
# =============================================================================

SCHEDULER = "cosine"

MIN_LR = 1e-6

# =============================================================================
# EARLY STOPPING
# =============================================================================

EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# IMAGE NORMALIZATION
# =============================================================================

MEAN = [0.485]

STD = [0.229]

# =============================================================================
# CACHE
# =============================================================================

CACHE_DTYPE = "uint8"

CACHE_EXTENSION = ".npy"

VERIFY_CACHE_BEFORE_TRAINING = True

ALLOW_DICOM_DURING_TRAINING = False

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = "cuda"

USE_AMP = True

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

DETERMINISTIC = True

# =============================================================================
# CREATE DIRECTORIES
# =============================================================================

OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)

TENSORBOARD_DIR.mkdir(exist_ok=True)

GRADCAM_DIR.mkdir(exist_ok=True)