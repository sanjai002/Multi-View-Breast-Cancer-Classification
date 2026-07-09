"""Export final Phase 1 results on Colab, reading/writing the Drive-persisted
checkpoint and outputs produced by ``run_colab_training.py``.

Usage (see NLBS_Colab_Training.ipynb, step 8):
    NLBS_TEST_CHECKPOINT_DIR=/content/drive/MyDrive/NLBS_Phase1/checkpoints \
    NLBS_TEST_OUTPUT_DIR=/content/drive/MyDrive/NLBS_Phase1/outputs \
    NLBS_TEST_MANIFEST=/content/nlbs_data/patient_manifest.csv \
    NLBS_TEST_METADATA=/content/nlbs_data/metadata.csv \
    python run_colab_test.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import get_config
from run_colab_training import ORIGINAL_LOCAL_DATA_ROOT
from training.test import (evaluate_test_split, export_features_and_predictions,
                           generate_explanations, load_tables, load_trained_model)
from utils import get_device, get_logger


def main() -> None:
    cfg = get_config()
    cfg.paths.data_root = ORIGINAL_LOCAL_DATA_ROOT
    cfg.data.image_size = 224
    cfg.data.cache_preprocessed = True
    cfg.data.cache_dir = os.environ.get(
        "NLBS_TEST_CACHE_DIR", "/content/nlbs_data/preproc_cache"
    )
    cfg.data.num_workers = 2

    cfg.paths.checkpoint_dir = os.environ["NLBS_TEST_CHECKPOINT_DIR"]
    cfg.paths.output_dir = os.environ["NLBS_TEST_OUTPUT_DIR"]
    cfg.paths.manifest_csv = os.environ["NLBS_TEST_MANIFEST"]
    cfg.paths.metadata_csv = os.environ["NLBS_TEST_METADATA"]
    cfg.paths.gradcam_dir = os.path.join(cfg.paths.output_dir, "gradcam_images")
    cfg.paths.log_dir = os.path.join(cfg.paths.output_dir, "..", "logs")
    cfg.create_dirs()

    logger = get_logger("phase1_colab_test", cfg.paths.log_dir)
    device = get_device()
    logger.info("Device: %s", device)

    model = load_trained_model(cfg, device, logger)
    table = load_tables(cfg, logger)

    metrics = evaluate_test_split(model, cfg, table, device, logger)
    export_features_and_predictions(model, cfg, table, device, logger)
    generate_explanations(model, cfg, table, device, logger)

    import json
    with open(os.path.join(cfg.paths.output_dir, "test_metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                  f, indent=2, default=float)
    logger.info("Export complete. Artefacts in %s (Google Drive)", cfg.paths.output_dir)


if __name__ == "__main__":
    main()
