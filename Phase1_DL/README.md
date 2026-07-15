# Phase 1 DL Pipeline

This document describes the current Phase 1 deep learning pipeline for the breast cancer detection project.
It covers dataset layout, metadata, caching, training/test split behavior, and the exact commands to run the pipeline locally and in Colab.

## Project structure

- `Phase1_DL/`
  - `config/` — configuration dataclasses for preprocessing, data, model, and training.
  - `preprocessing/` — DICOM decoding, breast segmentation, orientation normalization, intensity normalization, and preprocessing pipeline.
  - `cache_system/` — deterministic cache key generation, cache file path resolution, and cache generation logic.
  - `metadata/` — image-level metadata scanning and patient-level manifest building.
  - `datasets/` — cached PyTorch dataset implementation that reads only `.npy` cache files.
  - `models/` — small fusion model proof-of-concept.
  - `scripts/` — test scripts and cache generation tools.
  - `outputs/` — generated cache, manifests, checkpoints, tensorboard logs, and helper files.

## Data sources

- `Phase1_DL/data/metadata.csv` is the primary image-level metadata file.
- It currently contains `26988` image rows and `5997` unique patients.
- Each row includes:
  - `Patient_ID`
  - `Age`
  - `Image_Laterality` (L or R)
  - `View_Position` (CC, MLO)
  - `Cancer` (0 normal, 1 abnormal)
  - `Image_Path` (absolute DICOM path)

## Patient manifest

- `Phase1_DL/outputs/patient_manifest.csv` is the patient-level manifest generated from the image-level metadata.
- It contains one row per patient with the following fields:
  - `Patient_ID`
  - `Age`
  - `Cancer`
  - `path_LCC`, `path_LMLO`, `path_RCC`, `path_RMLO`
  - `n_LCC`, `n_LMLO`, `n_RCC`, `n_RMLO`
- Current manifest counts:
  - `5997` total patients
  - `149` abnormal patients
  - `5848` normal patients

## Cache design

### Purpose

- Training must not decode DICOM files during dataset loading.
- All preprocessing is done once and saved as `.npy` cache files.
- The dataset loader reads only cached `.npy` files and treats missing views as zero tensors.

### Cache key generation

- Cache keys are deterministic SHA-256 hashes of `relative_path + image_size`.
- The system supports portability by resolving the data root using either:
  1. `NLBS_ORIGINAL_DATA_ROOT` environment variable
  2. `Phase1_DL/outputs/cache_origin.txt`
  3. fallback to the raw absolute path string
- This avoids cache misses when the dataset is mounted in a different base directory on Colab.

### Existing cache state

- Currently only `16` cache files were present from the small test run.
- After the balanced generation step, `1311` cached images are available for the selected cohort.

## Balanced training cohort

- A balanced patient subset was created containing:
  - `149` abnormal patients
  - `149` matched normal patients
- The generated files are:
  - `Phase1_DL/outputs/patient_manifest_balanced.csv`
  - `Phase1_DL/outputs/metadata_balanced.csv`
- Cache generation for this balanced cohort processed `1311` images successfully with `0` failures.

## Scripts

### `scripts/test_cache_and_train.py`

- Builds a small metadata subset from `Phase1_DL/data/metadata.csv`.
- Generates cache for the first 16 images.
- Builds the patient manifest if missing.
- Runs a short PyTorch training loop using cached images only.

### `scripts/generate_balanced_patient_cache.py`

- Selects all abnormal patients and an equal number of normal patients.
- Writes a balanced patient manifest and metadata CSV.
- Generates cache for the balanced cohort.

### `prepare_colab_bundle.sh`

- Creates `colab_bundle/code.zip` containing the codebase.
- Creates `colab_bundle/data_bundle.zip` containing:
  - `data/metadata.csv`
  - `data/patient_manifest.csv` (if present)
  - `data/cache/` with cached `.npy` files
  - `data/cache_origin.txt`

## How to run locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the patient manifest

If `Phase1_DL/outputs/patient_manifest.csv` is missing:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from config.base import get_config
from metadata.patient_manifest import PatientManifestBuilder
cfg = get_config()
PatientManifestBuilder(cfg).build_manifest(cfg.metadata_csv, cfg.patient_manifest_csv)
PY
```

### 3. Generate balanced cache

```bash
python3 scripts/generate_balanced_patient_cache.py
```

This creates the balanced metadata and manifest files and caches all images for the selected cohort.

### 4. Run the short training smoke test

```bash
PYTHONPATH=Phase1_DL python3 scripts/test_cache_and_train.py
```

### 5. Build the Colab bundles

```bash
bash prepare_colab_bundle.sh
```

## How training works in the DL phase

### Dataset loading

- `Phase1_DL/datasets/cached_dataset.py` provides a `CachedMammoDataset` that:
  - loads patient rows from a manifest CSV
  - resolves view paths using the cache generator
  - reads `.npy` files only
  - returns 4 views in fixed order: `LCC`, `LMLO`, `RCC`, `RMLO`
  - fills missing views with zeros and uses a binary presence mask

### Model

- A small fusion model proof-of-concept exists in `Phase1_DL/models/simple_fusion.py`.
- It uses a shared backbone and masked fusion over the four views.

### Training loop

- The current smoke-test training loop uses:
  - Adam optimizer
  - Cross-entropy loss
  - batch size `2`
  - a few update steps for verification
- A production training loop is not yet implemented in full, but the current pipeline proves the cache-only workflow.

## Notes and caveats

- The DICOM paths in `Phase1_DL/data/metadata.csv` are absolute. Use `NLBS_ORIGINAL_DATA_ROOT` or `cache_origin.txt` for cross-machine cache portability.
- Training currently uses a balanced pilot cohort of `298` patients.
- For full dataset training, generate cache for all image rows in `Phase1_DL/data/metadata.csv` and build a corresponding full patient manifest.

## Useful commands summary

```bash
# Small smoke test
PYTHONPATH=Phase1_DL python3 scripts/test_cache_and_train.py

# Balanced cache generation
python3 scripts/generate_balanced_patient_cache.py

# Build Colab bundle
bash prepare_colab_bundle.sh
```

## What to do next

- Add a full training script with split-aware dataset creation, optimizer schedule, checkpointing, and evaluation.
- Add an explicit stratified train/val/test split in the patient manifest.
- Add a Colab notebook cell that uses `NLBS_ORIGINAL_DATA_ROOT` and verifies the unpacked cache bundle.
