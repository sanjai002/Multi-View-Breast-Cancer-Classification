# Phase 1 — Deep Learning Binary Breast Cancer Classification (NLBS)

Deep-learning, multi-view mammography classifier for the **Newfoundland and
Labrador Breast Screening (NLBS)** dataset. Binary patient-level
classification from the four standard screening views (LCC, LMLO, RCC, RMLO):

| Class | Meaning        |
|:-----:|:---------------|
| 0     | Normal         |
| 1     | Abnormal       |

> This is a **standalone** project. It contains no reinforcement learning. Its
> exported CNN feature vectors and predictions become the **inputs to Phase 2**
> (`../Phase2_RL/`).

---

## 1. Architecture

```
          LCC  LMLO  RCC  RMLO            (4 DICOM views / patient)
            │    │    │    │
            ▼    ▼    ▼    ▼
     ┌───────────────────────────┐
     │  Shared ResNet-50 backbone │  ImageNet-pretrained, conv1 adapted to 1ch
     └───────────────────────────┘
            │ (per-view 2048-d feature map)
            ▼
        SE block  →  GAP  →  Linear projection  →  per-view embedding (512-d)
            │
   ┌────────┴─────────┐        Dual-branch multi-view fusion
   ▼                  ▼
 CC branch         MLO branch    (masked gated attention pooling)
 {LCC,RCC}         {LMLO,RMLO}
   └────────┬─────────┘
            ▼
     Attention fusion  →  patient embedding (512-d)  →  MLP head  →  2 logits
```

* **Single shared backbone** across views (memory efficient, standard for
  multi-view mammography) — enables clean per-view Grad-CAM in one pass.
* **SE blocks** recalibrate channels; **gated attention pooling** fuses views
  and branches while **masking missing views**.
* Missing views are zero-filled and excluded via the attention mask, so patients
  with fewer than four views are handled gracefully.

## 2. Pipeline highlights

* **DICOM** (`dicom.py`): VOI LUT windowing, MONOCHROME1 inversion, rescale
  slope/intercept, robust [0,1] normalisation, compressed transfer-syntax
  decoding (pylibjpeg / GDCM).
* **Preprocessing** (`preprocess.py`): CLAHE → breast segmentation
  (Otsu + largest connected component) → artifact/label removal →
  black-border crop → aspect-preserving resize → orientation normalisation.
* **Augmentation** (`augmentation.py`): rotation, affine, brightness, contrast,
  gamma, random resized crop, Gaussian noise, random erasing (Albumentations) +
  batch-level **MixUp / CutMix**.
* **Training** (`training/train.py`): mixed precision (bf16/fp16), **SAM**
  optimiser, **EMA**, cosine / plateau scheduling with warm-up, gradient
  clipping, **progressive unfreezing** with **differential learning rates**,
  early stopping, checkpointing, TensorBoard.
* **Loss**: Focal (default, recommended) vs. class-weighted cross entropy —
  switch with `--loss`.
* **Evaluation**: accuracy, precision, recall, F1, ROC AUC, sensitivity,
  specificity, confusion matrix, calibration (ECE).
* **Explainability**: Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients.

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
```

Google Colab: upload `Phase1_DL/`, `pip install -r requirements.txt`, then add
the project root to the path before importing:

```python
import sys; sys.path.insert(0, "/content/Phase1_DL")
```

## 4. Data

Point the pipeline at your DICOM root and (optionally) a metadata CSV:

```bash
export NLBS_DATA_ROOT=/path/to/NLBS/dicoms
export NLBS_METADATA_CSV=/path/to/metadata.csv   # optional
```

Expected metadata columns (spelling variants auto-normalised):
`Patient_ID, Age, Image_Laterality, View_Position, Cancer, Image_Path`.

If no CSV is given, metadata is built by scanning DICOM headers (the
`Cancer` flags then default to 0 and must be supplied from ground truth). Patient labels aggregate to binary labels: **Normal=0, Abnormal=1**. False Positive patients are dropped before the patient table is built.

**Splitting is strictly patient-level** (70 / 15 / 15, label-stratified); a
`Patient_ID` never appears in more than one split.

## 5. Usage

```bash
# Quick smoke test (2 epochs, few patients) to validate the whole pipeline
python -m training.train --epochs 2 --limit 40

# Full training
python -m training.train

# Evaluate best checkpoint + export all Phase-1 artefacts
python -m training.test

# Monitor
tensorboard --logdir tensorboard/
```

## 6. Outputs (consumed by Phase 2)

Written to `outputs/` and `checkpoints/`:

| File | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Best checkpoint (weights + epoch + config) |
| `checkpoints/best_weights.pth` | Best model `state_dict` |
| `checkpoints/feature_extractor.pth` | Encoder weights (no classifier head) |
| `outputs/patient_features.npy` | Per-patient embedding `(N, 512)` |
| `outputs/image_features.npy` | Per-view embedding `(N·4, 512)` |
| `outputs/patient_feature_index.csv` | Row → patient / split / binary label / pred |
| `outputs/image_feature_index.csv` | Row → patient / view / split / present |
| `outputs/prediction_probabilities.csv` | Per-patient class probabilities |
| `outputs/patient_predictions.csv` | Per-patient predicted vs. true class |
| `outputs/classification_report.pdf` | Full metric report |
| `outputs/confusion_matrix.png`, `roc_curve.png`, `precision_recall.png`, `calibration_curve.png` | Diagnostic plots |
| `outputs/gradcam_images/` | Per-patient Grad-CAM/++/Score-CAM/IG overlays |

## 7. Project layout

```
Phase1_DL/
├── config.py            # all hyper-parameters (nested dataclasses)
├── dicom.py             # DICOM reading (VOI LUT, MONOCHROME1, decode)
├── preprocess.py        # CLAHE, segmentation, artifact removal, orient, resize
├── augmentation.py      # Albumentations + MixUp / CutMix
├── dataset.py           # on-demand multi-view Dataset + DataLoaders
├── utils.py             # logging, seeding, splitting, losses, AMP, checkpoints
├── models/              # resnet50.py · attention.py · fusion.py
├── training/            # train.py · validate.py · test.py · callbacks.py
├── evaluation/          # metrics.py · confusion_matrix.py · roc_curve.py ·
│                        # precision_recall.py · calibration_curve.py
├── explainability/      # gradcam.py · gradcam_plus.py · scorecam.py ·
│                        # integrated_gradients.py
├── outputs/ checkpoints/ tensorboard/ logs/
└── requirements.txt
```

## 8. Notes & assumptions

* Grayscale input: ResNet-50 `conv1` is adapted from 3→1 channels by **summing**
  the pretrained RGB kernels, keeping the pretrained response intact.
* bf16 mixed precision is preferred (no gradient scaler, composes with SAM);
  fp16 automatically falls back to a `GradScaler` when SAM is off.
* Everything is loaded lazily — the full DICOM dataset is never held in RAM.
* Reproducibility via a single `seed` (NumPy / PyTorch / MONAI / DataLoader
  workers).
