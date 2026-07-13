# Project Documentation — Deep-Learning Multi-View Binary Breast Cancer Classification (NLBS)

**Everything needed to write up Phase 1 for publication.** This document
describes the dataset, methodology, architecture, training protocol, evaluation
and reproducibility details, and points to exactly which generated file supplies
each number/figure in a paper.

> **Title (Phase 1):** *Deep Learning Based Binary Breast Cancer
> Classification using Multi-View Mammography Images from the NLBS Dataset.*

---

## 1. Overview / contribution

We classify screening mammography **at the patient level** into two classes —
**Normal and Abnormal/Cancer** — from the four standard views (L-CC, L-MLO,
R-CC, R-MLO). The model is a **multi-view convolutional network** with a shared
ResNet-50 encoder, squeeze-and-excitation channel attention, and a **dual-branch
(CC / MLO) attention-fusion** head. The learned per-patient feature vectors are
exported to serve as the state representation for a downstream reinforcement-
learning phase (Phase 2).

Key methodological points a reviewer will look for, and how we address them:
- **Patient-level data splitting** (no image-level leakage across train/val/test).
- **Class balance** enforced by dropping false-positive patients and undersampling
  the larger of Normal/Abnormal before patient-level splitting; evaluation still
  uses imbalance-aware metrics (macro-F1, balanced accuracy, per-class
  sensitivity/specificity, ROC-AUC) rather than accuracy alone.
- **Transfer learning** with progressive unfreezing and differential learning
  rates.
- **Explainability** (Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients).
- **Calibration** reporting (reliability curves, ECE).

---

## 2. Dataset — Newfoundland & Labrador Breast Screening (NLBS)

- **Format:** DICOM (uncompressed MONOCHROME2; typical size ≈ 3000 × 2400).
- **Organisation:** `{normal, abnormal}/p_XXXX/{left,left-c,right,right-c}/{CC,MLO}/*.dcm`.
  The **top-level folder is the ground-truth class label**.
- **Metadata:** `NLBSD_Metadata.csv` (per-image: File Path, Image Laterality,
  View Position, Age, Window Center/Width, Cancer flag, "Hard to State Negative").
- **Class mapping used** (`prepare_metadata.py`): `normal → Normal (0)`,
  `abnormal → Abnormal/Cancer (1)`. False Positive patients are excluded before patient-level aggregation.

### Cohort statistics (as used)
| | Patients | Images (raw) | Images used by model |
|---|---:|---:|---:|
| **Total** | 5,997 | 26,988 | 23,816 |
| Normal | 4,332 | 19,942 | — |
| excluded false-positive | 1,516 | 6,394 | — |
| Cancer | 149 | 652 | — |

- The model uses **one image per view slot** per patient (LCC/LMLO/RCC/RMLO);
  3,172 duplicate-view images are therefore not fed in. 5,909/5,997 patients have
  all four views; the remainder are handled by view masking.
- A small number of source DICOMs are corrupt/truncated; the loader logs and
  skips them (blank masked view), so they neither crash training nor bias it.

### Splits (patient-level, label-stratified, seed = 42)
| Split | Patients | Normal | Abnormal |
|---|---:|---:|---:|
| Train (70%) | balanced | balanced | balanced |
| Val (15%) | balanced | balanced | balanced |
| Test (15%) | balanced | balanced | balanced |

The exact assignment is saved to `outputs/patient_manifest.csv` (a `split`
column per `Patient_ID`) — cite this for reproducibility.

---

## 3. Preprocessing (`dicom.py`, `preprocess.py`)

Per image, in order:
1. **DICOM decode** with Modality/Rescale slope-intercept.
2. **VOI LUT / windowing** applied from the header (`apply_voi_lut`).
3. **MONOCHROME1 → MONOCHROME2** inversion when required.
4. **Robust intensity scaling** to [0,1] via 1st–99th percentile clipping.
5. **CLAHE** (clip 2.0, 8×8 tiles) for local contrast enhancement.
6. **Breast segmentation** — Otsu threshold + morphological close/open.
7. **Artifact/label removal** — keep only the largest connected component
   (removes tags, scanner markings).
8. **Black-border removal** — crop to the breast bounding box.
9. **Aspect-preserving resize** to a square canvas (224 CPU run / 512 GPU config).
10. **Orientation normalisation** — flip so the chest wall is on a consistent side
    (estimated from image mass, robust to header inconsistencies).
11. **Standardisation** (mean/std) applied once, in the dataset transform.

Data are **loaded lazily on demand**; the full dataset is never held in RAM. An
optional uint8 on-disk cache accelerates epochs after the first.

---

## 4. Data augmentation (training only; `augmentation.py`)

Per-view (Albumentations): affine (rotate ±15°, scale 0.9–1.1, shear, translate),
random resized crop, brightness/contrast, gamma, Gaussian noise, random erasing —
all clipped to [0,1]. Batch-level: **MixUp** and **CutMix** applied to the stacked
multi-view tensor with interpolated labels. Validation/test use no augmentation.

---

## 5. Model architecture (`models/`)

```
4 views ─► shared ResNet-50 encoder (ImageNet-pretrained, conv1 adapted 3→1 ch by
           summing RGB kernels) ─► SE block ─► GAP ─► Linear proj → 512-d per view
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
              CC branch (LCC,RCC)                               MLO branch (LMLO,RMLO)
              masked gated attention pooling                    masked gated attention pooling
                    └─────────────────────────┬─────────────────────────┘
                              attention fusion over the 2 branch embeddings
                                        │
                              patient embedding (512-d) ─► MLP head ─► 2 logits
```

- **Shared encoder** across views (parameter-efficient, standard for multi-view
  mammography, enables single-pass per-view Grad-CAM).
- **SE blocks** (Hu et al., 2018) for channel recalibration.
- **Gated attention pooling** (Ilse et al., 2018, attention-based MIL) with a
  validity mask so missing views are ignored.
- **Dual-branch fusion:** ipsilateral view-type branches (CC vs MLO) fused, then a
  final attention fusion → single patient embedding.
- Params ≈ **26.0 M total** (2.5 M trainable while the backbone is frozen).
- Exported embeddings: per-view `image_features.npy` and per-patient
  `patient_features.npy` (both 512-d) — the Phase 2 state.

---

## 6. Training methodology (`training/`)

| Component | Setting (CPU run) | Full/GPU config |
|---|---|---|
| Loss | **Focal loss (γ=2)** + inverse-freq class weights | same |
| Class weights | computed from the balanced training split | same |
| Sampler | shuffled balanced source table | same |
| Optimizer | AdamW | AdamW wrapped in **SAM** |
| LR schedule | cosine annealing + 3-epoch warm-up | same |
| Differential LR | head 3e-4 / backbone 3e-5 | same |
| Transfer learning | freeze → **progressive unfreeze** (layer4@5, layer3@12, layer2@20, layer1@30) | same |
| Regularisation | grad-clip 5.0, dropout 0.4, EMA (0.999), MixUp/CutMix | same |
| Mixed precision | off (CPU) | **bf16** |
| Image size / batch | 224 / 4 | 512 / 8 |
| Model selection | best **val macro-F1**; early stopping patience 12 | same |

**Binary balance strategy (important for the paper).** Screening-error patients
are excluded, then the larger of Normal/Abnormal is undersampled with the data
seed before splitting. Focal loss and optional class weighting remain available;
the balanced sampler is normally disabled because the split source table is
already balanced. Model selection and reporting use
**macro-averaged / per-class** metrics, never raw accuracy.

Per-epoch artefacts for the write-up: `outputs/metrics_history.csv`,
`outputs/training_curves.png`, `outputs/epoch_plots/epoch_XXX_val_confusion.png`,
and `checkpoints/epoch_XXX.pth`.

---

## 7. Evaluation protocol (`evaluation/`, `training/test.py`)

The held-out **test set** is evaluated once with the best checkpoint.
Metrics: accuracy, balanced accuracy, macro/weighted precision-recall-F1,
per-class **sensitivity & specificity**, one-vs-rest **ROC-AUC** (macro & micro),
average precision, Cohen's κ, and **calibration** (reliability curves + ECE).

Figures/tables and their source files:
| Paper element | File |
|---|---|
| Headline metrics table | `outputs/test_metrics.json`, `classification_report.pdf` |
| Confusion matrix | `outputs/confusion_matrix.png` / `_normalized.png` |
| ROC curves | `outputs/roc_curve.png` |
| PR curves | `outputs/precision_recall.png` |
| Calibration | `outputs/calibration_curve.png` |
| Training dynamics | `outputs/training_curves.png`, `outputs/metrics_history.csv` |
| Qualitative explainability | `outputs/gradcam_images/*.png` |

---

## 8. Explainability (`explainability/`)

Four attribution methods over the multi-view input, each producing per-view
heatmaps overlaid on the preprocessed image:
- **Grad-CAM** (Selvaraju et al., 2017)
- **Grad-CAM++** (Chattopadhyay et al., 2018)
- **Score-CAM** (Wang et al., 2020, gradient-free)
- **Integrated Gradients** (Sundararajan et al., 2017)

Generated for a sample of test patients by `training/test.py` into
`outputs/gradcam_images/`.

---

## 9. Reproducibility

- **Software:** Python 3.11 target (tested on 3.10), PyTorch 2.x, torchvision,
  Albumentations 2.x, MONAI, pydicom, OpenCV, scikit-learn, matplotlib. Pinned in
  `requirements.txt`.
- **Seed:** 42 (NumPy / PyTorch / MONAI / DataLoader workers).
- **Determinism:** patient split, class weights and manifest are saved and
  reused.
- **Hardware (this run):** CPU-only, 8 cores, ~9 GB RAM (hence 224px, no SAM). The
  identical code runs the full 512px + SAM configuration on a CUDA GPU.
- **Config:** every hyper-parameter lives in `config.py`; the exact config is
  embedded inside `best_model.pth` and printed to `outputs/train_run.log`.
- **Commands:** `python prepare_metadata.py` → `python run_cpu_training.py`
  (or `python -m training.train` for full config) → `python -m training.test`.

---

## 10. Results tables to fill (from `test_metrics.json` after `test.py`)

**Overall:**
| Metric | Value |
|---|---|
| Accuracy | ⟨fill⟩ |
| Balanced accuracy | ⟨fill⟩ |
| Macro-F1 | ⟨fill⟩ |
| Macro ROC-AUC | ⟨fill⟩ |
| Cohen's κ | ⟨fill⟩ |

**Per class (Normal / Abnormal):**
| Class | Precision | Recall (Sens.) | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| Normal | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| Abnormal | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

---

## 11. Suggested ablations (strengthen the paper)

Run by toggling flags in `run_cpu_training.py` / `config.py`:
- Loss: `--loss weighted_ce` vs focal.
- `use_balanced_sampler` on/off; class weighting on/off.
- Fusion: single-view vs multi-view; attention fusion vs mean pooling.
- SE block on/off; SAM on/off; MixUp/CutMix on/off.
- Backbone: frozen vs progressive unfreezing.

Each writes its own `metrics_history.csv` / `test_metrics.json` for comparison.

---

## 12. Limitations

- Only **149 Cancer patients** (104 in train) — cancer recall is the hardest and
  most variable metric; report confidence intervals / repeated seeds if possible.
- CPU run uses 224px (vs 512px) and omits SAM — note this if reporting these
  numbers; the GPU config is the intended final setting.
- Patient labels are derived from the folder organisation; a few corrupt DICOMs
  are skipped.

---

## 13. Phase 2 (Reinforcement Learning) — extension

Phase 1's exported per-patient/per-view embeddings and prediction probabilities
become the **MDP state** for an RL agent (DQN / Double DQN / Dueling DQN) that
learns longitudinal screening-decision policies. Scaffolded in `../Phase2_RL/`;
implemented after Phase 1 approval.

---

## 14. Key references

- He et al., *Deep Residual Learning* (ResNet), CVPR 2016.
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.
- Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018.
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
- Foret et al., *Sharpness-Aware Minimization*, ICLR 2021.
- Zhang et al., *mixup*, ICLR 2018; Yun et al., *CutMix*, ICCV 2019.
- Selvaraju et al., *Grad-CAM*, ICCV 2017; Chattopadhyay et al., *Grad-CAM++*, WACV 2018.
- Wang et al., *Score-CAM*, CVPRW 2020; Sundararajan et al., *Integrated Gradients*, ICML 2017.
