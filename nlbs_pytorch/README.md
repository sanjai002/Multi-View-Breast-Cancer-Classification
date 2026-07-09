# NLBS Dual-View Breast-Cancer Classification (PyTorch 2.x)

Production pipeline for **binary Normal-vs-Cancer** classification on the
Newfoundland & Labrador Breast Screening (NLBS) mammography dataset, using
**dual-view (CC + MLO)** fusion, modern medical-imaging training practices, full
explainability, Optuna tuning, and a multi-backbone ensemble.

> **Honest performance note.** The stated targets (Acc > 0.80, F1 > 0.80,
> AUC > 0.90) are the *design goal*. NLBS is a hard, real screening dataset with
> only **~168 cancer breasts / 149 patients**; the reference literature reaches
> ~0.57–0.80 AUC on it. This pipeline implements every lever known to help, but
> those numbers are aspirational and require a **GPU** to train at full
> resolution. All evaluation is **patient-level-safe** (no leakage) — reported
> numbers are real, not inflated by image-level splits.

## Folder structure
```
nlbs_pytorch/
├── configs/config.yaml            # every hyperparameter / toggle
├── requirements.txt
├── src/
│   ├── config.py                  # YAML + CLI dotted overrides
│   ├── data/
│   │   ├── preprocessing.py       # DICOM, CLAHE, breast seg, border crop, flip-left, QC
│   │   ├── augmentation.py        # medically-appropriate Albumentations
│   │   ├── build_index.py         # breast-level (CC+MLO) index from folders/metadata
│   │   ├── splitting.py           # PATIENT-level split + balancing
│   │   ├── sampler.py             # WeightedRandomSampler / class weights
│   │   ├── dataset.py             # DualViewBreastDataset
│   │   └── loaders.py             # DataLoaders (+ progressive-resize factory)
│   ├── models/
│   │   ├── backbones.py           # timm factory (ResNet/DenseNet/EffNetV2/ConvNeXt)
│   │   └── fusion.py              # early/late/feature/dualbranch/attention fusion
│   ├── losses/losses.py           # CE / weighted CE / Focal / BCE
│   ├── engine/trainer.py          # AMP, EMA, SAM, schedulers, mixup/cutmix, early stop, TB
│   └── utils/                     # seed, ema, sam, metrics
└── scripts/
    ├── train.py  test.py  inference.py
    ├── gradcam.py                 # Grad-CAM / ++ / Score-CAM / Integrated Gradients
    ├── tune_optuna.py  ensemble.py
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.data.build_index configs/config.yaml     # build breast index (once)
python scripts/train.py --smoke                        # fast end-to-end sanity check
python scripts/train.py                                # full training
python scripts/test.py --checkpoint <run>/checkpoints/best.pt
python scripts/gradcam.py --checkpoint <run>/checkpoints/best.pt
python scripts/tune_optuna.py --trials 30
python scripts/ensemble.py --train
tensorboard --logdir <output_root>
```

---
# Design decisions (the "why")

## Labels & dual-view construction
The NLBS `abnormal/` folder marks the cancer breast with a `-c` side folder
(`left-c` / `right-c`) — verified **100 % Cancer==1** for `-c` vs **0 %** otherwise.
So the unit of analysis is a **breast** = `(patient, side)` with its **CC + MLO**
views and a binary label. False-Positive folder is dropped (binary task).

## Preprocessing (`preprocessing.py`)
DICOM → VOI-LUT → MONOCHROME1 fix → **breast segmentation** (Otsu + largest
connected component) → **remove black border** (crop to breast bbox) → optional
median **denoise** → **CLAHE** (local contrast, standard in mammography) →
background zeroed (artifact/label removal) → **flip-left** (put chest wall on a
canonical side so all breasts are oriented consistently) → resize → [0,1].
`quality_ok()` rejects blank/constant frames.

## Augmentation — what to use and what NOT to (mammography)
**Use:** small **rotation (±12°)**, mild **affine** (scale/translate),
**random-resized crop**, **contrast/brightness**, **gamma**, mild **Gaussian
noise**, mild **elastic**, **random erasing** (CoarseDropout). These preserve
diagnostic content. **MixUp** (batch-level) is a good regularizer on small sets;
**CutMix** is off by default (pasting a patch can move/erase a lesion — use with
care). **Do NOT** use: **vertical flip** (anatomically invalid), **horizontal
flip** *after laterality standardization* (moves the chest wall — off by default),
heavy **color jitter / hue** (images are grayscale), large **elastic**
(distorts lesion morphology). CC and MLO of a breast get the **same** random
params so the pair stays consistent.

## Dataset splitting — 70/15/15, patient-level
**Always patient-level** (never image-level): all of a patient's breasts/views
go to one split, or the model cheats by re-identifying patients → inflated AUC.
**70/15/15 > 80/10/10 here:** with ~150 cancer patients, 80/10/10 leaves only
~15 cancer patients in test — too few for a stable AUC / confidence interval.
70/15/15 gives ~22 cancer test patients, a better bias/variance trade-off on a
small dataset. Splitting is **stratified** on patient cancer status.

## Class imbalance — recommendation
Implemented: **WeightedRandomSampler**, **Weighted CE**, **Focal Loss**,
downsampling. **SMOTE is not used** — interpolating whole mammograms produces
unrealistic tissue (SMOTE suits tabular/feature vectors, not raw images).
**Recommended: WeightedRandomSampler (balanced mini-batches) + Focal Loss.** The
sampler fixes batch composition without discarding data; Focal Loss then focuses
learning on hard/positive cancer cases. Mild majority downsampling
(`neg_per_pos`) further stabilizes training.

## Model & view-fusion — recommendation: **Attention fusion**
Backbone: **ResNet50** default (strong, well-behaved for fine-tuning), swappable
to DenseNet121 / EfficientNetV2-S / ConvNeXt-Tiny via timm.
Fusion options and verdict:
- **Early** (channel-concat → 1 backbone): loses view-specific features, forces
  premature mixing.
- **Late** (avg of two heads' logits): simple, but a fixed 50/50 average ignores
  that one view may be more informative for a given breast.
- **Feature / Dual-branch** (concat features → head): keeps view features; solid.
- **Attention** ✅ **recommended**: two encoders → learned per-view attention
  weights → weighted feature → head. Keeps view-specific representations *and*
  learns which view to trust per case — mirrors how radiologists weight CC vs
  MLO. Best accuracy/robustness trade-off on paired-view mammography.

## Transfer learning
ImageNet-pretrained backbones. **Two-phase schedule:** (1) **freeze the backbone**
for `freeze_backbone_epochs` and train only the head (lets the randomly-init head
settle without wrecking pretrained features — the exact failure we observed
otherwise); (2) **unfreeze** and fine-tune with **discriminative LRs** — low LR on
the backbone (`base_lr`), higher on the head (`base_lr × head_lr_mult`). Warmup +
cosine/OneCycle handles the LR ramp. This is the single biggest driver of
convergence on small medical data.

## Training configuration — recommendations
- **Optimizer:** AdamW (default); SGD-nesterov or **SAM** (flat-minima, better
  generalization) available.
- **Scheduler:** **OneCycleLR** (default) or cosine + warmup.
- **Weight decay:** 0.05 (AdamW). **Batch size:** 8 × `grad_accum 2` = eff. 16
  (raise on GPU). **Epochs:** 40 with **early stopping** (patience 10 on val AUC).
- **Gradient clipping** 1.0, **AMP** (GPU), **EMA** (decay 0.999), **warmup** 3 ep,
  **label smoothing** 0.05, **checkpointing** (best val AUC), **TensorBoard**.

## Loss — recommendation: **Focal Loss**
- CE: baseline, majority-biased. Weighted CE: helps but noisier.
- **Focal (γ=2, α=0.75 on cancer):** down-weights easy negatives, focuses on hard
  positives → best for this imbalance. **Recommended.**
- BCEWithLogits: equivalent binary form (pos_weight) — provided.
- Dice: a segmentation loss, **not** appropriate for image classification (kept
  only for completeness in the comparison).

## Evaluation
Accuracy, Precision, Recall/**Sensitivity**, **Specificity**, F1, **ROC-AUC**,
Avg-Precision, **Confusion Matrix**, **ROC**, **PR curve**, **Calibration curve** —
reported both **breast-level** and **patient-level** (patient positive if any
breast is predicted cancer). **Test-Time Augmentation** (flips) at eval.

## Explainability
`gradcam.py` produces **Grad-CAM**, **Grad-CAM++**, **Score-CAM** (via
`pytorch-grad-cam`) and **Integrated Gradients** (via `captum`) on the CC branch,
saved as side-by-side overlays.

## Hyperparameter tuning
`tune_optuna.py` maximizes val AUC over LR, weight decay, batch size, dropout,
focal γ, image size, optimizer, scheduler (short budget per trial).

## Ensemble — recommendation: **Weighted soft voting**
Compare ResNet50 / DenseNet121 / EfficientNetV2-S / ConvNeXt-Tiny.
- **Soft voting:** unweighted mean — simple, ignores model quality.
- **Weighted soft voting** ✅: mean weighted by each model's **val AUC** — lets
  stronger models lead, needs no extra held-out data. **Best default here.**
- **Stacking:** LR meta-learner — can win with enough data, but risks overfitting
  the meta-learner on a small validation set.

## Training tricks included
Test-Time Augmentation · Cosine Annealing · OneCycleLR · **SAM** · **EMA** ·
Gradient Accumulation · **Progressive Resizing** (256→384→512) · Balanced
mini-batches · **AMP** · Label smoothing · MixUp/CutMix · discriminative LRs ·
layer-unfreezing schedule.

## Avoiding overfitting
Patient-level split, heavy-but-safe augmentation, Focal Loss + class balancing,
dropout + LayerNorm head, weight decay, EMA, early stopping on val AUC, SAM
(flat minima), and majority downsampling — all target the small-data regime.
