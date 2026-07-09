"""Launch Phase 1 training on a Colab GPU, resumable across disconnects and
interchangeable with the local CPU run (`run_cpu_training.py`).

Design (see ../Phase1_DL/DOCS.md and NEXT_STEPS.md for the full rationale):

* **Checkpoints / logs live on Google Drive** (`DRIVE_ROOT`), not on Colab's
  ephemeral disk. If the Colab runtime disconnects or the session is recycled,
  nothing is lost — rerunning this script (`cfg.train.resume = True`) picks up
  from `checkpoints/last.pth` on Drive automatically.
* **Same resolution (224px) and same split manifest as the local CPU run**, so
  a checkpoint born here can be pulled down and continued locally on CPU (or
  vice versa) — see `pull_from_colab.sh`. The model architecture itself is
  resolution-agnostic (global average pooling), so this is a compatibility
  choice for a clean handoff, not a hard requirement.
* **The preprocessing cache is reused as-is** (uploaded once via
  `prepare_colab_bundle.sh`), so Colab never needs the 325 GB of raw DICOMs —
  only the ~1 GB of already-preprocessed arrays.
* GPU headroom is spent on the *full* method: SAM, AMP, a bigger batch and more
  workers than the CPU run could afford.

Usage inside the Colab notebook (see NLBS_Colab_Training.ipynb):
    !python run_colab_training.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import get_config
from training.train import Trainer
from utils import count_parameters, get_device

# --- Colab paths (overridable via environment variables from the notebook) ---
# MUST equal the ORIGINAL local data_root string used when metadata.csv's
# Image_Path column and the preprocessing cache were built. It does not need to
# exist as a real directory here: it is only used to compute the *relative*
# path of each image for the cache key, so cache hits line up with the files
# uploaded from the local machine. See dataset.py's `_cache_path`.
ORIGINAL_LOCAL_DATA_ROOT = os.environ.get(
    "NLBS_ORIGINAL_DATA_ROOT",
    "/mnt/NewVolume/projects/breast cancer detection",
)
# Where the (small) unzipped code + cache + metadata bundle lives on the Colab
# VM's local ephemeral disk this session (fast reads).
LOCAL_DATA_DIR = os.environ.get("NLBS_LOCAL_DATA_DIR", "/content/nlbs_data")
# Persistent Google-Drive-backed folder for checkpoints/logs/outputs — survives
# disconnects and is what you pull back down to continue locally.
DRIVE_ROOT = os.environ.get(
    "NLBS_DRIVE_ROOT", "/content/drive/MyDrive/NLBS_Phase1"
)


def main() -> None:
    cfg = get_config()
    cfg.experiment_name = "nlbs_multiview_resnet50_colab_gpu"

    # --- data (Colab-local ephemeral disk for fast reads) ---
    cfg.paths.data_root = ORIGINAL_LOCAL_DATA_ROOT
    cfg.paths.metadata_csv = os.path.join(LOCAL_DATA_DIR, "metadata.csv")
    cfg.data.image_size = 224          # must match the uploaded cache + local run
    cfg.data.cache_preprocessed = True
    cfg.data.cache_dir = os.path.join(LOCAL_DATA_DIR, "preproc_cache")
    cfg.data.num_workers = 2
    cfg.data.pin_memory = True
    # Balanced sampler OFF: stacking it with class-weighted focal loss double-
    # compensates for imbalance and collapses the model to always predicting
    # one minority class (confirmed on both CPU and this GPU run: AUC=0.5,
    # macroF1 near-zero, 100% recall on a single class). Class weighting alone
    # in the loss is enough; pick one compensation mechanism, not both.
    cfg.data.use_balanced_sampler = False

    # --- persistent outputs (Google Drive) ---
    cfg.paths.manifest_csv = os.path.join(LOCAL_DATA_DIR, "patient_manifest.csv")
    cfg.paths.output_dir = os.path.join(DRIVE_ROOT, "outputs")
    cfg.paths.checkpoint_dir = os.path.join(DRIVE_ROOT, "checkpoints")
    cfg.paths.tensorboard_dir = os.path.join(DRIVE_ROOT, "tensorboard")
    cfg.paths.log_dir = os.path.join(DRIVE_ROOT, "logs")
    cfg.paths.gradcam_dir = os.path.join(DRIVE_ROOT, "outputs", "gradcam_images")

    # --- full-spec optimisation, now that a GPU is available ---
    cfg.model.pretrained = True
    cfg.model.freeze_backbone = True
    cfg.train.epochs = 40
    cfg.train.batch_size = 16
    cfg.train.use_sam = True
    cfg.train.use_ema = True
    cfg.train.use_amp = True
    cfg.train.amp_dtype = "bf16"       # auto-falls back to fp16 if unsupported
    cfg.train.loss = "focal"
    cfg.train.class_weighting = True
    cfg.train.early_stopping_patience = 12
    cfg.train.save_every_epoch = True
    cfg.train.resume = True            # auto-continue from checkpoints/last.pth

    cfg.validate()
    cfg.create_dirs()

    with open(os.path.join(cfg.paths.output_dir, "train.pid"), "w") as f:
        f.write(str(os.getpid()))

    device = get_device()
    print("=" * 70)
    print("NLBS Phase 1 - Colab GPU training")
    print(f"  device       : {device}"
          + (f" ({__import__('torch').cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  data (local) : {LOCAL_DATA_DIR}")
    print(f"  persistent   : {DRIVE_ROOT}  <- checkpoints/logs survive disconnects")
    print(f"  image_size   : {cfg.data.image_size}   batch: {cfg.train.batch_size}"
          f"   workers: {cfg.data.num_workers}")
    print(f"  epochs       : {cfg.train.epochs}   SAM: {cfg.train.use_sam}"
          f"   AMP: {cfg.train.use_amp}/{cfg.train.amp_dtype}   EMA: {cfg.train.use_ema}")
    print(f"  balanced_samp: {cfg.data.use_balanced_sampler}   "
          f"class_weighting: {cfg.train.class_weighting}   loss: {cfg.train.loss}")
    print(f"  resume       : {cfg.train.resume}  (from checkpoints/last.pth if present)")
    print("=" * 70)

    if device.type != "cuda":
        print("WARNING: no GPU detected — Runtime > Change runtime type > GPU in Colab.")

    trainer = Trainer(cfg)
    trainable, total = count_parameters(trainer.model)
    print(f"Model: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total params")
    trainer.fit()


if __name__ == "__main__":
    main()
