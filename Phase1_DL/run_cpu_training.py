"""Launch full-dataset Phase 1 training tuned for this CPU-only, low-RAM box.

Chosen for feasibility on 8 cores / ~9 GB RAM / limited disk with no GPU:
  * image_size 224 (ResNet's native size; 1/5 the pixels of 512)
  * batch_size 4, num_workers 2 (fits RAM)
  * preprocessing cache ON, stored as uint8 (~1 GB) so epochs after the first
    skip DICOM decoding entirely
  * SAM OFF (it doubles forward/backward cost; re-enable on a GPU)
  * Focal loss + inverse-frequency class weights for the strong imbalance
    (balanced Normal / Abnormal patient subsets)

Everything else (EMA, cosine schedule, progressive unfreezing, differential LRs,
early stopping, TensorBoard) is unchanged from the full method.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from config import get_config
from training.train import Trainer
from utils import count_parameters


def main() -> None:
    cfg = get_config()
    cfg.experiment_name = "nlbs_multiview_resnet50_cpu_full"

    # --- data ---
    cfg.paths.metadata_csv = os.path.join(_HERE, "data", "metadata.csv")
    cfg.data.image_size = 224
    # num_workers=1: ONE prefetch worker overlaps cache reads with compute (big
    # speedup vs 0) while using far less RAM than the 2 workers that caused the
    # earlier OOM. Safe with the current ~3.7 GB free.
    cfg.data.num_workers = 1
    cfg.data.pin_memory = False
    cfg.data.cache_preprocessed = True
    # Balanced sampler OFF: stacking it with class-weighted focal loss double-
    # compensates for imbalance and collapses the model to always predicting
    # one minority class
    # -> always Cancer, AUC=0.5, macroF1 near-zero). Class weighting alone in
    # the loss is enough; pick one compensation mechanism, not both.
    cfg.data.use_balanced_sampler = False

    # --- model / transfer learning ---
    cfg.model.pretrained = True
    cfg.model.freeze_backbone = True

    # --- optimisation (CPU-tractable) ---
    cfg.train.epochs = 40
    # batch 4 processes bigger matmuls (faster on CPU) at only ~+0.3 GB. Safe
    # because the OOM came from the *worker processes* (now num_workers=0), not
    # the batch size.
    cfg.train.batch_size = 4
    cfg.train.use_sam = False
    cfg.train.use_ema = True
    cfg.train.use_amp = False          # no benefit on CPU
    cfg.train.loss = "focal"
    cfg.train.class_weighting = True
    cfg.train.early_stopping_patience = 12
    cfg.train.save_every_epoch = True      # checkpoints/epoch_XXX.pth after each epoch
    cfg.train.resume = True                # auto-continue from checkpoints/last.pth

    cfg.validate()
    cfg.create_dirs()

    # Record our own PID so the monitor / stop commands target the real process.
    with open(os.path.join(cfg.paths.output_dir, "train.pid"), "w") as f:
        f.write(str(os.getpid()))

    print("=" * 70)
    print("NLBS Phase 1 - full-dataset CPU training")
    print(f"  metadata     : {cfg.paths.metadata_csv}")
    print(f"  image_size   : {cfg.data.image_size}   batch: {cfg.train.batch_size}"
          f"   workers: {cfg.data.num_workers}")
    print(f"  epochs       : {cfg.train.epochs}   SAM: {cfg.train.use_sam}"
          f"   EMA: {cfg.train.use_ema}   loss: {cfg.train.loss}")
    print(f"  balanced_samp: {cfg.data.use_balanced_sampler}   "
          f"save_every_epoch: {cfg.train.save_every_epoch}")
    print(f"  cache_preproc: {cfg.data.cache_preprocessed} -> {cfg.data.cache_dir}")
    print("=" * 70)

    trainer = Trainer(cfg)
    trainable, total = count_parameters(trainer.model)
    print(f"Model: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total params")
    trainer.fit()


if __name__ == "__main__":
    main()
