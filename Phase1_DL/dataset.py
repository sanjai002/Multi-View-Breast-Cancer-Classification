"""Memory-efficient multi-view mammography dataset.

Each item corresponds to **one patient** and returns the four standard views
(LCC, LMLO, RCC, RMLO) stacked into a single tensor together with a validity
mask that flags missing views. DICOM files are decoded and preprocessed *on
demand* inside ``__getitem__`` so the entire dataset is never held in RAM.

An optional on-disk cache (``cfg.data.cache_preprocessed``) stores the resulting
float arrays as ``.npy`` to accelerate repeated epochs without inflating memory.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import Config
from dicom import DicomReader
from preprocess import MammoPreprocessor
from utils import get_logger, seed_worker


class MultiViewMammographyDataset(Dataset):
    """Patient-level dataset yielding four preprocessed, augmented views."""

    def __init__(self, table: pd.DataFrame, cfg: Config,
                 transform: Optional[Callable] = None, train: bool = False) -> None:
        """
        Parameters
        ----------
        table:
            Patient-level dataframe (one row per patient) with ``path_<VIEW>``
            columns, ``label`` and ``Patient_ID``.
        transform:
            An Albumentations ``Compose`` applied per view. Receives an
            ``HxWx1`` float image in [0, 1] and must return a ``(1, H, W)``
            tensor (i.e. it should end in ``Normalize`` + ``ToTensorV2``).
        train:
            Only affects logging / cache namespacing; augmentation is decided by
            the ``transform`` that is passed in.
        """
        self.table = table.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self.train = train
        self.view_order = list(cfg.data.view_order)

        self.reader = DicomReader(apply_voi=True)
        self.preprocessor = MammoPreprocessor(cfg)
        self._logger = get_logger()

        if cfg.data.cache_preprocessed:
            os.makedirs(cfg.data.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> Dict:
        row = self.table.iloc[idx]
        patient_id = str(row["Patient_ID"])

        view_tensors: List[torch.Tensor] = []
        mask = torch.zeros(len(self.view_order), dtype=torch.float32)

        for v_idx, view in enumerate(self.view_order):
            path = row.get(f"path_{view}", np.nan)
            laterality = view[0]  # 'L' or 'R'
            if isinstance(path, str) and os.path.isfile(path):
                img = self._load_and_preprocess(path, laterality, view, patient_id)
                mask[v_idx] = 1.0
            else:
                img = np.zeros(
                    (self.cfg.data.image_size, self.cfg.data.image_size), np.float32
                )
            view_tensors.append(self._to_tensor(img))

        views = torch.stack(view_tensors, dim=0)  # (V, C, H, W)
        label = int(row["label"])
        return {
            "views": views,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id,
            "age": float(row.get("Age", float("nan"))),
        }

    # ------------------------------------------------------------------ #
    def _load_and_preprocess(self, path: str, laterality: str, view: str,
                             patient_id: str) -> np.ndarray:
        """Return a preprocessed [0, 1] float image, using the cache if enabled."""
        if self.cfg.data.cache_preprocessed:
            cache_path = self._cache_path(path)
            if os.path.isfile(cache_path):
                try:
                    # Cache is stored as compact uint8; restore to float [0, 1].
                    return np.load(cache_path).astype(np.float32) / 255.0
                except Exception:
                    pass  # corrupt cache entry; recompute
        try:
            raw = self.reader.read_pixels(path)
            img = self.preprocessor(raw, laterality)
        except Exception as exc:  # never let one bad file kill an epoch
            self._logger.warning("Failed to read %s (%s): %s", view, path, exc)
            return np.zeros(
                (self.cfg.data.image_size, self.cfg.data.image_size), np.float32
            )

        if self.cfg.data.cache_preprocessed:
            try:
                np.save(self._cache_path(path),
                        (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8))
            except Exception:
                pass
        return img

    def _cache_path(self, path: str) -> str:
        # Key on the path *relative to data_root* (not the absolute path) so the
        # cache is portable across machines (e.g. local <-> Colab) where the
        # dataset is mounted at a different absolute location.
        try:
            rel = os.path.relpath(os.path.abspath(path), self.cfg.paths.data_root)
        except ValueError:
            rel = os.path.basename(path)
        rel = rel.replace(os.sep, "/")
        key = hashlib.md5(f"{rel}_{self.cfg.data.image_size}".encode()).hexdigest()
        return os.path.join(self.cfg.data.cache_dir, f"{key}.npy")

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Apply the transform (or a bare normalise) and guarantee (C, H, W)."""
        img_hwc = img[..., None].astype(np.float32)  # HxWx1
        if self.transform is not None:
            tensor = self.transform(image=img_hwc)["image"]
        else:
            mean = float(self.cfg.data.normalize_mean[0])
            std = float(self.cfg.data.normalize_std[0])
            tensor = torch.from_numpy(((img - mean) / (std + 1e-8))[None, ...])
        if tensor.dim() == 2:  # ToTensorV2 on 2-D input drops the channel axis
            tensor = tensor.unsqueeze(0)
        return tensor.float()


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def build_dataloaders(train_table: pd.DataFrame, val_table: pd.DataFrame,
                      test_table: pd.DataFrame, cfg: Config,
                      train_transform: Callable,
                      eval_transform: Callable) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders with reproducible workers."""
    generator = torch.Generator()
    generator.manual_seed(cfg.data.seed)

    datasets = {
        "train": MultiViewMammographyDataset(train_table, cfg, train_transform, train=True),
        "val": MultiViewMammographyDataset(val_table, cfg, eval_transform, train=False),
        "test": MultiViewMammographyDataset(test_table, cfg, eval_transform, train=False),
    }

    # Optional class-balanced oversampling for the training split.
    train_sampler = None
    if cfg.data.use_balanced_sampler:
        labels = train_table["label"].values.astype(int)
        counts = np.bincount(labels, minlength=cfg.data.num_classes).astype(np.float64)
        class_weight = 1.0 / np.clip(counts, 1.0, None)
        sample_weights = class_weight[labels]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(labels), replacement=True, generator=generator,
        )

    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=cfg.train.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
            drop_last=True, worker_init_fn=seed_worker, generator=generator,
            persistent_workers=cfg.data.num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.num_workers > 0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        ),
    }
    return loaders
