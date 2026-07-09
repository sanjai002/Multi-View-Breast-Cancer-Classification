"""
Dual-view breast dataset: yields (CC, MLO, label) per breast.

Preprocessed single-channel images are replicated to 3 channels and normalized
with ImageNet stats. CC and MLO of the same breast share augmentation params.
DICOMs are read on demand (never the whole dataset in RAM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_image


class DualViewBreastDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, transforms, img_size: int | None = None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.pcfg = dict(cfg.preprocess)
        if img_size is not None:                    # progressive resizing override
            self.pcfg["img_size"] = int(img_size)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def set_img_size(self, size: int) -> None:
        self.pcfg["img_size"] = int(size)

    def _load(self, path: str) -> np.ndarray:
        img = preprocess_image(path, _DictCfg(self.pcfg))     # HxW float32 [0,1]
        return np.repeat(img[..., None], 3, axis=-1)          # HxWx3

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        cc = self._load(row["cc_path"])
        mlo = self._load(row["mlo_path"])
        out = self.transforms(image=cc, mlo=mlo)              # shared params
        return {
            "cc": out["image"].float(),
            "mlo": out["mlo"].float(),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "patient_id": str(row["patient_id"]),
        }


class _DictCfg(dict):
    """Minimal .get/attr shim so preprocessing can read a plain dict."""
    def get(self, k, d=None):
        return dict.get(self, k, d)
