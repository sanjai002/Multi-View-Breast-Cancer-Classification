"""Build train/val/test DataLoaders from the breast-level index + splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import train_transforms, eval_transforms
from src.data.dataset import DualViewBreastDataset
from src.data.sampler import make_weighted_sampler, class_weights
from src.data.splitting import make_splits
from src.data.build_index import build_index


def _ensure_index(cfg) -> pd.DataFrame:
    idx_path = Path(cfg.paths.index_csv)
    if idx_path.exists():
        return pd.read_csv(idx_path)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx = build_index(cfg)
    idx.to_csv(idx_path, index=False)
    return idx


def build_loaders(cfg):
    idx = _ensure_index(cfg)
    df = make_splits(idx, cfg)
    df.to_csv(cfg.paths.splits_csv, index=False)
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)

    mt = int(cfg.data.get("max_train", 0))       # >0 caps dataset (fast smoke)
    me = int(cfg.data.get("max_eval", 0))
    if mt:
        tr = tr.groupby("label", group_keys=False).head(max(mt // 2, 1)).reset_index(drop=True)
    if me:
        va = va.groupby("label", group_keys=False).head(max(me // 2, 1)).reset_index(drop=True)
        te = te.groupby("label", group_keys=False).head(max(me // 2, 1)).reset_index(drop=True)
    print(f"[loaders] train={len(tr)} val={len(va)} test={len(te)} | "
          f"train cancer={int((tr.label==1).sum())} normal={int((tr.label==0).sum())}")

    nw = int(cfg.data.num_workers)
    final = int(cfg.preprocess.img_size)

    def train_loader_fn(size: int) -> DataLoader:
        ds = DualViewBreastDataset(tr, cfg, train_transforms(cfg.augment, size), size)
        sampler = (make_weighted_sampler(tr)
                   if cfg.train.sampler in ("weighted", "balanced") else None)
        return DataLoader(ds, batch_size=cfg.train.batch_size,
                          sampler=sampler, shuffle=sampler is None,
                          num_workers=nw, pin_memory=torch.cuda.is_available(),
                          drop_last=True)

    val_loader = DataLoader(
        DualViewBreastDataset(va, cfg, eval_transforms(final), final),
        batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw)
    test_loader = DataLoader(
        DualViewBreastDataset(te, cfg, eval_transforms(final), final),
        batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw)
    return train_loader_fn, val_loader, test_loader, df, class_weights(tr)
