"""
dataset.py

Cache-only PyTorch dataset for the minimal NLBS four-view pipeline.

This dataset:
- NEVER reads DICOM files.
- Loads only preprocessed .npy cache files.
- Supports missing views with a binary mask.
- Returns patient-level samples with four standard views:
    LCC, LMLO, RCC, RMLO
- Fails immediately if any required cache file is missing.

Expected patient-manifest columns:
- Patient_ID
- Age
- label
- path_LCC
- path_LMLO
- path_RCC
- path_RMLO
- has_LCC
- has_LMLO
- has_RCC
- has_RMLO
- num_views
- split
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import config as C
except Exception:  # pragma: no cover
    C = None  # type: ignore


# ---------------------------------------------------------------------
# Defaults from config.py with safe fallbacks
# ---------------------------------------------------------------------
def _cfg(name: str, default):
    if C is None:
        return default
    return getattr(C, name, default)


DEFAULT_DATA_ROOT = Path(_cfg("DATA_ROOT", ".")).expanduser().resolve()
DEFAULT_CACHE_DIR = Path(_cfg("CACHE_DIR", "outputs/preproc_cache")).expanduser().resolve()
DEFAULT_IMAGE_SIZE = int(_cfg("IMAGE_SIZE", 224))
DEFAULT_VIEW_ORDER: Tuple[str, ...] = tuple(_cfg("VIEW_ORDER", ("LCC", "LMLO", "RCC", "RMLO")))
DEFAULT_MEAN = float((_cfg("MEAN", [0.5])[0]))
DEFAULT_STD = float((_cfg("STD", [0.5])[0]))


# ---------------------------------------------------------------------
# Cache key logic must match preprocess.py exactly
# ---------------------------------------------------------------------
def _resolve_path(path_str: str, data_root: Path) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (data_root / p).resolve()


def cache_key_for(image_path: str, data_root: Path, image_size: int) -> str:
    """
    Generate the portable cache key.

    The key depends on:
      1) path relative to data_root
      2) image_size

    This must be identical to preprocess.py.
    """
    abs_path = _resolve_path(image_path, data_root)
    try:
        rel = os.path.relpath(str(abs_path), str(data_root.resolve()))
    except Exception:
        rel = abs_path.name
    rel = rel.replace(os.sep, "/")
    return hashlib.md5(f"{rel}_{image_size}".encode()).hexdigest()


def cache_path_for(image_path: str, data_root: Path, cache_dir: Path, image_size: int) -> Path:
    return cache_dir / f"{cache_key_for(image_path, data_root, image_size)}.npy"


def _missing_view_image(image_size: int) -> np.ndarray:
    return np.zeros((image_size, image_size), dtype=np.float32)


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------
def load_patient_manifest(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"patient manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "Patient_ID",
        "Age",
        "label",
        "path_LCC",
        "path_LMLO",
        "path_RCC",
        "path_RMLO",
        "has_LCC",
        "has_LMLO",
        "has_RCC",
        "has_RMLO",
        "num_views",
        "split",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"patient manifest is missing required columns: {missing}")

    out = df.copy()
    out["Patient_ID"] = out["Patient_ID"].astype(str)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
    for v in DEFAULT_VIEW_ORDER:
        out[f"path_{v}"] = out[f"path_{v}"].astype(str)
        out[f"has_{v}"] = pd.to_numeric(out[f"has_{v}"], errors="coerce").fillna(0).astype(int)
    out["num_views"] = pd.to_numeric(out["num_views"], errors="coerce").fillna(0).astype(int)
    out["split"] = out["split"].astype(str)
    return out


def verify_cache_for_manifest(
    manifest: pd.DataFrame,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> List[str]:
    """
    Return a list of missing cache files.

    Missing views (has_* == 0 or empty path) are ignored.
    """
    missing: List[str] = []

    for row in manifest.itertuples(index=False):
        for view in DEFAULT_VIEW_ORDER:
            has_col = f"has_{view}"
            path_col = f"path_{view}"
            has_view = int(getattr(row, has_col, 0))
            path_str = str(getattr(row, path_col, ""))
            if has_view <= 0 or not path_str or path_str.lower() == "nan":
                continue

            cpath = cache_path_for(path_str, data_root, cache_dir, image_size)
            if not cpath.is_file():
                missing.append(f"{getattr(row, 'Patient_ID')} | {view} | {path_str} -> {cpath}")

    return missing


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class NLBSDataset(Dataset):
    """
    Patient-level cache-only dataset.

    Returns a dict with:
      - views: FloatTensor [4, 1, H, W]
      - mask:  FloatTensor [4]
      - label: LongTensor
      - patient_id: str
      - age: FloatTensor scalar
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        image_size: int = DEFAULT_IMAGE_SIZE,
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
        mean: float = DEFAULT_MEAN,
        std: float = DEFAULT_STD,
        strict_cache: bool = True,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.manifest = load_patient_manifest(manifest)
        else:
            self.manifest = manifest.reset_index(drop=True).copy()

        self.data_root = Path(data_root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.image_size = int(image_size)
        self.view_order = tuple(view_order)
        self.mean = float(mean)
        self.std = float(std)
        self.strict_cache = bool(strict_cache)
        self.transform = transform

        if len(self.view_order) != 4:
            raise ValueError(f"view_order must contain 4 views, got {self.view_order}")

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache directory not found: {self.cache_dir}")

        if self.strict_cache:
            missing = verify_cache_for_manifest(
                self.manifest,
                data_root=self.data_root,
                cache_dir=self.cache_dir,
                image_size=self.image_size,
            )
            if missing:
                preview = "\n".join(missing[:20])
                raise FileNotFoundError(
                    "Cache verification failed. Missing cache files:\n"
                    f"{preview}\n"
                    f"Total missing: {len(missing)}"
                )

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_cache(self, image_path: str) -> np.ndarray:
        cpath = cache_path_for(image_path, self.data_root, self.cache_dir, self.image_size)
        if not cpath.is_file():
            raise FileNotFoundError(f"Missing cache file: {cpath}")
        arr = np.load(cpath)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"Cache file must be 2D, got shape {arr.shape} from {cpath}")
        return arr

    def _to_tensor(self, img_u8: np.ndarray) -> torch.Tensor:
        """
        Convert a uint8 image to a normalized tensor [1, H, W].
        """
        img = img_u8.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(img)

        img = (img - self.mean) / (self.std + 1e-8)
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image after transform, got shape {img.shape}")
        return torch.from_numpy(img[None, ...].astype(np.float32))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.manifest.iloc[idx]
        patient_id = str(row["Patient_ID"])

        views: List[torch.Tensor] = []
        mask = torch.zeros(len(self.view_order), dtype=torch.float32)

        for i, view in enumerate(self.view_order):
            has_view = int(row[f"has_{view}"])
            path_str = str(row[f"path_{view}"])

            if has_view > 0 and path_str and path_str.lower() != "nan":
                img_u8 = self._load_cache(path_str)
                mask[i] = 1.0
            else:
                img_u8 = _missing_view_image(self.image_size)

            views.append(self._to_tensor(img_u8))

        views_tensor = torch.stack(views, dim=0)  # [4, 1, H, W]
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        age = torch.tensor(float(row["Age"]) if pd.notna(row["Age"]) else float("nan"), dtype=torch.float32)

        return {
            "views": views_tensor,
            "mask": mask,
            "label": label,
            "patient_id": patient_id,
            "age": age,
        }


# ---------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------
def make_dataloader(
    manifest: pd.DataFrame | str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    strict_cache: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    ds = NLBSDataset(
        manifest=manifest,
        data_root=data_root,
        cache_dir=cache_dir,
        image_size=image_size,
        strict_cache=strict_cache,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def split_manifest(manifest: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in manifest.columns:
        raise ValueError("manifest does not contain a split column")
    return manifest[manifest["split"].astype(str) == split_name].reset_index(drop=True)


def load_split_manifest(csv_path: str | Path, split_name: str) -> pd.DataFrame:
    manifest = load_patient_manifest(csv_path)
    return split_manifest(manifest, split_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick cache-only dataset check")
    parser.add_argument("--manifest", type=str, default="outputs/patient_manifest.csv")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = parser.parse_args()

    manifest = load_patient_manifest(args.manifest)
    missing = verify_cache_for_manifest(
        manifest,
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        image_size=args.image_size,
    )
    print(f"patients: {len(manifest)}")
    print(f"missing cache files: {len(missing)}")
    if missing:
        print("first missing examples:")
        for line in missing[:20]:
            print("  ", line)
        raise SystemExit(1)
    ds = NLBSDataset(
        manifest,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        strict_cache=True,
    )
    sample = ds[0]
    print("sample views:", sample["views"].shape)
    print("sample mask:", sample["mask"])
    print("sample label:", sample["label"].item())
    print("sample patient_id:", sample["patient_id"])