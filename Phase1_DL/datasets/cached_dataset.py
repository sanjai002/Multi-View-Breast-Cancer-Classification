from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from config.base import get_config, Config
from cache_system.generator import CacheGenerator


VIEW_ORDER = ["LCC", "LMLO", "RCC", "RMLO"]


class CachedMammoDataset(Dataset):
    """Patient-level dataset that loads preprocessed `.npy` cache only.

    Expects a patient manifest CSV with columns `Patient_ID`, `Cancer`, and
    `path_<VIEW>` for each view in VIEW_ORDER.
    """

    def __init__(self, manifest_csv: Path, cfg: Optional[Config] = None):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        self.cfg = cfg or get_config()
        self.df = __import__("pandas").read_csv(self.manifest_csv)
        self.cache_gen = CacheGenerator(self.cfg.preprocessing)
        self.image_size = self.cfg.preprocessing.image_size

    def __len__(self) -> int:
        return len(self.df)

    def _load_view(self, image_path: str) -> Optional[np.ndarray]:
        # Accept NaN or empty as missing
        if not image_path or str(image_path) == "nan":
            return None

        cache_path = self.cache_gen.get_cache_path(image_path, self.image_size)
        if not cache_path.exists():
            return None

        arr = np.load(cache_path)
        return arr

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Collect views in fixed order
        views = []
        mask = []
        for view in VIEW_ORDER:
            key = f"path_{view}"
            image_path = row.get(key, "")
            arr = self._load_view(image_path)
            if arr is None:
                # missing view -> zeros
                views.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))
                mask.append(0)
            else:
                views.append(arr.astype(np.float32))
                mask.append(1)

        # Stack to (views, 1, H, W) and convert to tensor
        imgs = np.stack(views, axis=0)  # (V, H, W)
        imgs = imgs[:, None, :, :]
        imgs_t = torch.from_numpy(imgs)
        mask_t = torch.tensor(mask, dtype=torch.float32)

        label = int(row.get("Cancer", 0))
        label_t = torch.tensor(label, dtype=torch.long)

        return {"images": imgs_t, "mask": mask_t, "label": label_t, "patient_id": row["Patient_ID"]}
"""PyTorch Dataset for cached mammography images."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config.base import Config, PreprocessingConfig
from cache_system.generator import CacheGenerator
from utils.logging import get_logger


class CachedMammographyDataset(Dataset):
    """Load preprocessed images from cache (NO DICOM READING).

    Returns:
        Dictionary with keys:
        - views: (4, H, W) float32 tensor (4 mammography views)
        - mask: (4,) float32 tensor (1 if view present, 0 if missing)
        - label: int64 tensor (0=Normal, 1=Abnormal)
        - patient_id: str
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        cfg: Config,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset.

        Args:
            manifest_df: Patient manifest DataFrame with split column.
            cfg: Configuration.
            split: Dataset split ('train', 'val', 'test').
            transform: Optional augmentation transform (training only).
        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.logger = get_logger("dataset")

        # Filter to requested split
        self.df = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.logger.info(f"Loaded {len(self.df)} patients for split '{split}'")

        if len(self.df) == 0:
            raise ValueError(f"No patients found for split '{split}'")

        # Initialize cache generator for key computation
        self.cache_gen = CacheGenerator(cfg.preprocessing)

        # Precompute cache paths to fail fast if cache is missing
        self._validate_cache()

    def _validate_cache(self) -> None:
        """Validate all required cache files exist.

        Raises:
            FileNotFoundError: If any cache file is missing.
        """
        self.logger.info("Validating cache completeness...")
        missing = []

        for idx, row in self.df.iterrows():
            for view in ["LCC", "LMLO", "RCC", "RMLO"]:
                path_col = f"path_{view}"
                n_col = f"n_{view}"

                # Skip if view is missing
                if pd.isna(row[path_col]) or row[n_col] == 0:
                    continue

                image_path = row[path_col]
                cache_path = self.cache_gen.get_cache_path(
                    image_path, self.cfg.preprocessing.image_size
                )

                if not cache_path.exists():
                    missing.append((idx, view, image_path, cache_path))

        if missing:
            self.logger.error(f"Cache validation FAILED: {len(missing)} missing files")
            for idx, view, img_path, cache_path in missing[:5]:
                self.logger.error(f"  Patient {idx} {view}: {cache_path}")
            raise FileNotFoundError(
                f"Cache missing for {len(missing)} view(s). "
                f"Run: python scripts/generate_cache.py"
            )

        self.logger.info("✓ Cache validation complete: all files present")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """Load cached images for a patient.

        Args:
            idx: Patient index.

        Returns:
            Dictionary with:
            - views: (4, H, W) float32
            - mask: (4,) float32 (1=present, 0=missing)
            - label: int64
            - patient_id: str
        """
        row = self.df.iloc[idx]
        patient_id = str(row["Patient_ID"])

        views = []
        mask = []

        # Load each view in order: LCC, LMLO, RCC, RMLO
        for view in ["LCC", "LMLO", "RCC", "RMLO"]:
            path_col = f"path_{view}"
            n_col = f"n_{view}"

            if pd.isna(row[path_col]) or row[n_col] == 0:
                # Missing view: add zeros
                views.append(np.zeros((self.cfg.preprocessing.image_size,
                                      self.cfg.preprocessing.image_size), dtype=np.float32))
                mask.append(0.0)
            else:
                # Load from cache
                image_path = row[path_col]
                cache_path = self.cache_gen.get_cache_path(
                    image_path, self.cfg.preprocessing.image_size
                )

                try:
                    img = np.load(cache_path, allow_pickle=False).astype(np.float32)
                    views.append(img)
                    mask.append(1.0)
                except Exception as e:
                    self.logger.error(f"Failed to load cache {cache_path}: {e}")
                    raise

        # Stack views: (4, H, W)
        views_array = np.stack(views, axis=0)

        # Apply augmentation (training only)
        if self.transform is not None:
            views_array = self.transform(views_array)

        # Convert to tensors
        views_tensor = torch.from_numpy(views_array).float()  # (4, H, W)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)  # (4,)
        label_tensor = torch.tensor(int(row["Cancer"]), dtype=torch.int64)

        return {
            "views": views_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "patient_id": patient_id,
        }
