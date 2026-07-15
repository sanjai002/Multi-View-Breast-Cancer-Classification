"""Cache validation and integrity checking."""

import hashlib
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from config.base import PreprocessingConfig
from utils.logging import get_logger


class CacheValidator:
    """Validate cache integrity and completeness."""

    def __init__(self, cfg: PreprocessingConfig) -> None:
        """Initialize cache validator.

        Args:
            cfg: Preprocessing configuration.
        """
        self.cfg = cfg
        self.logger = get_logger("cache_validator")

    def validate_cache_exists(self, cache_path: Path) -> bool:
        """Check if cache file exists.

        Args:
            cache_path: Path to cache file.

        Returns:
            True if file exists and is readable.
        """
        if not cache_path.exists():
            return False

        try:
            _ = np.load(cache_path, mmap_mode="r")
            return True
        except Exception as e:
            self.logger.error(f"Cache file corrupted or unreadable: {cache_path}: {e}")
            return False

    def validate_cache_shape(self, cache_path: Path, expected_shape: Tuple[int, int]) -> bool:
        """Check cache file has expected shape.

        Args:
            cache_path: Path to cache file.
            expected_shape: Expected shape (H, W).

        Returns:
            True if shape matches.
        """
        try:
            arr = np.load(cache_path, mmap_mode="r")
            if arr.shape != expected_shape:
                self.logger.error(
                    f"Shape mismatch for {cache_path}: "
                    f"expected {expected_shape}, got {arr.shape}"
                )
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to read cache shape {cache_path}: {e}")
            return False

    def validate_all_cached(
        self,
        metadata_csv: Path,
        cache_dir: Path,
        cache_key_func,
        image_size: int,
    ) -> Tuple[bool, int, int]:
        """Validate all metadata entries have corresponding cache files.

        Args:
            metadata_csv: Path to metadata CSV.
            cache_dir: Cache directory.
            cache_key_func: Function(image_path, image_size) -> key string.
            image_size: Target image size.

        Returns:
            Tuple of (all_cached_bool, found_count, missing_count).
        """
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        df = pd.read_csv(metadata_csv)

        if "Image_Path" not in df.columns:
            raise ValueError("Metadata CSV missing Image_Path column")

        self.logger.info(f"Validating cache for {len(df)} entries")

        found = 0
        missing = 0
        missing_paths = []

        for image_path in df["Image_Path"]:
            key = cache_key_func(image_path, image_size)
            cache_path = cache_dir / f"{key}.npy"

            if self.validate_cache_exists(cache_path):
                found += 1
            else:
                missing += 1
                missing_paths.append((image_path, cache_path))

        if missing > 0:
            self.logger.error(f"Cache validation FAILED: {missing} missing files")
            for img_path, cache_path in missing_paths[:10]:  # Show first 10
                self.logger.error(f"  Missing: {cache_path} (from {img_path})")
            if len(missing_paths) > 10:
                self.logger.error(f"  ... and {len(missing_paths) - 10} more missing")
            return False, found, missing
        else:
            self.logger.info(f"Cache validation SUCCESS: all {found} entries cached")
            return True, found, missing
