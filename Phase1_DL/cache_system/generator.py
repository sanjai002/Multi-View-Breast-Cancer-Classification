"""Cache file generation from DICOM metadata."""

import hashlib
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.base import PreprocessingConfig
from preprocessing.pipeline import PreprocessingPipeline
from utils.logging import get_logger


class CacheGenerator:
    """Generate preprocessed cache files from metadata."""

    def __init__(self, cfg: PreprocessingConfig) -> None:
        """Initialize cache generator.

        Args:
            cfg: Preprocessing configuration.
        """
        self.cfg = cfg
        self.pipeline = PreprocessingPipeline(cfg)
        self.logger = get_logger("cache_generator")

    def generate_cache_key(self, image_path: str | Path, image_size: int) -> str:
        """Generate deterministic cache key for image.

        Args:
            image_path: Path to DICOM file.
            image_size: Target image size.

        Returns:
            Hex string cache key.
        """
        # Normalize path and make it portable across machines by using an
        # origin root if provided. Priority:
        # 1. NLBS_ORIGINAL_DATA_ROOT env var
        # 2. Phase1_DL/outputs/cache_origin.txt (saved at cache creation)
        # 3. fallback to the provided path string

        str_path = str(image_path).replace("\\", "/")

        origin = os.environ.get("NLBS_ORIGINAL_DATA_ROOT")
        if origin is None:
            # try cache_origin file
            try:
                origin_file = Path(__file__).parent.parent / "outputs" / "cache_origin.txt"
                if origin_file.exists():
                    origin = origin_file.read_text().strip()
            except Exception:
                origin = None

        if origin:
            origin = origin.rstrip("/\\")
            if str_path.startswith(origin):
                rel = str_path[len(origin) :].lstrip("/\\")
            else:
                rel = str_path
        else:
            rel = str_path

        key_str = f"{rel}_{image_size}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_cache_path(self, image_path: str | Path, image_size: int) -> Path:
        """Get full cache file path.

        Args:
            image_path: Path to DICOM file.
            image_size: Target image size.

        Returns:
            Path to .npy cache file.
        """
        key = self.generate_cache_key(image_path, image_size)
        return self.cfg.cache_dir / f"{key}.npy"

    def process_and_cache(
        self,
        image_path: str | Path,
        laterality: str,
        image_size: int,
    ) -> Optional[Path]:
        """Process DICOM and save to cache.

        Args:
            image_path: Path to DICOM file.
            laterality: 'L' or 'R'.
            image_size: Target image size.

        Returns:
            Path to saved cache file, or None if processing failed.
        """
        cache_path = self.get_cache_path(image_path, image_size)

        # Skip if already cached
        if cache_path.exists():
            return cache_path

        try:
            # Process DICOM
            processed = self.pipeline.process(image_path, laterality)

            # Verify size matches
            if processed.shape != (image_size, image_size):
                self.logger.warning(
                    f"Size mismatch for {image_path}: "
                    f"got {processed.shape}, expected {image_size}x{image_size}"
                )
                return None

            # Save to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, processed)

            return cache_path

        except Exception as e:
            self.logger.error(f"Failed to cache {image_path}: {e}")
            return None

    def generate_from_metadata(
        self,
        metadata_csv: Path,
        image_size: int,
    ) -> tuple[int, int]:
        """Generate cache from metadata CSV.

        Args:
            metadata_csv: Path to metadata CSV with Image_Path, Image_Laterality columns.
            image_size: Target image size.

        Returns:
            Tuple of (successful_count, failed_count).
        """
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        self.logger.info(f"Loading metadata from {metadata_csv}")
        df = pd.read_csv(metadata_csv)

        if "Image_Path" not in df.columns or "Image_Laterality" not in df.columns:
            raise ValueError("Metadata CSV missing Image_Path or Image_Laterality columns")

        self.logger.info(f"Processing {len(df)} images")

        successful = 0
        failed = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating cache"):
            image_path = row["Image_Path"]
            laterality = str(row["Image_Laterality"])

            if not Path(image_path).exists():
                self.logger.warning(f"DICOM file not found: {image_path}")
                failed += 1
                continue

            result = self.process_and_cache(image_path, laterality, image_size)
            if result:
                successful += 1
            else:
                failed += 1

        self.logger.info(f"Cache generation complete: {successful} successful, {failed} failed")
        return successful, failed
