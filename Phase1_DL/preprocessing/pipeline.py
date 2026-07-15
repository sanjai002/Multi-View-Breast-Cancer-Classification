"""Preprocessing pipeline orchestration."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from config.base import PreprocessingConfig
from preprocessing.dicom_reader import DicomReader
from preprocessing.breast_segmentation import segment_breast
from preprocessing.orientation import normalize_orientation
from preprocessing.normalization import normalize_intensity


class PreprocessingPipeline:
    """Orchestrate DICOM → preprocessed image pipeline."""

    def __init__(self, cfg: PreprocessingConfig) -> None:
        """Initialize pipeline.

        Args:
            cfg: Preprocessing configuration.
        """
        self.cfg = cfg
        self.reader = DicomReader(apply_voi=True, force=True)

    def process(self, dicom_path: str | Path, laterality: str = "L") -> np.ndarray:
        """Process DICOM file to preprocessed array.

        Args:
            dicom_path: Path to DICOM file.
            laterality: 'L' (left) or 'R' (right).

        Returns:
            Preprocessed image (H, W) at target size, normalized to [0, 1].

        Raises:
            IOError: If DICOM cannot be read or processing fails.
        """
        # 1. Read DICOM
        pixels = self.reader.read_pixels(dicom_path)

        # 2. Segment breast
        segmented, _ = segment_breast(
            pixels,
            method=self.cfg.breast_threshold_method,
            min_area_ratio=self.cfg.min_breast_area_ratio,
        )

        # 3. Normalize orientation
        oriented = normalize_orientation(segmented, laterality=laterality)

        # 4. Resize to target size
        resized = cv2.resize(
            oriented,
            (self.cfg.image_size, self.cfg.image_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

        # 5. Intensity normalization
        normalized = normalize_intensity(resized, method=self.cfg.normalize_method)

        return normalized.astype(np.float32)
