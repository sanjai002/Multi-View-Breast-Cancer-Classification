"""DICOM file reading with medical imaging handling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


@dataclass
class DicomImage:
    """Decoded DICOM with metadata."""

    pixels: np.ndarray
    """Pixel array as numpy array."""

    modality: str
    """Imaging modality (e.g., 'MG' for mammography)."""

    laterality: str
    """L (left) or R (right)."""

    view_position: str
    """View position (e.g., 'MLO', 'CC')."""

    patient_id: str
    """Patient ID."""

    photometric_interpretation: str
    """Photometric interpretation (e.g., 'MONOCHROME1', 'MONOCHROME2')."""


class DicomReader:
    """Read and decode DICOM mammography images."""

    def __init__(self, apply_voi: bool = True, force: bool = True) -> None:
        """Initialize DICOM reader.

        Args:
            apply_voi: Apply VOI LUT to windowing.
            force: Force read even if DICOM is non-compliant.
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required. Install with: pip install pydicom")
        self.apply_voi = apply_voi
        self.force = force

    def read(self, path: str | Path) -> DicomImage:
        """Read DICOM file and return metadata.

        Args:
            path: Path to DICOM file.

        Returns:
            DicomImage with metadata.

        Raises:
            IOError: If file cannot be read.
        """
        try:
            ds = pydicom.dcmread(path, force=self.force)
        except Exception as e:
            raise IOError(f"Failed to read DICOM {path}: {e}") from e

        modality = str(getattr(ds, "Modality", ""))
        laterality = str(getattr(ds, "ImageLaterality", ""))
        view_position = str(getattr(ds, "ViewPosition", ""))
        patient_id = str(getattr(ds, "PatientID", ""))
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))

        return DicomImage(
            pixels=self._decode_pixels(ds),
            modality=modality,
            laterality=laterality,
            view_position=view_position,
            patient_id=patient_id,
            photometric_interpretation=photometric,
        )

    def read_pixels(self, path: str | Path) -> np.ndarray:
        """Read DICOM and return pixel array.

        Args:
            path: Path to DICOM file.

        Returns:
            Pixel array as float32 in [0, 1].
        """
        img = self.read(path)
        return img.pixels

    def _decode_pixels(self, ds: "pydicom.Dataset") -> np.ndarray:
        """Decode DICOM pixels to float32 array in [0, 1].

        Handles:
        - VOI LUT windowing
        - MONOCHROME1 inversion
        - Rescale slope-intercept
        - Multi-frame images

        Args:
            ds: DICOM dataset.

        Returns:
            Pixel array as float32 in [0, 1].
        """
        # Get pixel array
        if not hasattr(ds, "pixel_array"):
            raise ValueError(f"DICOM {ds.filename} has no pixel data")

        arr = ds.pixel_array.astype(np.float32)

        # Apply VOI LUT if available
        if self.apply_voi:
            try:
                arr = apply_voi_lut(arr, ds).astype(np.float32)
            except Exception:
                # Fallback: use rescale slope-intercept if available
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

        # Handle MONOCHROME1 (invert)
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr

        # Normalize to [0, 1]
        arr = np.clip(arr, 0, None)
        arr_max = arr.max()
        if arr_max > 0:
            arr = arr / arr_max
        else:
            arr = np.zeros_like(arr)

        return arr.astype(np.float32)
