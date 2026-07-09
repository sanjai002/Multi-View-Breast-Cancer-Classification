"""DICOM reading for mammography.

Handles the pixel-value transforms that must happen *before* any spatial
preprocessing:

* Decompression of common transfer syntaxes (via pylibjpeg / GDCM handlers).
* Modality/Rescale + VOI LUT application (windowing) so pixel intensities match
  the intended presentation.
* MONOCHROME1 -> MONOCHROME2 correction (invert so higher value == brighter).
* Robust rescaling to a floating point image in [0, 1].

Images are read lazily, one file at a time, to keep memory usage flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


@dataclass
class DicomImage:
    """A decoded DICOM with the metadata preprocessing needs."""

    pixels: np.ndarray            # float32, shape (H, W), range [0, 1]
    laterality: str               # "L" | "R" | ""
    view_position: str            # "CC" | "MLO" | ""
    photometric: str              # original PhotometricInterpretation


class DicomReader:
    """Reads a single DICOM file into a normalised float image.

    Parameters
    ----------
    apply_voi:
        Apply the VOI LUT / windowing stored in the header.
    force:
        Pass ``force=True`` to :func:`pydicom.dcmread` for files with a missing
        preamble (common in de-identified research exports).
    """

    def __init__(self, apply_voi: bool = True, force: bool = True) -> None:
        self.apply_voi = apply_voi
        self.force = force

    # ------------------------------------------------------------------ #
    def read(self, path: str) -> DicomImage:
        ds = pydicom.dcmread(path, force=self.force)
        arr = self._decode_pixels(ds)
        arr = self._apply_voi_lut(ds, arr)
        arr = self._correct_monochrome(ds, arr)
        arr = self._to_unit_float(arr)

        laterality = str(
            getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")) or ""
        ).strip().upper()[:1]
        view = str(getattr(ds, "ViewPosition", "") or "").strip().upper()
        photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").strip()
        return DicomImage(
            pixels=arr.astype(np.float32),
            laterality=laterality,
            view_position=view,
            photometric=photometric,
        )

    def read_pixels(self, path: str) -> np.ndarray:
        """Convenience wrapper returning only the normalised pixel array."""
        return self.read(path).pixels

    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode_pixels(ds: pydicom.Dataset) -> np.ndarray:
        """Decode pixel data, applying Modality LUT / Rescale slope-intercept."""
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or intercept != 0.0:
            arr = arr * slope + intercept
        return arr

    def _apply_voi_lut(self, ds: pydicom.Dataset, arr: np.ndarray) -> np.ndarray:
        if not self.apply_voi:
            return arr
        try:
            # apply_voi_lut expects an integer array; round-trip through the
            # original dtype range so windowing/LUT lookups behave.
            work = arr
            if not np.issubdtype(work.dtype, np.integer):
                work = np.rint(work).astype(np.int32)
            out = apply_voi_lut(work, ds)
            return out.astype(np.float32)
        except Exception:
            # Header may lack VOI LUT / WindowCenter; fall back to raw values.
            return arr

    @staticmethod
    def _correct_monochrome(ds: pydicom.Dataset, arr: np.ndarray) -> np.ndarray:
        """Invert MONOCHROME1 so that higher pixel value == brighter tissue."""
        photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
        return arr

    @staticmethod
    def _to_unit_float(arr: np.ndarray) -> np.ndarray:
        """Robustly rescale to [0, 1] using percentile clipping."""
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo, hi = np.percentile(finite, (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)
