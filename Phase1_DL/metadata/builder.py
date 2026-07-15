"""Image-level metadata building from DICOM scanning."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

from utils.logging import get_logger


class MetadataBuilder:
    """Scan DICOM directory and build image-level metadata CSV."""

    def __init__(self) -> None:
        """Initialize builder."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required. Install: pip install pydicom")
        self.logger = get_logger("metadata_builder")

    def scan_directory(
        self,
        root_dir: Path,
        output_csv: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Scan directory recursively for DICOM files and extract metadata.

        Args:
            root_dir: Root directory to scan.
            output_csv: Optional path to save metadata CSV.

        Returns:
            DataFrame with columns: Patient_ID, Age, Cancer, Image_Laterality,
            View_Position, Image_Path.
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        self.logger.info(f"Scanning DICOM files in {root_dir}")

        rows: List[Dict[str, Any]] = []
        dcm_count = 0

        for dcm_path in sorted(root_dir.rglob("*.dcm")):
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
                dcm_count += 1

                row = {
                    "Patient_ID": str(getattr(ds, "PatientID", dcm_path.parent.name)),
                    "Age": self._parse_age(getattr(ds, "PatientAge", None)),
                    "Cancer": 0,  # Default: will be set by label assignment
                    "Image_Laterality": str(getattr(ds, "ImageLaterality", "")),
                    "View_Position": str(getattr(ds, "ViewPosition", "")),
                    "Image_Path": str(dcm_path.absolute()),
                }
                rows.append(row)

            except Exception as e:
                self.logger.warning(f"Failed to read {dcm_path}: {e}")
                continue

        self.logger.info(f"Found {dcm_count} DICOM files")

        if not rows:
            raise ValueError(f"No valid DICOM files found in {root_dir}")

        metadata = pd.DataFrame(rows)
        self.logger.info(f"Built metadata: {len(metadata)} images")

        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            metadata.to_csv(output_csv, index=False)
            self.logger.info(f"Saved metadata to {output_csv}")

        return metadata

    @staticmethod
    def _parse_age(age_str) -> float:
        """Parse DICOM Age String like '045Y' to float years."""
        if age_str is None:
            return float("nan")
        try:
            s = str(age_str).strip().upper()
            if s.endswith("Y"):
                return float(s[:-1])
            return float(s)
        except (ValueError, IndexError):
            return float("nan")
