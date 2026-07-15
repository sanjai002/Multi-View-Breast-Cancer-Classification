"""Patient-level manifest generation."""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from config.base import Config
from utils.logging import get_logger


class PatientManifestBuilder:
    """Build patient-level manifest from image-level metadata."""

    def __init__(self, cfg: Config) -> None:
        """Initialize manifest builder.

        Args:
            cfg: Configuration.
        """
        self.cfg = cfg
        self.logger = get_logger("manifest_builder")

    def build_manifest(
        self,
        metadata_csv: Path,
        output_csv: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Build patient-level manifest from image metadata.

        Args:
            metadata_csv: Path to image-level metadata CSV.
            output_csv: Optional path to save manifest CSV.

        Returns:
            Patient-level manifest DataFrame with columns:
            - Patient_ID
            - Age
            - Cancer (label)
            - path_LCC, path_LMLO, path_RCC, path_RMLO
            - n_LCC, n_LMLO, n_RCC, n_RMLO (view counts)
            - split (will be added by splitter)
        """
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

        self.logger.info(f"Loading image metadata from {metadata_csv}")
        df = pd.read_csv(metadata_csv)

        required_cols = ["Patient_ID", "Age", "Cancer", "Image_Laterality", "View_Position", "Image_Path"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns: {missing}")

        self.logger.info(f"Aggregating {len(df)} images to patient level")

        # Create view key: laterality + view position
        df["view_key"] = df["Image_Laterality"].astype(str) + df["View_Position"].astype(str)

        records = []
        for patient_id, group in df.groupby("Patient_ID"):
            rec = {
                "Patient_ID": str(patient_id),
                "Age": float(group["Age"].iloc[0]) if "Age" in group else np.nan,
                "Cancer": int(group["Cancer"].max()),  # Max: abnormal if any image is abnormal
            }

            # For each view, store path and count
            for view in ["LCC", "LMLO", "RCC", "RMLO"]:
                matches = group[group["view_key"] == view]["Image_Path"].tolist()
                rec[f"path_{view}"] = matches[0] if matches else np.nan
                rec[f"n_{view}"] = len(matches)

            records.append(rec)

        manifest = pd.DataFrame(records)
        self.logger.info(f"Built manifest: {len(manifest)} patients")

        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(output_csv, index=False)
            self.logger.info(f"Saved manifest to {output_csv}")

        return manifest
