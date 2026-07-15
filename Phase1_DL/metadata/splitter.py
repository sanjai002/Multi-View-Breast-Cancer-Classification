"""Patient-level data splitting."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.base import Config
from utils.logging import get_logger


class PatientSplitter:
    """Perform patient-level stratified split."""

    def __init__(self, cfg: Config) -> None:
        """Initialize splitter.

        Args:
            cfg: Configuration.
        """
        self.cfg = cfg
        self.logger = get_logger("splitter")

    def split_manifest(
        self,
        manifest_df: pd.DataFrame,
        output_csv: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Add split column to manifest.

        Performs stratified patient-level split ensuring:
        - No patient leakage between splits
        - Class balance across splits
        - Reproducible with fixed seed

        Args:
            manifest_df: Patient-level manifest DataFrame.
            output_csv: Optional path to save split manifest.

        Returns:
            Manifest with 'split' column added (train/val/test).
        """
        self.logger.info(f"Splitting {len(manifest_df)} patients")

        labels = manifest_df["Cancer"].values
        seed = self.cfg.data.seed

        # Check if stratification is possible
        unique_labels, counts = np.unique(labels, return_counts=True)
        if (counts < 2).any():
            self.logger.warning("Some classes have < 2 samples; using non-stratified split")
            stratify = None
        else:
            stratify = labels

        # Train/temp split
        train_df, temp_df = train_test_split(
            manifest_df,
            test_size=self.cfg.data.val_ratio + self.cfg.data.test_ratio,
            random_state=seed,
            stratify=stratify,
        )

        # Val/test split
        val_test_ratio = self.cfg.data.test_ratio / (self.cfg.data.val_ratio + self.cfg.data.test_ratio)
        stratify_temp = temp_df["Cancer"].values if stratify is not None else None
        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_test_ratio,
            random_state=seed,
            stratify=stratify_temp,
        )

        # Add split column
        result = manifest_df.copy()
        result["split"] = "train"
        result.loc[result["Patient_ID"].isin(val_df["Patient_ID"]), "split"] = "val"
        result.loc[result["Patient_ID"].isin(test_df["Patient_ID"]), "split"] = "test"

        # Log split statistics
        for split in ["train", "val", "test"]:
            subset = result[result["split"] == split]
            counts = subset["Cancer"].value_counts().to_dict()
            self.logger.info(f"Split {split:5s}: {len(subset):4d} patients | {counts}")

        if output_csv:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_csv, index=False)
            self.logger.info(f"Saved split manifest to {output_csv}")

        return result
