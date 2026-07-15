#!/usr/bin/env python3
"""Generate cache for all abnormal patients and an equal number of normal patients."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure the Phase1_DL package can be imported from the repository root.
ROOT = Path(__file__).resolve().parents[1]
PHASE1_DL = ROOT / "Phase1_DL"
if str(PHASE1_DL) not in sys.path:
    sys.path.insert(0, str(PHASE1_DL))

import pandas as pd

from config.base import get_config
from cache_system.generator import CacheGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a balanced cache subset for abnormal and normal patients."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting normal patients.",
    )
    parser.add_argument(
        "--normal-sample",
        type=int,
        default=None,
        help="Number of normal patients to sample; defaults to the number of abnormal patients.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="balanced",
        help="Prefix for generated metadata and manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    metadata_csv = cfg.metadata_csv
    manifest_csv = cfg.patient_manifest_csv

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Patient manifest not found: {manifest_csv}. Build it first with metadata/patient_manifest.py."
        )

    df_manifest = pd.read_csv(manifest_csv)
    abnormal = df_manifest[df_manifest["Cancer"] == 1]["Patient_ID"].unique().tolist()
    normal = df_manifest[df_manifest["Cancer"] == 0]["Patient_ID"].unique().tolist()

    if not abnormal:
        raise ValueError("No abnormal patients found in the patient manifest.")

    num_abnormal = len(abnormal)
    num_normal = args.normal_sample or num_abnormal
    if len(normal) < num_normal:
        raise ValueError(
            f"Not enough normal patients available ({len(normal)}) to sample {num_normal}."
        )

    rng = random.Random(args.seed)
    normal_sample = sorted(rng.sample(normal, num_normal))
    abnormal_sorted = sorted(abnormal)

    balanced_ids = sorted(abnormal_sorted + normal_sample)
    print(f"Selected {len(abnormal_sorted)} abnormal patients and {len(normal_sample)} normal patients.")

    selected_manifest = df_manifest[df_manifest["Patient_ID"].isin(balanced_ids)].copy()
    selected_manifest_path = cfg.output_dir / f"patient_manifest_{args.output_prefix}.csv"
    selected_manifest.to_csv(selected_manifest_path, index=False)
    print(f"Saved balanced manifest to {selected_manifest_path}")

    metadata_df = pd.read_csv(metadata_csv)
    selected_metadata = metadata_df[metadata_df["Patient_ID"].isin(balanced_ids)].copy()
    selected_metadata_path = cfg.output_dir / f"metadata_{args.output_prefix}.csv"
    selected_metadata.to_csv(selected_metadata_path, index=False)
    print(f"Saved balanced metadata to {selected_metadata_path} ({len(selected_metadata)} image rows)")

    cache_gen = CacheGenerator(cfg.preprocessing)
    success, failed = cache_gen.generate_from_metadata(selected_metadata_path, cfg.preprocessing.image_size)
    print(f"Cache generation complete: {success} successful, {failed} failed")

    print("Balanced cache generation finished.")


if __name__ == "__main__":
    main()
