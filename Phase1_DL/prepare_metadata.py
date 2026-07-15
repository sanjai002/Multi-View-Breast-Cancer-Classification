"""
prepare_metadata.py

Build patient-level metadata and manifest files for the minimal NLBS
four-view classification pipeline.

Input:
    - Raw NLBS metadata CSV (preferred)
      Typical columns:
        * File Path
        * Image Laterality
        * View Position
        * Age
        * Cancer (or equivalent)
        * False_Positive (or equivalent)
    - Or an already-cleaned metadata CSV with canonical columns:
        * Patient_ID
        * Age
        * Image_Laterality
        * View_Position
        * Cancer
        * False_Positive
        * Image_Path

Output:
    - metadata.csv
      One row per image, canonical columns.
    - patient_manifest.csv
      One row per patient, with one column per standard view and a split.

The script excludes false-positive cases entirely.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


VIEW_ORDER: Tuple[str, ...] = ("LCC", "LMLO", "RCC", "RMLO")
CLASS_NAMES: Tuple[str, ...] = ("Normal", "Abnormal")
FOLDER_TO_LABEL = {"normal": 0, "abnormal": 1}
FALSE_POSITIVE_FOLDER = "false positive"

CANONICAL_COLUMNS = [
    "Patient_ID",
    "Age",
    "Image_Laterality",
    "View_Position",
    "Cancer",
    "False_Positive",
    "Image_Path",
]


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def _clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _norm_laterality(x) -> str:
    s = _clean_text(x).upper()
    if not s:
        return ""
    return s[0]


def _norm_view(x) -> str:
    s = _clean_text(x).upper().replace("-", "").replace("_", "")
    if not s:
        return ""
    if s in {"CC", "CRANIOCAUDAL", "CRANIOCAUDALVIEW"}:
        return "CC"
    if s in {"MLO", "MEDIOLATERALOBLIQUE", "MEDIOLATERALOBLIQUEVIEW"}:
        return "MLO"
    return s


def _norm_bool_int(x) -> int:
    if pd.isna(x):
        return 0
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    try:
        return int(float(x) != 0.0)
    except Exception:
        s = str(x).strip().lower()
        return int(s in {"1", "true", "yes", "y"})


def _detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = name.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def _is_raw_schema(df: pd.DataFrame) -> bool:
    return _detect_column(df, ["File Path"]) is not None


def _standardise_raw_metadata(source_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """
    Convert the raw NLBS export into the canonical image-level schema.
    False-positive cases are retained here only long enough to be filtered out
    in the next step.
    """
    file_path_col = _detect_column(source_df, ["File Path", "Image_Path", "Path", "FilePath"])
    age_col = _detect_column(source_df, ["Age", "Patient Age", "PatientAge"])
    lat_col = _detect_column(source_df, ["Image Laterality", "Image_Laterality", "Laterality", "Side"])
    view_col = _detect_column(source_df, ["View Position", "View_Position", "View", "ViewPosition"])
    cancer_col = _detect_column(source_df, ["Cancer", "Malignant"])
    fp_col = _detect_column(source_df, ["False_Positive", "False Positive", "FalsePositive", "FP"])

    if file_path_col is None:
        raise ValueError("Raw metadata must contain a 'File Path' column.")
    if age_col is None:
        raise ValueError("Raw metadata must contain an age column.")
    if lat_col is None:
        raise ValueError("Raw metadata must contain an image laterality column.")
    if view_col is None:
        raise ValueError("Raw metadata must contain a view position column.")

    df = source_df.copy()
    rel = df[file_path_col].astype(str).str.replace("\\", "/", regex=False)

    # Canonical top-level folder is the first component.
    top_folder = rel.str.split("/").str[0].str.strip().str.lower()
    patient_part = rel.str.split("/").str[1].astype(str)

    if cancer_col is not None:
        cancer = df[cancer_col].map(_norm_bool_int).astype(int)
    else:
        cancer = top_folder.map(FOLDER_TO_LABEL).fillna(0).astype(int)

    if fp_col is not None:
        false_positive = df[fp_col].map(_norm_bool_int).astype(int)
    else:
        false_positive = (top_folder == FALSE_POSITIVE_FOLDER).astype(int)

    out = pd.DataFrame(
        {
            "Patient_ID": top_folder.astype(str) + "_" + patient_part.astype(str),
            "Age": pd.to_numeric(df[age_col], errors="coerce"),
            "Image_Laterality": df[lat_col].map(_norm_laterality),
            "View_Position": df[view_col].map(_norm_view),
            "Cancer": cancer,
            "False_Positive": false_positive,
            "Image_Path": [str((data_root / Path(p)).resolve()) for p in rel],
        }
    )

    return out


def _standardise_canonical_metadata(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise an already-cleaned metadata CSV into the canonical schema.
    """
    df = source_df.copy()
    rename_map = {}
    for c in df.columns:
        k = str(c).strip().lower()
        if k in {"patient id", "patient_id", "patientid"}:
            rename_map[c] = "Patient_ID"
        elif k in {"age", "patient age"}:
            rename_map[c] = "Age"
        elif k in {"image laterality", "image_laterality", "laterality"}:
            rename_map[c] = "Image_Laterality"
        elif k in {"view position", "view_position", "view"}:
            rename_map[c] = "View_Position"
        elif k in {"cancer", "malignant"}:
            rename_map[c] = "Cancer"
        elif k in {"false positive", "false_positive", "falsepositive", "fp"}:
            rename_map[c] = "False_Positive"
        elif k in {"image path", "image_path", "path", "file path", "filename", "file"}:
            rename_map[c] = "Image_Path"

    df = df.rename(columns=rename_map).copy()

    required = {"Patient_ID", "Age", "Image_Laterality", "View_Position", "Cancer", "Image_Path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Canonical metadata CSV is missing required columns: {missing}")

    if "False_Positive" not in df.columns:
        df["False_Positive"] = 0

    out = pd.DataFrame(
        {
            "Patient_ID": df["Patient_ID"].astype(str),
            "Age": pd.to_numeric(df["Age"], errors="coerce"),
            "Image_Laterality": df["Image_Laterality"].map(_norm_laterality),
            "View_Position": df["View_Position"].map(_norm_view),
            "Cancer": df["Cancer"].map(_norm_bool_int).astype(int),
            "False_Positive": df["False_Positive"].map(_norm_bool_int).astype(int),
            "Image_Path": df["Image_Path"].astype(str),
        }
    )
    return out


def load_and_standardise(source: Path, data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    if _is_raw_schema(df):
        return _standardise_raw_metadata(df, data_root=data_root)
    return _standardise_canonical_metadata(df)


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only normal and abnormal rows and drop any false-positive rows.
    """
    out = df.copy()

    # Drop explicit false-positive rows first.
    if "False_Positive" in out.columns:
        out = out[out["False_Positive"].fillna(0).astype(int) == 0].copy()

    # Keep only the two class labels.
    out = out[out["Cancer"].isin([0, 1])].copy()

    # Normalize missing values.
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Image_Laterality"] = out["Image_Laterality"].map(_norm_laterality)
    out["View_Position"] = out["View_Position"].map(_norm_view)
    out["Patient_ID"] = out["Patient_ID"].astype(str)
    out["Image_Path"] = out["Image_Path"].astype(str)

    # Remove empty/invalid rows.
    required_nonempty = (
        out["Patient_ID"].str.len() > 0
    ) & (out["Image_Path"].str.len() > 0) & (out["Image_Laterality"].isin(["L", "R"])) & (
        out["View_Position"].isin(["CC", "MLO"])
    )
    out = out[required_nonempty].copy()

    return out.reset_index(drop=True)


def build_patient_manifest(
    image_df: pd.DataFrame,
    seed: int = 42,
    split_ratios: SplitRatios = SplitRatios(),
    balance_patients: bool = True,
) -> pd.DataFrame:
    """
    Collapse image-level metadata into one row per patient.

    Patient label rule:
        Abnormal if ANY image for that patient is abnormal.
        Normal otherwise.
    """
    split_ratios.validate()
    df = image_df.copy()

    view_key = df["Image_Laterality"].astype(str) + df["View_Position"].astype(str)
    df = df.assign(_view_key=view_key)

    records: List[Dict[str, object]] = []
    for patient_id, group in df.groupby("Patient_ID", sort=False):
        rec: Dict[str, object] = {
            "Patient_ID": str(patient_id),
            "Age": float(pd.to_numeric(group["Age"], errors="coerce").mean()),
        }

        label = 1 if int(group["Cancer"].max()) == 1 else 0
        rec["label"] = label

        for view in VIEW_ORDER:
            matches = group.loc[group["_view_key"] == view, "Image_Path"].tolist()
            rec[f"path_{view}"] = matches[0] if len(matches) > 0 else np.nan
            rec[f"has_{view}"] = 1 if len(matches) > 0 else 0

        rec["num_views"] = int(sum(rec[f"has_{v}"] for v in VIEW_ORDER))
        records.append(rec)

    table = pd.DataFrame.from_records(records)

    if balance_patients:
        # Simple undersampling to keep the dataset balanced at patient level.
        counts = table["label"].value_counts()
        if len(counts) == 2 and counts.min() > 0:
            n = int(counts.min())
            table = (
                pd.concat(
                    [
                        table[table["label"] == cls].sample(n=n, random_state=seed)
                        for cls in sorted(table["label"].unique())
                    ],
                    axis=0,
                )
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )

    # Deterministic split by patient.
    rng = np.random.RandomState(seed)
    indices = np.arange(len(table))
    rng.shuffle(indices)
    table = table.iloc[indices].reset_index(drop=True)

    n = len(table)
    n_train = int(round(n * split_ratios.train))
    n_val = int(round(n * split_ratios.val))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    n_test = max(0, n - n_train - n_val)

    split = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    if len(split) < n:
        split.extend(["test"] * (n - len(split)))
    table["split"] = split[:n]

    # Keep only patients with at least one view.
    table = table[table["num_views"] > 0].reset_index(drop=True)

    # Reorder columns.
    cols = ["Patient_ID", "Age", "label"] + [f"path_{v}" for v in VIEW_ORDER] + [
        f"has_{v}" for v in VIEW_ORDER
    ] + ["num_views", "split"]
    table = table[cols]

    return table


def save_outputs(
    image_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_df.to_csv(out_dir / "metadata.csv", index=False)
    patient_df.to_csv(out_dir / "patient_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NLBS metadata and patient manifest")
    p.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to raw or cleaned metadata CSV.",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing the NLBS image folders.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Output directory for metadata.csv and patient_manifest.csv.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the patient-level split.",
    )
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable patient-level class balancing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source CSV not found: {source}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    image_df = load_and_standardise(source, data_root=data_root)
    image_df = filter_rows(image_df)

    patient_df = build_patient_manifest(
        image_df,
        seed=args.seed,
        split_ratios=SplitRatios(),
        balance_patients=not args.no_balance,
    )

    save_outputs(image_df, patient_df, out_dir)

    print(f"Wrote: {out_dir / 'metadata.csv'}")
    print(f"Wrote: {out_dir / 'patient_manifest.csv'}")
    print(f"Images: {len(image_df)}")
    print(f"Patients: {len(patient_df)}")
    print("Class counts:", patient_df["label"].value_counts().sort_index().to_dict())
    print("Split counts:", patient_df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()