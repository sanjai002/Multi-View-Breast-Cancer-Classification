"""Adapter: build the pipeline's metadata CSV from the raw NLBS export.

The raw ``NLBSD_Metadata.csv`` uses Windows-style ``File Path`` values and
encodes the class in the *top-level folder* (``normal`` / ``abnormal`` /
``False Positive``) rather than in a single label column. This script converts it
into the schema the rest of Phase 1 expects:

    Patient_ID, Age, Image_Laterality, View_Position, Cancer, False_Positive, Image_Path

Label mapping (from the folder):
    normal          -> Normal   (Cancer=0, False_Positive=0)
    abnormal        -> Cancer   (Cancer=1, False_Positive=0)
    False Positive  -> FalsePos (Cancer=0, False_Positive=1)

Run::

    python prepare_metadata.py \
        --source ../NLBSD_Metadata.csv --data-root .. --out data/metadata.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

FOLDER_TO_LABEL = {"normal": 0, "abnormal": 1, "false positive": 2}


def build(source: str, data_root: str, out: str, verify: bool = True) -> pd.DataFrame:
    data_root = os.path.abspath(data_root)
    df = pd.read_csv(source)

    rel = df["File Path"].astype(str).str.replace("\\", "/", regex=False)
    top = rel.str.split("/").str[0].str.strip().str.lower()
    patient = rel.str.split("/").str[1]

    label = top.map(FOLDER_TO_LABEL)
    unknown = label.isna()
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} rows have an unrecognised top-level folder: "
            f"{sorted(top[unknown].unique())}"
        )
    label = label.astype(int)

    out_df = pd.DataFrame({
        "Patient_ID": top + "_" + patient,          # unique across folders
        "Age": pd.to_numeric(df["Age"], errors="coerce"),
        "Image_Laterality": df["Image Laterality"].astype(str).str.strip().str.upper().str[0],
        "View_Position": df["View Position"].astype(str).str.strip().str.upper(),
        "Cancer": (label == 1).astype(int),
        "False_Positive": (label == 2).astype(int),
        "Image_Path": [os.path.join(data_root, p) for p in rel],
    })

    if verify:
        exists = out_df["Image_Path"].map(os.path.isfile)
        missing = int((~exists).sum())
        if missing:
            print(f"[warn] {missing}/{len(out_df)} referenced files not found; dropping them.")
        out_df = out_df[exists].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    out_df.to_csv(out, index=False)

    n_patients = out_df["Patient_ID"].nunique()
    print(f"Wrote {out}: {len(out_df)} images, {n_patients} patients")
    print("Per-image class counts (0=Normal,1=Cancer,2=FalsePositive):")
    cls = out_df["Cancer"].mul(1).add(out_df["False_Positive"].mul(2))
    print(cls.value_counts().sort_index().to_dict())
    per_patient = out_df.groupby("Patient_ID").apply(
        lambda g: 1 if g["Cancer"].max() else (2 if g["False_Positive"].max() else 0),
        include_groups=False,
    )
    print("Per-patient class counts:", per_patient.value_counts().sort_index().to_dict())
    return out_df


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Build pipeline metadata from NLBS export")
    p.add_argument("--source", default=os.path.join(here, "..", "NLBSD_Metadata.csv"))
    p.add_argument("--data-root", default=os.path.join(here, ".."))
    p.add_argument("--out", default=os.path.join(here, "data", "metadata.csv"))
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    build(args.source, args.data_root, args.out, verify=not args.no_verify)


if __name__ == "__main__":
    main()
