"""Adapter: build the pipeline's metadata CSV from the raw NLBS export.

The raw ``NLBSD_Metadata.csv`` uses Windows-style ``File Path`` values and
encodes the class in the top-level folder. This script keeps only the
``normal`` and ``abnormal`` folders, drops screening-error folders, and converts
the remaining records into the schema the rest of Phase 1 expects:

    Patient_ID, Age, Image_Laterality, View_Position, Cancer, Image_Path

Label mapping (from the folder):
    normal          -> Normal   (Cancer=0)
    abnormal        -> Abnormal (Cancer=1)

Run::

    python prepare_metadata.py \
        --source ../NLBSD_Metadata.csv --data-root .. --out data/metadata.csv
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

FOLDER_TO_LABEL = {"normal": 0, "abnormal": 1}
FALSE_POSITIVE_FOLDER = "false positive"


def build(source: str, data_root: str, out: str, verify: bool = True) -> pd.DataFrame:
    data_root = os.path.abspath(data_root)
    df = pd.read_csv(source)

    rel = df["File Path"].astype(str).str.replace("\\", "/", regex=False)
    top = rel.str.split("/").str[0].str.strip().str.lower()
    patient = rel.str.split("/").str[1]

    known = top.isin(set(FOLDER_TO_LABEL) | {FALSE_POSITIVE_FOLDER})
    if not known.all():
        raise ValueError(
            f"{int((~known).sum())} rows have an unrecognised top-level folder: "
            f"{sorted(top[~known].unique())}"
        )
    keep = top.isin(FOLDER_TO_LABEL)
    dropped = int((top == FALSE_POSITIVE_FOLDER).sum())
    if dropped:
        print(
            f"[info] Dropping {dropped} false-positive rows before metadata export."
        )
    df = df[keep].reset_index(drop=True)
    rel = rel[keep].reset_index(drop=True)
    top = top[keep].reset_index(drop=True)
    patient = patient[keep].reset_index(drop=True)
    label = top.map(FOLDER_TO_LABEL).astype(int)

    out_df = pd.DataFrame({
        "Patient_ID": top + "_" + patient,          # unique across folders
        "Age": pd.to_numeric(df["Age"], errors="coerce"),
        "Image_Laterality": df["Image Laterality"].astype(str).str.strip().str.upper().str[0],
        "View_Position": df["View Position"].astype(str).str.strip().str.upper(),
        "Cancer": (label == 1).astype(int),
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
    print("Per-image class counts (0=Normal,1=Abnormal):")
    cls = out_df["Cancer"].astype(int)
    print(cls.value_counts().sort_index().to_dict())
    per_patient = out_df.groupby("Patient_ID").apply(
        lambda g: 1 if g["Cancer"].max() else 0,
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
