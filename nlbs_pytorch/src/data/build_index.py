"""
Build a BREAST-LEVEL index for dual-view (CC+MLO) training.

Each row = one breast (patient + side) with its CC and MLO DICOM paths and a
binary label. Label comes from the NLBS folder convention: a side folder ending
in ``-c`` (e.g. ``left-c``) is the biopsy-confirmed cancer breast (verified:
100% Cancer==1), everything else is normal. The ``False Positive`` folder is
ignored (binary normal-vs-cancer task).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import load_config


def build_index(cfg) -> pd.DataFrame:
    src = Path(cfg.paths.source_root)
    df = pd.read_csv(cfg.paths.metadata_csv,
                     usecols=["File Path", "Age", "Image Laterality", "View Position"])
    parts = df["File Path"].astype(str).str.split("\\")
    df["folder"] = parts.str[0]
    df["patient_id"] = parts.str[1]
    df["side"] = parts.str[2]
    df["view"] = parts.str[3].str.upper()
    df = df[df["folder"].isin(list(cfg.data.use_folders))].copy()
    df["diskpath"] = df["File Path"].astype(str).str.replace("\\", "/", regex=False)
    df["abspath"] = df["diskpath"].map(lambda p: str(src / p))
    df = df[df["abspath"].map(os.path.exists)]

    rows = []
    for (folder, pid, side), g in df.groupby(["folder", "patient_id", "side"]):
        label = 1 if str(side).endswith("-c") else 0
        laterality = "L" if str(side).startswith("left") else "R"
        cc = g.loc[g["view"] == "CC", "abspath"]
        mlo = g.loc[g["view"] == "MLO", "abspath"]
        cc_p = cc.iloc[0] if len(cc) else (mlo.iloc[0] if len(mlo) else None)
        mlo_p = mlo.iloc[0] if len(mlo) else (cc.iloc[0] if len(cc) else None)
        if cc_p is None or mlo_p is None:
            continue                               # need at least one usable view
        age = g["Age"].dropna()
        rows.append({
            "patient_id": pid, "side": side, "laterality": laterality,
            "label": label, "cc_path": cc_p, "mlo_path": mlo_p,
            "age": int(age.iloc[0]) if len(age) else -1, "folder": folder,
        })
    idx = pd.DataFrame(rows)
    return idx


def main(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    Path(cfg.paths.output_root).mkdir(parents=True, exist_ok=True)
    idx = build_index(cfg)
    idx.to_csv(cfg.paths.index_csv, index=False)
    print(f"[build_index] {len(idx)} breasts | "
          f"cancer={int((idx.label==1).sum())} normal={int((idx.label==0).sum())} "
          f"| patients={idx.patient_id.nunique()}")
    print(f"[build_index] wrote {cfg.paths.index_csv}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
