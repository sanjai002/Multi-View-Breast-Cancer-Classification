"""
CSAW-CC data audit for Phase 2 (RL screening-interval policy).

Reproduces every empirical number quoted in METHODOLOGY.md. Run before any
modelling work so that design claims stay tied to the actual file on disk:

    python data_audit.py [--csv CSAW-CC_breast_cancer_screening_data.csv]

Nothing here is model code; it is the evidence base for the MDP design.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).parent / "CSAW-CC_breast_cancer_screening_data.csv"


def to_exam_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 4 image rows of a screening round into one exam row.

    'rad_' variables are exam-level and repeat across the four rows, so max()
    is an identity here rather than an aggregation choice. libra_* are
    image-level and are averaged over the four views.
    """
    ex = (
        df.groupby(["anon_patientid", "exam_year"])
        .agg(
            n_img=("anon_filename", "count"),
            case=("x_case", "max"),
            age_bin=("x_age", "max"),
            timing=("rad_timing", "max"),
            recall=("rad_recall", "max"),
            r1=("rad_r1", "max"),
            r2=("rad_r2", "max"),
            tumour_type=("x_type", "max"),
            node_met=("x_lymphnode_met", "max"),
            laterality=("x_cancer_laterality", "first"),
            pct_density=("libra_percentdensity", "mean"),
            breast_area=("libra_breastarea", "mean"),
        )
        .reset_index()
        .sort_values(["anon_patientid", "exam_year"])
    )
    ex["gap"] = ex.groupby("anon_patientid").exam_year.diff()
    return ex


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ex = to_exam_level(df)
    per_patient = ex.groupby("anon_patientid").agg(
        n_visits=("exam_year", "count"), case=("case", "max")
    )

    section("1. Scale")
    print(f"image rows       : {len(df):,}")
    print(f"screening exams  : {len(ex):,}")
    print(f"patients         : {ex.anon_patientid.nunique():,}")
    print(f"case patients    : {(per_patient.case == 1).sum():,}")
    print(f"control patients : {(per_patient.case == 0).sum():,}")
    print(f"patient-level cancer prevalence: "
          f"{(per_patient.case == 1).mean():.1%}  (case-enriched by design)")

    section("2. Views per exam (quality control)")
    print(ex.n_img.value_counts().to_string())
    dupes = ex[ex.n_img != 4]
    if len(dupes):
        print(f"\n!! {len(dupes)} exam(s) without exactly 4 views -> QC rule needed:")
        print(dupes[["anon_patientid", "exam_year", "n_img"]].to_string(index=False))

    section("3. Trajectory length (visits per patient)")
    print(pd.crosstab(per_patient.n_visits, per_patient.case,
                      rownames=["n_visits"], colnames=["case"]).to_string())
    n = per_patient.n_visits
    print(f"\npatients with >=2 visits : {(n >= 2).sum():,}")
    print(f"patients with  1 visit   : {(n == 1).sum():,} "
          f"({(n == 1).mean():.1%}) -> single-decision episodes, no transition")
    print(f"total observed transitions: {(n - 1).sum():,}")

    section("4. Inter-exam gap = the behaviour policy's action")
    print(ex.gap.value_counts(dropna=False).sort_index().to_string())
    print("\ngap (rows) x age_bin (cols); 1 = 40-55y, 2 = 55+y")
    print(pd.crosstab(ex.gap, ex.age_bin).to_string())
    g = ex.gap.dropna()
    print(f"\nshare of gaps in {{1,2,3}} years: {g.isin([1, 2, 3]).mean():.1%}")
    print("NOTE: gaps are structurally determined by age bin (Swedish programme:")
    print("      18-month invites for 40-54, 24-month for 55-74) -> the logged")
    print("      behaviour policy is near-deterministic given age. This is the")
    print("      positivity/overlap problem that drives the offline-RL design.")

    section("5. Outcome timing (rad_timing) — cases only")
    cases = ex[ex.case == 1]
    print("exam-level rad_timing (1=screen-detected <60d, 2=interval 60-729d, 3=prior 730d+):")
    print(cases.timing.value_counts(dropna=False).sort_index().to_string())

    seqs = cases.groupby("anon_patientid").timing.apply(
        lambda s: tuple(s.astype("Int64"))
    )
    violations = sum(
        1
        for t in seqs
        if any(
            pd.notna(t[i]) and pd.notna(t[i + 1]) and t[i] < t[i + 1]
            for i in range(len(t) - 1)
        )
    )
    print(f"\nnon-monotone timing sequences: {violations} "
          "(0 confirms timing decreases toward diagnosis)")
    print("\nmost common per-patient timing sequences:")
    print(seqs.value_counts().head(8).to_string())

    last = cases.sort_values("exam_year").groupby("anon_patientid").timing.last()
    print("\nrad_timing of each case patient's FINAL observed exam:")
    print(last.value_counts(dropna=False).sort_index().to_string())
    print("  1 -> cancer found AT that screen (screen-detected)")
    print("  2 -> cancer surfaced 60-729d AFTER it  (INTERVAL cancer = a miss)")
    print("  3 -> diagnosis >=730d later; images stop at diagnosis (censored)")

    section("6. Recall behaviour")
    print("recall rate by case status:")
    print(ex.groupby("case").recall.mean().to_string())
    print("\nrecall (cols) x rad_timing (rows), case exams:")
    print(pd.crosstab(cases.timing, cases.recall, dropna=False).to_string())
    print("\n-> interval cancers were overwhelmingly NOT recalled at the prior")
    print("   screen: this is the empirical miss signal the reward must price.")

    section("7. Stage at detection: screen-detected vs interval")
    det = cases[cases.timing.isin([1.0, 2.0])]
    typ = pd.crosstab(det.timing, det.tumour_type, normalize="index")
    cnt = pd.crosstab(det.timing, det.tumour_type)
    print("x_type counts (1=in situ, 2=invasive <=15mm, 3=invasive >15mm):")
    print(cnt.to_string())
    print("\nrow-normalised:")
    print(typ.round(3).to_string())
    nod = pd.crosstab(det.timing, det.node_met)
    print("\nlymph-node metastasis counts (0=no, 1=yes):")
    print(nod.to_string())
    print("\nnode-positive rate:")
    print((nod[1.0] / nod.sum(axis=1)).round(3).to_string())
    print("\n-> interval cancers are systematically MORE ADVANCED. The cost of")
    print("   delay is therefore ESTIMATED FROM THIS DATA, not assumed.")

    section("8. Risk covariates available for the state")
    print("x_age is BINNED to 2 levels — continuous age is NOT recoverable:")
    print(df.x_age.value_counts().sort_index().to_string())
    print("\nlibra_percentdensity (per-exam mean over 4 views):")
    print(ex.pct_density.describe().round(2).to_string())
    print("\nmean percent density by age bin:")
    print(ex.groupby("age_bin").pct_density.mean().round(2).to_string())

    section("9. Missingness")
    miss = pd.DataFrame({
        "nulls": df.isna().sum(),
        "pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    print(miss.to_string())
    print(f"\nexams with missing rad_recall: {ex.recall.isna().sum():,} "
          f"({ex.recall.isna().mean():.1%}) -> requires an explicit "
          "missing-indicator channel, not imputation to 0.")


if __name__ == "__main__":
    main()
