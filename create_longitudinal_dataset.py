"""
create_longitudinal_dataset.py
==============================

Build a synthetic longitudinal mammography dataset from the patients RESERVED
by ``create_classification_dataset.py`` (so there is zero overlap with the
classification dataset). Creates 100 synthetic patients, each with 3-5 dated
visits following a cancer-progression trajectory.

Run AFTER the classification builder:
    python create_classification_dataset.py
    python create_longitudinal_dataset.py
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from tqdm import tqdm

import config
from utils import get_logger, set_global_seed, load_inventory

log = get_logger("longitudinal")

LABEL_NAME = {0: "normal", 1: "abnormal"}


@dataclass
class Visit:
    label: int
    src: object
    rel: str
    filename: str
    laterality: str
    view: str


@dataclass
class SynthPatient:
    sid: str
    original_pid: str
    base_age: int
    visits: list[Visit] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    ages: list[int] = field(default_factory=list)
    trajectory: str = ""
    split: str = ""


def trajectory_labels(length: int, rng: random.Random) -> list[int]:
    """Return a per-visit label sequence for a random trajectory type."""
    kind = rng.choice(["type1", "type2", "type3", "type4", "type5"])
    if kind == "type1":                         # Normal -> Normal -> Normal
        return [0] * length
    if kind == "type2":                         # Normal ... -> Abnormal
        return [0] * (length - 1) + [1]
    if kind == "type3":                         # Normal -> Abnormal -> Abnormal
        return [0] + [1] * (length - 1)
    if kind == "type4":                         # Abnormal -> Abnormal -> Abnormal
        return [1] * length
    seq = [rng.randint(0, 1) for _ in range(length)]     # Type 5: mixed
    if len(set(seq)) == 1:
        seq[rng.randrange(length)] ^= 1
    return seq


def build_reserved_pools(inv: pd.DataFrame, reserved: dict[str, list[str]]):
    """Return (normal_imgs, abnormal_imgs, base_patients, age_by_patient)."""
    reserved_ids = set(reserved["normal"]) | set(reserved["abnormal"])
    pool = inv[inv.patient_id.isin(reserved_ids)].copy()

    def to_visits(sub):
        return [Visit(r.label, r.src, r.rel, r.filename, r.laterality, r.view)
                for r in sub.itertuples(index=False)]

    normal_imgs = to_visits(pool[pool.label == 0])
    abnormal_imgs = to_visits(pool[pool.label == 1])

    # Base age per reserved patient (median of non-null ages; else None).
    age_by_patient: dict[str, int | None] = {}
    for pid, sub in pool.groupby("patient_id"):
        ages = [int(a) for a in sub["age"].tolist() if a is not None and pd.notna(a)]
        age_by_patient[pid] = int(pd.Series(ages).median()) if ages else None

    base_patients = sorted(reserved_ids)
    return normal_imgs, abnormal_imgs, base_patients, age_by_patient


def generate_patients(inv: pd.DataFrame, reserved: dict[str, list[str]],
                      rng: random.Random) -> list[SynthPatient]:
    normal_imgs, abnormal_imgs, base_patients, age_by_patient = \
        build_reserved_pools(inv, reserved)
    rng.shuffle(normal_imgs)
    rng.shuffle(abnormal_imgs)
    rng.shuffle(base_patients)
    log.info("reserved pool: %d normal + %d abnormal images across %d patients",
             len(normal_imgs), len(abnormal_imgs), len(base_patients))

    pools = {0: normal_imgs, 1: abnormal_imgs}
    patients: list[SynthPatient] = []
    for i in range(config.LONGITUDINAL_PATIENTS):
        length = rng.randint(config.LONG_MIN_VISITS, config.LONG_MAX_VISITS)
        if len(normal_imgs) + len(abnormal_imgs) < length:
            log.warning("reserved pool exhausted after %d patients", i)
            break
        wants = trajectory_labels(length, rng)

        visits: list[Visit] = []
        for want in wants:
            if pools[want]:
                visits.append(pools[want].pop())
            elif pools[1 - want]:               # adaptive fallback
                visits.append(pools[1 - want].pop())
            else:
                break
        if len(visits) < config.LONG_MIN_VISITS:
            break

        base_pid = base_patients[i % len(base_patients)]
        base_age = age_by_patient.get(base_pid) or rng.randint(45, 74)
        sid = f"Patient_{i + 1:06d}"
        traj = "_".join(LABEL_NAME[v.label] for v in visits)

        start_year = rng.randint(*config.LONG_START_YEAR_RANGE)
        d0 = date(start_year, rng.randint(1, 12), rng.randint(1, 28))
        dates, ages = [d0], [base_age]
        for k in range(1, len(visits)):
            dates.append(dates[-1] + timedelta(
                days=365 * config.LONG_YEARS_BETWEEN_VISITS + rng.randint(0, 20)))
            ages.append(base_age + (dates[-1].year - d0.year))

        patients.append(SynthPatient(sid=sid, original_pid=base_pid,
                                     base_age=base_age, visits=visits,
                                     dates=dates, ages=ages, trajectory=traj))
    return patients


def assign_splits(patients: list[SynthPatient], rng: random.Random) -> None:
    ids = [p.sid for p in patients]
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(config.TRAIN_FRAC * n))
    n_val = int(round(config.VAL_FRAC * n))
    split_of = {}
    for sid in ids[:n_train]:
        split_of[sid] = "train"
    for sid in ids[n_train:n_train + n_val]:
        split_of[sid] = "validation"
    for sid in ids[n_train + n_val:]:
        split_of[sid] = "test"
    for p in patients:
        p.split = split_of[p.sid]


def materialize(patients: list[SynthPatient]) -> pd.DataFrame:
    rows = []
    total = sum(len(p.visits) for p in patients)
    with tqdm(total=total, desc="copy longitudinal", unit="img") as bar:
        for p in patients:
            for vnum, (v, dt, age) in enumerate(zip(p.visits, p.dates, p.ages), 1):
                vdir = config.LONGITUDINAL_DIR / p.sid / f"Visit_{vnum:02d}"
                vdir.mkdir(parents=True, exist_ok=True)
                dest = vdir / v.filename          # preserve original filename
                shutil.copy2(v.src, dest)          # copy, never move
                rows.append({
                    "Synthetic_Patient_ID": p.sid,
                    "Original_Patient_ID": p.original_pid,
                    "Visit_Number": vnum,
                    "Visit_Date": dt.isoformat(),
                    "Image_Path": f"{p.sid}/Visit_{vnum:02d}/{v.filename}",
                    "Cancer_Label": v.label,
                    "Age": age,
                    "Image_Laterality": v.laterality,
                    "View_Position": v.view,
                    "Trajectory_Type": p.trajectory,
                    "Split": p.split,
                })
                bar.update(1)
    return pd.DataFrame(rows)


def validate(meta: pd.DataFrame, patients: list[SynthPatient],
             manifest: dict, inv: pd.DataFrame) -> None:
    log.info("=== VALIDATION CHECKS ===")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        log.info("  [%s] %s %s", "OK" if cond else "FAIL", name, detail)

    # No duplicate images (unique output paths + each source used once).
    dup_out = meta["Image_Path"].duplicated().sum()
    used_src = [v.rel for p in patients for v in p.visits]
    dup_src = len(used_src) - len(set(used_src))
    check("no duplicate images", dup_out == 0 and dup_src == 0,
          f"({dup_out} dup paths, {dup_src} reused sources)")

    # No overlap with the classification dataset (image + patient level).
    class_used = set(manifest["used_src_rel"])
    overlap_img = len(set(used_src) & class_used)
    class_patients = {pid for lst in manifest["classification_patients"].values()
                      for pid in lst}
    long_orig = set(p.original_pid for p in patients)
    check("no overlap with classification",
          overlap_img == 0 and len(long_orig & class_patients) == 0,
          f"({overlap_img} shared images)")

    # No patient leakage across synthetic splits.
    sets = {s: set(meta.loc[meta.Split == s, "Synthetic_Patient_ID"])
            for s in config.SPLITS}
    leak = (sets["train"] & sets["validation"]) | (sets["train"] & sets["test"]) \
        | (sets["validation"] & sets["test"])
    check("no patient leakage across splits", len(leak) == 0, f"({len(leak)} shared)")

    # Metadata rows match files on disk / all DICOMs exist.
    missing = sum(0 if (config.LONGITUDINAL_DIR / p).exists() else 1
                  for p in meta["Image_Path"])
    check("metadata rows match files on disk", missing == 0, f"({missing} missing)")

    # Folder structure valid (every synthetic patient dir + visit dirs exist).
    bad_struct = 0
    for p in patients:
        for vnum in range(1, len(p.visits) + 1):
            if not (config.LONGITUDINAL_DIR / p.sid / f"Visit_{vnum:02d}").is_dir():
                bad_struct += 1
    check("folder structure valid", bad_struct == 0, f"({bad_struct} bad)")

    # Dates strictly increasing per synthetic patient.
    bad_dates = 0
    for _, g in meta.sort_values(["Synthetic_Patient_ID", "Visit_Number"]) \
            .groupby("Synthetic_Patient_ID"):
        d = pd.to_datetime(g["Visit_Date"]).tolist()
        if not all(d[i] < d[i + 1] for i in range(len(d) - 1)):
            bad_dates += 1
    check("dates strictly increasing", bad_dates == 0, f"({bad_dates} bad patients)")

    # Every synthetic patient has 3-5 visits.
    vc = meta.groupby("Synthetic_Patient_ID")["Visit_Number"].count()
    bad_visits = int(((vc < config.LONG_MIN_VISITS) | (vc > config.LONG_MAX_VISITS)).sum())
    check("every patient has 3-5 visits", bad_visits == 0, f"({bad_visits} bad)")

    if not ok:
        raise SystemExit("Validation FAILED.")
    log.info("=== ALL CHECKS PASSED ===")


def print_statistics(meta: pd.DataFrame, patients: list[SynthPatient],
                     manifest: dict, inv: pd.DataFrame) -> None:
    used_src = {v.rel for p in patients for v in p.visits}
    class_used = set(manifest["used_src_rel"])
    unused = len(inv) - len(class_used) - len(used_src)
    log.info("=== LONGITUDINAL STATISTICS ===")
    log.info("Longitudinal Patients : %d", meta["Synthetic_Patient_ID"].nunique())
    log.info("Longitudinal Visits   : %d", len(meta))
    log.info("Normal visits         : %d", int((meta.Cancer_Label == 0).sum()))
    log.info("Abnormal visits       : %d", int((meta.Cancer_Label == 1).sum()))
    for s in config.SPLITS:
        log.info("%-11s patients : %d", s,
                 meta.loc[meta.Split == s, "Synthetic_Patient_ID"].nunique())
    log.info("Classification images used : %d", len(class_used))
    log.info("Unused images (remaining)  : %d", unused)


def main() -> None:
    set_global_seed()
    config.ensure_output_dirs()
    rng = random.Random(config.SEED + 1)        # distinct stream from classification

    if not config.MANIFEST_JSON.exists():
        raise SystemExit("Run create_classification_dataset.py first "
                         f"(missing {config.MANIFEST_JSON}).")
    with open(config.MANIFEST_JSON) as f:
        manifest = json.load(f)
    reserved = manifest["reserved_longitudinal_patients"]

    inv = load_inventory(log)
    patients = generate_patients(inv, reserved, rng)
    assign_splits(patients, rng)
    log.info("generated %d synthetic longitudinal patients", len(patients))

    meta = materialize(patients)
    meta.to_csv(config.LONGITUDINAL_META_CSV, index=False)
    log.info("wrote %s", config.LONGITUDINAL_META_CSV)

    validate(meta, patients, manifest, inv)
    print_statistics(meta, patients, manifest, inv)
    log.info("Done. Output at %s", config.OUTPUT_ROOT)


if __name__ == "__main__":
    main()
