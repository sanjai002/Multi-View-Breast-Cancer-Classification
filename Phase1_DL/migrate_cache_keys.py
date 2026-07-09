"""One-time migration: re-key the preprocessing cache from absolute-path hashes
to data_root-relative-path hashes (see dataset.py's ``_cache_path``).

Why: the cache used to be keyed by ``hashlib.md5(absolute_path + size)``. That
breaks the moment the dataset is mounted at a different absolute location (e.g.
uploading the cache to Google Colab), because every lookup misses even though
the pixel data is identical. This script copies each already-cached array to
its new, portable key — no DICOM re-decoding needed — so a cache built locally
can be ziped up and reused unchanged on another machine.

Safe to run while local training is active: it only reads metadata + existing
cache files and writes NEW files; it does not touch or import anything the
running trainer process has already loaded in memory.

Run:
    python migrate_cache_keys.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import get_config  # noqa: E402


def old_key(abs_path: str, size: int) -> str:
    return hashlib.md5(f"{abs_path}_{size}".encode()).hexdigest()


def new_key(abs_path: str, data_root: str, size: int) -> str:
    try:
        rel = os.path.relpath(abs_path, data_root)
    except ValueError:
        rel = os.path.basename(abs_path)
    rel = rel.replace(os.sep, "/")
    return hashlib.md5(f"{rel}_{size}".encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=224,
                   help="MUST match the image_size the cache was actually built "
                        "with (run_cpu_training.py currently uses 224, NOT the "
                        "config.py default of 512).")
    args, _ = ap.parse_known_args()

    cfg = get_config()
    cache_dir = cfg.data.cache_dir
    size = args.image_size
    data_root = cfg.paths.data_root

    meta = pd.read_csv(cfg.paths.metadata_csv)
    paths = meta["Image_Path"].astype(str).tolist()
    print(f"metadata rows: {len(paths)}  |  cache_dir: {cache_dir}  |  image_size: {size}")

    migrated, missing, already = 0, 0, 0
    for p in paths:
        ap = os.path.abspath(p)
        ok = old_key(ap, size)
        nk = new_key(ap, data_root, size)
        old_file = os.path.join(cache_dir, f"{ok}.npy")
        new_file = os.path.join(cache_dir, f"{nk}.npy")
        if os.path.isfile(new_file):
            already += 1
            continue
        if os.path.isfile(old_file):
            with open(old_file, "rb") as src, open(new_file, "wb") as dst:
                dst.write(src.read())
            migrated += 1
        else:
            missing += 1

    print(f"\nmigrated : {migrated}")
    print(f"already new-keyed: {already}")
    print(f"not cached (will decode on first use): {missing}")
    total_new = sum(1 for f in os.listdir(cache_dir) if f.endswith(".npy"))
    print(f"\ncache_dir now contains {total_new} .npy files "
          f"({sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if f.endswith('.npy')) / 1e6:.0f} MB)")
    print("\nOld-keyed files were left in place (harmless, ignored by the new code).")
    print("You can delete them later to save space with:")
    print("  python migrate_cache_keys.py --purge-old   (see --help)")


if __name__ == "__main__":
    if "--purge-old" in sys.argv:
        cfg = get_config()
        size = 224
        if "--image-size" in sys.argv:
            size = int(sys.argv[sys.argv.index("--image-size") + 1])
        meta = pd.read_csv(cfg.paths.metadata_csv)
        keep = set()
        for p in meta["Image_Path"].astype(str):
            ap = os.path.abspath(p)
            keep.add(new_key(ap, cfg.paths.data_root, size) + ".npy")
        removed = 0
        for f in os.listdir(cfg.data.cache_dir):
            if f.endswith(".npy") and f not in keep:
                os.remove(os.path.join(cfg.data.cache_dir, f))
                removed += 1
        print(f"purged {removed} old-keyed cache files")
    else:
        main()
