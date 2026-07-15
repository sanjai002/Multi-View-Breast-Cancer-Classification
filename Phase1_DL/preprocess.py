"""
preprocess.py

Deterministic DICOM preprocessing and on-disk cache generation for the
minimal NLBS four-view pipeline.

What this file does:
- Reads image paths from the image-level metadata CSV.
- Loads each DICOM exactly once.
- Applies deterministic preprocessing:
    1) MONOCHROME1 correction
    2) rescale/windowing to float image
    3) simple breast foreground extraction
    4) crop to the foreground bounding box
    5) resize to a square canvas (224x224 by default)
- Saves each preprocessed image as a uint8 .npy file in the cache directory.
- Verifies the cache before training.

Training must never read DICOM files. It should only consume the .npy cache
produced by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
VIEW_ORDER: Tuple[str, ...] = ("LCC", "LMLO", "RCC", "RMLO")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CACHE_DTYPE = np.uint8


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def _clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _norm_laterality(x) -> str:
    s = _clean_text(x).upper()
    return s[:1] if s else ""


def _norm_view(x) -> str:
    s = _clean_text(x).upper().replace("-", "").replace("_", "")
    if not s:
        return ""
    if s in {"CC", "CRANIOCAUDAL", "CRANIOCAUDALVIEW"}:
        return "CC"
    if s in {"MLO", "MEDIOLATERALOBLIQUE", "MEDIOLATERALOBLIQUEVIEW"}:
        return "MLO"
    return s


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"Patient_ID", "Age", "Image_Laterality", "View_Position", "Cancer", "Image_Path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")

    out = df.copy()
    out["Patient_ID"] = out["Patient_ID"].astype(str)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Image_Laterality"] = out["Image_Laterality"].map(_norm_laterality)
    out["View_Position"] = out["View_Position"].map(_norm_view)
    out["Cancer"] = pd.to_numeric(out["Cancer"], errors="coerce").fillna(0).astype(int)
    out["Image_Path"] = out["Image_Path"].astype(str)
    return out


def cache_key_for(image_path: str, data_root: Path, image_size: int) -> str:
    """
    Generate the portable cache key.

    The key must depend on the path relative to data_root plus image_size.
    This ensures the same key is produced locally and on Colab, even if the
    absolute mount path differs.
    """
    abs_path = Path(image_path).expanduser().resolve()
    try:
        rel = os.path.relpath(str(abs_path), str(data_root.resolve()))
    except Exception:
        rel = abs_path.name
    rel = rel.replace(os.sep, "/")
    return hashlib.md5(f"{rel}_{image_size}".encode()).hexdigest()


def cache_path_for(image_path: str, data_root: Path, cache_dir: Path, image_size: int) -> Path:
    return cache_dir / f"{cache_key_for(image_path, data_root, image_size)}.npy"


# ---------------------------------------------------------------------
# DICOM decoding
# ---------------------------------------------------------------------
def _decode_dicom(path: Path) -> Tuple[np.ndarray, str]:
    """
    Decode a DICOM to a float32 array in [0, 1] and return laterality.

    The pixel conversion is intentionally simple and deterministic:
    - read pixel_array
    - apply rescale slope/intercept if present
    - invert MONOCHROME1
    - normalize robustly to [0, 1]
    """
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    # Robust normalization to [0, 1]
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

    laterality = str(getattr(ds, "ImageLaterality", getattr(ds, "Laterality", "")) or "")
    laterality = laterality.strip().upper()[:1]
    return out, laterality


# ---------------------------------------------------------------------
# Deterministic preprocessing
# ---------------------------------------------------------------------
def _largest_component_mask(mask_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255


def _foreground_mask(img_u8: np.ndarray) -> np.ndarray:
    """
    Simple breast foreground extraction.

    This is intentionally conservative and deterministic:
    - blur
    - Otsu threshold
    - keep largest connected component
    - fill with morphological close/open
    """
    blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heuristic: if threshold selects too little, invert it.
    if thr.mean() < 127:
        thr = cv2.bitwise_not(thr)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    largest = _largest_component_mask(thr)
    return largest


def _crop_to_mask(img_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return img_u8
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img_u8[y0:y1, x0:x1]


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=img.dtype)

    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size), dtype=img.dtype)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_image(
    dicom_path: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Return a uint8 square image in [0, 255].
    """
    img_f, laterality = _decode_dicom(dicom_path)

    img_u8 = (img_f * 255.0).astype(np.uint8)

    mask = _foreground_mask(img_u8)
    img_u8 = cv2.bitwise_and(img_u8, img_u8, mask=mask)
    img_u8 = _crop_to_mask(img_u8, mask)
    img_u8 = _resize_square(img_u8, image_size)

    # Optional orientation normalization:
    # flip so the denser side is on the left.
    col_mass = img_u8.sum(axis=0)
    half = img_u8.shape[1] // 2
    left_mass = col_mass[:half].sum()
    right_mass = col_mass[half:].sum()
    if right_mass > left_mass:
        img_u8 = np.fliplr(img_u8)

    return np.ascontiguousarray(img_u8, dtype=np.uint8)


# ---------------------------------------------------------------------
# Cache generation / verification
# ---------------------------------------------------------------------
def build_cache(
    metadata: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    overwrite: bool = False,
    strict: bool = True,
    limit: Optional[int] = None,
) -> Tuple[int, int, List[str]]:
    """
    Build the cache for every image in metadata.

    Returns:
        (created, skipped, errors)
    """
    ensure_dir(cache_dir)

    created = 0
    skipped = 0
    errors: List[str] = []

    rows = metadata.itertuples(index=False)
    if limit is not None and limit > 0:
        rows = list(rows)[:limit]

    for row in rows:
        image_path = Path(getattr(row, "Image_Path"))
        if not image_path.is_absolute():
            image_path = (data_root / image_path).resolve()

        if not image_path.is_file():
            msg = f"Missing DICOM: {image_path}"
            errors.append(msg)
            if strict:
                raise FileNotFoundError(msg)
            continue

        out_path = cache_path_for(str(image_path), data_root, cache_dir, image_size)

        if out_path.is_file() and not overwrite:
            skipped += 1
            continue

        try:
            arr = preprocess_image(image_path, image_size=image_size)
            np.save(out_path, arr.astype(DEFAULT_CACHE_DTYPE))
            created += 1
        except Exception as exc:
            msg = f"Failed to preprocess {image_path}: {exc}"
            errors.append(msg)
            if strict:
                raise RuntimeError(msg) from exc

    return created, skipped, errors


def verify_cache(
    metadata: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    limit: Optional[int] = None,
) -> Tuple[int, List[str]]:
    """
    Verify that every metadata row has a matching cache file.
    """
    missing: List[str] = []
    rows = metadata.itertuples(index=False)
    if limit is not None and limit > 0:
        rows = list(rows)[:limit]

    total = 0
    for row in rows:
        total += 1
        image_path = Path(getattr(row, "Image_Path"))
        if not image_path.is_absolute():
            image_path = (data_root / image_path).resolve()
        out_path = cache_path_for(str(image_path), data_root, cache_dir, image_size)
        if not out_path.is_file():
            missing.append(str(image_path))

    return total, missing


def summarize_cache(cache_dir: Path) -> Tuple[int, int]:
    files = [p for p in cache_dir.glob("*.npy") if p.is_file()]
    total = len(files)
    size_mb = int(round(sum(p.stat().st_size for p in files) / 1e6))
    return total, size_mb


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and verify the NLBS image cache")
    p.add_argument(
        "--metadata-csv",
        type=str,
        default="outputs/metadata.csv",
        help="Path to image-level metadata.csv produced by prepare_metadata.py",
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing the raw DICOM files",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="outputs/preproc_cache",
        help="Directory where .npy cache files are written",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output size for cached images",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not stop on first missing file / preprocessing error",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify cache presence; do not build anything",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_csv}")
    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    metadata = load_metadata(metadata_csv)

    if args.verify_only:
        total, missing = verify_cache(
            metadata,
            data_root=data_root,
            cache_dir=cache_dir,
            image_size=args.image_size,
            limit=args.limit,
        )
        if missing:
            print(f"Verified {total} rows. Missing cache files: {len(missing)}")
            print("First missing examples:")
            for p in missing[:20]:
                print("  ", p)
            raise SystemExit(1)
        print(f"Verified {total} rows. All cache files are present.")
        total_files, size_mb = summarize_cache(cache_dir)
        print(f"Cache directory: {cache_dir}")
        print(f"Cache files: {total_files}")
        print(f"Cache size: {size_mb} MB")
        return

    created, skipped, errors = build_cache(
        metadata,
        data_root=data_root,
        cache_dir=cache_dir,
        image_size=args.image_size,
        overwrite=args.overwrite,
        strict=not args.no_strict,
        limit=args.limit,
    )

    total, missing = verify_cache(
        metadata,
        data_root=data_root,
        cache_dir=cache_dir,
        image_size=args.image_size,
        limit=args.limit,
    )

    total_files, size_mb = summarize_cache(cache_dir)

    print("=" * 72)
    print("Cache build summary")
    print("=" * 72)
    print(f"Metadata rows      : {total}")
    print(f"Created            : {created}")
    print(f"Skipped            : {skipped}")
    print(f"Missing after build: {len(missing)}")
    print(f"Cache files        : {total_files}")
    print(f"Cache size         : {size_mb} MB")
    if missing:
        print("First missing cache examples:")
        for p in missing[:20]:
            print("  ", p)
        raise SystemExit(1)
    if errors and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()