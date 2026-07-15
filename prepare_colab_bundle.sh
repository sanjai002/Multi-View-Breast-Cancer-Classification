#!/usr/bin/env bash
# Prepare Colab bundle: code.zip and data_bundle.zip

set -euo pipefail

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/colab_bundle"
CODE_ZIP="$OUT_DIR/code.zip"
DATA_ZIP="$OUT_DIR/data_bundle.zip"

mkdir -p "$OUT_DIR"

echo "Creating code bundle..."
# Include Phase1_DL code and scripts
zip -r "$CODE_ZIP" Phase1_DL scripts -x "*/__pycache__/*" "*.pyc" >/dev/null

echo "Creating data bundle..."
# Include cache, metadata and patient manifest if present
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/data"

if [ -d "Phase1_DL/outputs/cache" ]; then
  cp -r Phase1_DL/outputs/cache "$TEMP_DIR/data/"
fi

if [ -f "Phase1_DL/data/metadata.csv" ]; then
  mkdir -p "$TEMP_DIR/data"
  cp Phase1_DL/data/metadata.csv "$TEMP_DIR/data/"
fi

if [ -f "Phase1_DL/outputs/metadata_balanced.csv" ]; then
  cp Phase1_DL/outputs/metadata_balanced.csv "$TEMP_DIR/data/"
fi

if [ -f "Phase1_DL/outputs/patient_manifest.csv" ]; then
  cp Phase1_DL/outputs/patient_manifest.csv "$TEMP_DIR/data/"
fi

if [ -f "Phase1_DL/outputs/patient_manifest_balanced.csv" ]; then
  cp Phase1_DL/outputs/patient_manifest_balanced.csv "$TEMP_DIR/data/"
fi

if [ -f "Phase1_DL/outputs/cache_origin.txt" ]; then
  cp Phase1_DL/outputs/cache_origin.txt "$TEMP_DIR/data/"
fi

pushd "$TEMP_DIR" >/dev/null
zip -r "$DATA_ZIP" data >/dev/null
popd >/dev/null

rm -rf "$TEMP_DIR"

echo "Bundles created: $CODE_ZIP and $DATA_ZIP"
