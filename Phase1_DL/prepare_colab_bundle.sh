#!/usr/bin/env bash
# Build the two zip files you upload to Google Drive to train on Colab:
#
#   colab_bundle/code.zip         - Phase1_DL source (small, no data)
#   colab_bundle/data_bundle.zip  - preprocessing cache + metadata + split
#                                   manifest + current checkpoint (so Colab
#                                   continues your progress instead of
#                                   starting over)
#
# Run from inside Phase1_DL/:
#   bash prepare_colab_bundle.sh
#
# Upload both zips into a Google Drive folder named exactly "NLBS_Phase1"
# (e.g. MyDrive/NLBS_Phase1/code.zip, MyDrive/NLBS_Phase1/data_bundle.zip) -
# the Colab notebook expects that layout.

set -euo pipefail
cd "$(dirname "$0")"

OUT=colab_bundle
rm -rf "$OUT"
mkdir -p "$OUT"

echo "== code.zip (source only) =="
zip -q -r "$OUT/code.zip" . \
    -x "outputs/*" -x "checkpoints/*" -x "tensorboard/*" -x "logs/*" \
    -x "data/*" -x "colab_bundle/*" -x "*__pycache__*" -x "*.pyc"
du -h "$OUT/code.zip"

echo ""
echo "== data_bundle.zip (cache + metadata + manifest + checkpoint) =="
STAGE=$(mktemp -d)
mkdir -p "$STAGE/nlbs_data"
cp -r outputs/preproc_cache "$STAGE/nlbs_data/preproc_cache"
cp data/metadata.csv "$STAGE/nlbs_data/metadata.csv"
if [ -f outputs/patient_manifest.csv ]; then
    cp outputs/patient_manifest.csv "$STAGE/nlbs_data/patient_manifest.csv"
    echo "  included patient_manifest.csv (locks in the exact train/val/test split)"
else
    echo "  WARNING: outputs/patient_manifest.csv not found yet - Colab will"
    echo "           derive its own split (should be identical given the fixed"
    echo "           seed, but not guaranteed across environments)."
fi
if [ -f checkpoints/last.pth ]; then
    # Staged separately from the live Drive checkpoint dir: the notebook only
    # copies this in as a *seed* the first time (if Drive has no progress of
    # its own yet), so re-running the notebook later never clobbers whatever
    # Colab has already trained past this point.
    mkdir -p "$STAGE/seed_checkpoint"
    cp checkpoints/last.pth "$STAGE/seed_checkpoint/last.pth"
    [ -f checkpoints/best_model.pth ] && cp checkpoints/best_model.pth "$STAGE/seed_checkpoint/best_model.pth"
    [ -f checkpoints/best_weights.pth ] && cp checkpoints/best_weights.pth "$STAGE/seed_checkpoint/best_weights.pth"
    echo "  included seed_checkpoint/ (from checkpoints/last.pth) - Colab will"
    echo "           RESUME your current local progress on first run"
else
    echo "  no checkpoints/last.pth yet - Colab will start a fresh run"
fi

(cd "$STAGE" && zip -q -r "$OLDPWD/$OUT/data_bundle.zip" .)
rm -rf "$STAGE"
du -h "$OUT/data_bundle.zip"

echo ""
echo "Done. Upload both files from '$OUT/' to Google Drive folder MyDrive/NLBS_Phase1/"
echo "Then open NLBS_Colab_Training.ipynb in Colab and run all cells."
