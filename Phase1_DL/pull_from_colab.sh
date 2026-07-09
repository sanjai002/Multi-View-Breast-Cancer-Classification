#!/usr/bin/env bash
# Bring Colab's Drive-persisted training progress back to this local machine so
# the CPU run can continue from wherever Colab left off.
#
# 1. Download the "NLBS_Phase1" folder from Google Drive (Drive web UI: right-
#    click the folder > Download), or sync it with `rclone`/Google Drive
#    Desktop, so you have it locally, e.g. at ~/Downloads/NLBS_Phase1/.
# 2. Run this script pointing at that downloaded folder:
#      bash pull_from_colab.sh ~/Downloads/NLBS_Phase1
#
# It compares epochs and only overwrites local checkpoints if Colab's progress
# is FURTHER ALONG (never accidentally regresses your local run).

set -euo pipefail
cd "$(dirname "$0")"

SRC="${1:?Usage: bash pull_from_colab.sh <path to downloaded NLBS_Phase1 folder>}"
SRC_CKPT="$SRC/checkpoints/last.pth"

if [ ! -f "$SRC_CKPT" ]; then
    echo "No checkpoints/last.pth found under $SRC — nothing to pull."
    exit 1
fi

python3 - "$SRC_CKPT" "checkpoints/last.pth" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
src_epoch = torch.load(src, map_location="cpu", weights_only=False).get("epoch", -1)
try:
    dst_epoch = torch.load(dst, map_location="cpu", weights_only=False).get("epoch", -1)
except FileNotFoundError:
    dst_epoch = -1
print(f"Colab checkpoint epoch: {src_epoch}   |   local checkpoint epoch: {dst_epoch}")
if src_epoch <= dst_epoch:
    print("Local is already at or past Colab's progress - not overwriting.")
    sys.exit(1)
print("Colab is further along - will pull it in.")
PY

if [ $? -eq 0 ]; then
    cp "$SRC_CKPT" checkpoints/last.pth
    [ -f "$SRC/checkpoints/best_model.pth" ] && cp "$SRC/checkpoints/best_model.pth" checkpoints/best_model.pth
    [ -f "$SRC/checkpoints/best_weights.pth" ] && cp "$SRC/checkpoints/best_weights.pth" checkpoints/best_weights.pth
    # Bring the metrics history along too so training_curves.png keeps one
    # continuous record instead of restarting the plot.
    [ -f "$SRC/outputs/metrics_history.csv" ] && cp "$SRC/outputs/metrics_history.csv" outputs/metrics_history.csv
    echo ""
    echo "Pulled Colab's checkpoint + history into checkpoints/ and outputs/."
    echo "Resume locally with:"
    echo "  kill -- -\$(cat outputs/train_group.pid) 2>/dev/null   # stop any local run first"
    echo "  setsid nohup bash run_forever.sh >> outputs/train_run.log 2>&1 < /dev/null &"
    echo "(run_cpu_training.py already has cfg.train.resume = True, so it will"
    echo " continue from this checkpoint's epoch automatically.)"
fi
