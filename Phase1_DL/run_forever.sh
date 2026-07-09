#!/usr/bin/env bash
# Keep NLBS Phase 1 training alive on this low-RAM box: if the trainer is killed
# (e.g. OOM), automatically restart it — it resumes from checkpoints/last.pth, so
# no completed epoch is lost. Exits when training finishes normally.
cd "$(dirname "$0")" || exit 1
echo $$ > outputs/train_group.pid          # process-group leader PID (for stopping)

attempt=0
until env OMP_NUM_THREADS=6 python3 -u run_cpu_training.py; do
    status=$?
    attempt=$((attempt + 1))
    echo "==================================================================="
    echo "$(date '+%F %T') trainer exited (code $status) — auto-restart #$attempt,"
    echo "                    resuming from last checkpoint in 15s ..."
    echo "==================================================================="
    sleep 15
done
echo "$(date '+%F %T') TRAINING FINISHED NORMALLY (run_forever exiting)."
