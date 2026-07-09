# NEXT STEPS — complete runbook

Everything below can be done without me. Local commands run from inside `Phase1_DL/`:
```bash
cd "/mnt/NewVolume/projects/breast cancer detection/Phase1_DL"
```

---

## Current status (as of last update)

Two training runs may be active in parallel — **they are independent, stopping
one does not affect the other**:

| | Where | How to check |
|---|---|---|
| **CPU run** | This machine, background process | `kill -0 $(cat outputs/train.pid) 2>/dev/null && echo ALIVE \|\| echo DONE` |
| **Colab run** | Google Colab notebook (your browser) | Look at the notebook's cell 6 output, or check `MyDrive/NLBS_Phase1/outputs/metrics_history.csv` |

Both use the **same fixed config**: class-weighted focal loss, **balanced
sampler OFF** (see "known issue" below), SAM (Colab only), EMA, progressive
unfreezing, 224px, same train/val/test split (`patient_manifest.csv`).

### ⚠️ Known issue we hit and fixed
An earlier version had the **balanced sampler ON at the same time as** class-
weighted focal loss. Stacking both over-corrects for the class imbalance and
collapses the model to always predicting one single class (symptoms: `AUC
≈0.5`, one class at 100% recall, others at 0%). **Fixed** by turning the
sampler off (keep only the loss weighting) in both `run_cpu_training.py` and
`run_colab_training.py`, and both were restarted fresh from ImageNet weights
(not resumed from the collapsed checkpoint). If you ever see `sens=0.333,
spec=0.667` with one class's recall pinned at exactly 1.0, that's this bug —
check `cfg.data.use_balanced_sampler` is `False`.

---

## A. How to stop training

### Stop the CPU run
It runs under a keep-alive wrapper that auto-restarts it if killed (e.g. after
an OOM) — so **kill the whole process group**, not just the python process,
or it will just come back:
```bash
kill -- -$(cat outputs/train_group.pid)
```
Verify it's actually gone:
```bash
kill -0 $(cat outputs/train.pid) 2>/dev/null && echo "still alive" || echo "stopped"
```
The best checkpoint so far is always saved — stopping early loses nothing.

### Stop the Colab run
In the Colab notebook: **Runtime > Interrupt execution** (or just close the
tab / let the session idle out). Checkpoints are saved to Google Drive after
every epoch, so nothing is lost either way.

---

## B. Watch training while it runs

**CPU (local):**
```bash
python monitor_training.py            # live dashboard, refreshes every 15 s
python monitor_training.py --once     # single snapshot
tail -f outputs/train_run.log         # raw log
```
Live plots (updated every epoch): `outputs/training_curves.png` (loss +
per-class recall/precision), `outputs/epoch_plots/epoch_XXX_val_confusion.png`,
`outputs/metrics_history.csv`.

**Colab:** don't run a second cell in the same notebook while training is
executing (Colab runs cells one at a time, so it would just queue/interrupt).
Instead, open **Google Drive in your browser** → `MyDrive/NLBS_Phase1/outputs/`
and open `training_curves.png` / `metrics_history.csv` directly — refresh the
page anytime to see the latest epoch without touching the running cell.

---

## C. Is training finished?

Stops automatically at 40 epochs **or** when early stopping triggers (val
macro-F1 hasn't improved for 12 epochs). Look for this line in the log:
```
Training complete. Best macro_f1=...
```

Checkpoints produced (in `checkpoints/` locally, or `MyDrive/NLBS_Phase1/checkpoints/`
for Colab):
- `best_model.pth` — best epoch (weights + epoch + config) ← used automatically by test
- `best_weights.pth` — best weights only
- `epoch_000.pth … epoch_0NN.pth` — one per epoch
- `last.pth` — most recent (has optimizer state; used to resume)
- `feature_extractor.pth` — encoder only (written at the end)

---

## D. Generate all final results + Phase-2 inputs ← the main step

**If the CPU run produced your final model:**
```bash
python -m training.test
```
**If the Colab run produced your final model:** run cell 8 in the notebook
(`run_colab_test.py`) — same thing, writes to `MyDrive/NLBS_Phase1/outputs/`
instead.

Both load `best_model.pth` and evaluate the **held-out test set** (900
patients, never seen during training), writing:

| File | What it is |
|---|---|
| `classification_report.pdf` | Full metric report (per-class P/R/F1, sensitivity, specificity, AUC) |
| `confusion_matrix.png`, `confusion_matrix_normalized.png` | Test confusion matrices |
| `roc_curve.png`, `precision_recall.png`, `calibration_curve.png` | Diagnostic curves |
| `test_metrics.json` | All headline numbers (copy into your paper's tables) |
| `prediction_probabilities.csv`, `patient_predictions.csv` | Per-patient predictions (all splits) |
| `patient_features.npy` (+ `patient_feature_index.csv`) | Per-patient embeddings → **Phase 2 input** |
| `image_features.npy` (+ `image_feature_index.csv`) | Per-view embeddings → **Phase 2 input** |
| `gradcam_images/<patient>.png` | Grad-CAM / Grad-CAM++ / Score-CAM / Integrated-Gradients overlays |

Local CPU note: `test.py` takes a while (test images cache on first read;
explainability adds a few minutes). Skip Grad-CAM for speed:
`python -m training.test --no-explain`.

---

## E. Read the results

- **Do not judge by accuracy** (majority class = Normal ≈ 72% of patients — a
  model that always says "Normal" scores ~72% while being useless). Look at
  **macro-F1**, **balanced accuracy**, and especially **per-class recall for
  Cancer** in `classification_report.pdf` / the confusion matrix.
- `training_curves.png` bottom-left panel shows whether Cancer recall climbed
  steadily during training (not pinned at 0.0 or 1.0 the whole time — that
  would indicate the collapse bug from section "Known issue" above).

**If Cancer recall is still too low** (but not collapsed), the fastest fix
needs no retraining — threshold tuning: lower the decision threshold for the
Cancer class on the validation set to trade some precision for higher
sensitivity. (Ask me next session and I'll add `evaluation/threshold_tuning.py`.)

---

## F. Resume / restart training

Resume is **automatic** (`cfg.train.resume = True` in both launchers): on any
(re)start it loads `checkpoints/last.pth` (model + optimizer + EMA + epoch +
metrics history) and continues from the next epoch.

- **Resume after a stop:**
  ```bash
  setsid nohup bash run_forever.sh >> outputs/train_run.log 2>&1 < /dev/null &
  ```
  (Colab: just re-run the training cell — it resumes from the Drive checkpoint.)
- **Train more epochs after it finished:** raise `cfg.train.epochs` in
  `run_cpu_training.py` (or `run_colab_training.py`), then relaunch the same way.
- **Restart from scratch instead** (e.g. after a bad run like the collapse
  bug): delete the checkpoints first, otherwise resume will reload the bad state.
  ```bash
  rm -f checkpoints/*.pth outputs/metrics_history.csv
  rm -rf outputs/epoch_plots && mkdir -p outputs/epoch_plots
  ```
  On Colab, delete the three files in `MyDrive/NLBS_Phase1/checkpoints/` via
  the Drive UI (or the Python snippet below) before re-running the train cell:
  ```python
  import shutil, os
  ckpt_dir = f'{DRIVE_ROOT}/checkpoints'
  shutil.rmtree(ckpt_dir, ignore_errors=True)
  os.makedirs(ckpt_dir, exist_ok=True)
  ```

---

## G. Run on Colab GPU (much faster, resumable back to this machine)

The raw dataset is 325 GB (too big to upload), so Colab trains from the small
**preprocessing cache** (already built by the CPU run) instead of raw DICOMs.
Checkpoints live on Google Drive, so a Colab disconnect never loses progress,
and you can pull that same checkpoint back here to keep training on CPU (or
vice versa) — same architecture, same train/val/test split, same cache format.

**One-time setup (rerun anytime the code changes):**
```bash
bash prepare_colab_bundle.sh
```
Creates `colab_bundle/code.zip` (source only, tiny) and `colab_bundle/data_bundle.zip`
(cache + metadata + split manifest + current checkpoint as a resume seed).
Upload both into a Google Drive folder named exactly **`MyDrive/NLBS_Phase1/`**
— **directly inside it**, not nested in a subfolder (a common mistake: dragging
the whole `colab_bundle` folder in creates `MyDrive/NLBS_Phase1/colab_bundle/...`,
which the notebook won't find — the two zip files must sit directly under
`NLBS_Phase1/`).

If you only changed code (not data), you don't need to rebuild/re-upload the
605 MB `data_bundle.zip` again — just re-zip and re-upload `code.zip`:
```bash
rm -f colab_bundle/code.zip
zip -q -r colab_bundle/code.zip . -x "outputs/*" -x "checkpoints/*" \
    -x "tensorboard/*" -x "logs/*" -x "data/*" -x "colab_bundle/*" \
    -x "*__pycache__*" -x "*.pyc"
```

**Then:** open `NLBS_Colab_Training.ipynb` in Google Colab (`Runtime > Change
runtime type > GPU`), and run cells in order:
1. Check GPU.
2. Mount Drive.
3. Unzip code + data to Colab's fast local disk.
4. Seed `MyDrive/NLBS_Phase1/checkpoints/` from your local progress **only if**
   Drive has no checkpoint of its own yet (safe to re-run; never clobbers
   further-along Colab progress). **Skip this step if you deliberately want a
   fresh start** — clear the Drive checkpoints folder first instead (§F).
5. Install dependencies.
6. **Train** (`run_colab_training.py`) — full spec: SAM + bf16 AMP + batch 16.
   Safe to interrupt/disconnect; re-running this cell resumes automatically.
   The startup banner prints the active config (including
   `balanced_samp: False` — verify this says `False`, not `True`) so you can
   confirm the right code is running without digging into files.
7. Live progress (only useful between runs / after a restart — see §B for
   checking progress while training is actively running).
8. Export final results (`run_colab_test.py`) — same artefacts as §D, written
   to `MyDrive/NLBS_Phase1/outputs/`.

**To verify the running code matches what you expect** (e.g. after a fix),
without interrupting training:
```python
!grep -n "use_balanced_sampler" /content/Phase1_DL/run_colab_training.py
```

**To bring Colab's progress back here and keep training on CPU:**
```bash
# after downloading the MyDrive/NLBS_Phase1 folder locally, e.g. to ~/Downloads/NLBS_Phase1
bash pull_from_colab.sh ~/Downloads/NLBS_Phase1
```
It only pulls in Colab's checkpoint if it's *further along* than your local
one (never regresses local progress), then the usual resume (§F) continues it
on CPU.

**Expected speedup:** the identical method that takes ~20–30 h on this CPU
finishes in roughly **1–2 h on a free Colab T4** (measured: ~1.5 s/batch,
262 batches/epoch ≈ 6–8 min/epoch, vs ~76 min/epoch on CPU).

---

## H. Hand off to Phase 2 (Reinforcement Learning)

Phase 2 consumes exactly these files (already in the right format), from
whichever run (CPU or Colab) produced your final model:
```
outputs/patient_features.npy        outputs/patient_feature_index.csv
outputs/image_features.npy          outputs/image_feature_index.csv
outputs/prediction_probabilities.csv
outputs/patient_predictions.csv
```
See **`../Phase2_RL/IMPLEMENTATION_GUIDE.md`** for the complete implementation
blueprint (MDP design, per-file spec, network architectures, training/eval
protocol) — detailed enough to implement without further clarification.
`../Phase2_RL/README.md` has the shorter overview.

---

## I. Troubleshooting — every scenario

### "The process died / isn't in the process list anymore"
Check the group PID, not just the trainer PID (the wrapper's process, not the
python subprocess it spawns):
```bash
kill -0 $(cat outputs/train_group.pid) 2>/dev/null && echo "wrapper alive" || echo "wrapper dead"
grep -c "auto-restart" outputs/train_run.log   # how many times it's already recovered from a crash
```
If the wrapper is dead too, something killed the whole group (manual stop,
system reboot, `pkill` from an unrelated command). Just relaunch (§F).
If you see repeated "auto-restart" messages close together in time (a crash
loop), that's a real problem, not transient — read the log lines right before
each restart for the actual error (usually `MemoryError`, `OSError`, or a
Python traceback) rather than assuming it'll keep self-healing forever.

### "Out of memory (OOM)" — this box has ~9GB RAM and it runs tight
This is the most likely local failure mode. Check current headroom:
```bash
free -h
ps -o pid,rss,%mem,etime -p $(cat outputs/train.pid)
```
The keep-alive wrapper (`run_forever.sh`) already auto-restarts and resumes
from the last checkpoint on any crash including OOM — so a single OOM is
**not an emergency**, just check it actually resumed (`grep "RESUMED" outputs/train_run.log`
or watch the epoch number continue climbing, not reset to 0). If it's
OOM-crash-looping repeatedly:
- Lower `cfg.train.batch_size` in `run_cpu_training.py` (currently 4 → try 2).
- Confirm `cfg.data.num_workers` is 1, not higher (each extra worker is a full
  process copy — this was the original cause of the very first OOM crash in
  this project; see project memory `nlbs-two-phase-project`).
- Close other applications on this machine competing for RAM.

### "Disk is full" / "No space left on device"
Disk was at 97% full (~16GB free) as of this writing — check current state:
```bash
df -h "/mnt/NewVolume"
du -sh outputs/preproc_cache checkpoints colab_bundle 2>/dev/null
```
Safe things to delete to free space (all regenerable):
- `checkpoints/epoch_*.pth` for epochs you don't need to keep individually
  (keep `best_model.pth`, `best_weights.pth`, `last.pth`) — these are the bulk
  of disk usage, ~100MB each.
- `colab_bundle/*.zip` after you've uploaded them to Drive (they're just a
  packaged copy of files already on disk elsewhere).
- Old-keyed cache entries, if `migrate_cache_keys.py --purge-old --image-size 224`
  hasn't been run recently (see project memory `nlbs-colab-training-setup`).
- **Never delete** `outputs/preproc_cache/` entirely unless you're fine with
  every image being re-decoded from DICOM on next use (slow, not broken —
  just a one-time cost per image again).

### "Colab says no GPU available" / quota exhausted
Free Colab GPU access is not guaranteed instantly — if `Runtime > Change
runtime type > GPU` doesn't get you one, or you see "you cannot currently
connect to a GPU" errors: wait a while and retry (quota resets), or fall back
to continuing the CPU run locally in the meantime (they're independent — see
"Current status" at the top of this doc). Nothing is lost either way.

### "Colab disconnected mid-training"
Expected and handled — checkpoints are on Google Drive, not Colab's ephemeral
disk. Just reopen the notebook, run cells 1-5 again (GPU check, mount, unzip,
seed — seeding is a no-op if Drive already has progress), then cell 6. It
resumes from the last completed epoch automatically (`cfg.train.resume = True`).

### "Colab notebook can't find data_bundle.zip / code.zip"
Almost always: the zip files were uploaded **nested inside a `colab_bundle/`
folder** in Drive instead of directly under `MyDrive/NLBS_Phase1/`. Check with:
```python
import os; os.listdir('/content/drive/MyDrive/NLBS_Phase1')
```
Should list `code.zip` and `data_bundle.zip` directly (with real MB sizes, not
`colab_bundle/ (dir)`). If nested, move them up one level in the Drive web UI.

### "Colab: AssertionError ... preproc_cache missing"
Either the upload of `data_bundle.zip` (605MB+) didn't fully complete/sync
before you ran the unzip cell, or it's nested wrong (see above). Diagnose:
```python
import os
print(os.path.getsize(f'{DRIVE_ROOT}/data_bundle.zip') / 1e6, 'MB')  # compare to local colab_bundle/data_bundle.zip size
!unzip -o "$DRIVE_ROOT/data_bundle.zip" -d /content/nlbs_data_bundle | tail -20
!find /content/nlbs_data_bundle -maxdepth 2
```
If the size is smaller than expected or unzip reports a central-directory
error, re-upload and **wait for 100% before running the notebook**.

### "Results look bad — one class always predicted, or near-zero macro-F1"
This is the class-collapse bug — see "Known issue" at the top of this doc and
project memory `nlbs-class-collapse-gotcha`. Symptoms: `AUC≈0.5000` exactly,
`sens=0.333`/`spec=0.667` exactly, one class's recall pinned at 1.0 and the
others at 0.0. Confirm `cfg.data.use_balanced_sampler = False` in both
launcher scripts, then **clear checkpoints and restart from scratch** (§F) —
resuming from an already-collapsed checkpoint won't fix itself.

### "CPU and Colab runs gave different final results — which do I trust?"
Both use the identical config, code, and train/val/test split (they share
`patient_manifest.csv`), so differences should be minor (randomness in
dropout/augmentation, or one simply trained more epochs than the other before
you stopped it). Prefer whichever trained **more epochs** /
had the **better validation macro-F1** at stopping time — check each run's
`outputs/metrics_history.csv` for the actual epoch count and best score
reached, don't just guess. You can also `pull_from_colab.sh` to bring Colab's
further-along checkpoint local and directly compare `test_metrics.json` from
both by running `python -m training.test` against each checkpoint (rename/
back up one `checkpoints/best_model.pth` before overwriting with the other,
so you can test both).

### "A few DICOM files fail to read" (warnings in the log)
Expected — a handful of source files in this dataset are corrupted/truncated
(e.g. `False Positive/p_5240/.../IM-0252-...dcm` — "pixel data less than
expected"). The loader catches this, logs a warning, and substitutes a
zero-filled masked view for that one view — it does not crash training and
does not silently corrupt other patients' data. Safe to ignore unless the
*same* warning repeats every epoch for a large fraction of patients (that
would indicate a real path/permissions problem, not a handful of bad files).

### "I changed a .py file and I'm not sure it's syntactically valid"
Always byte-compile before relying on a change (catches syntax errors in
seconds, before wasting an hours-long training run on a typo):
```bash
python3 -m compileall -q <changed_file.py>
```
`compileall` returns silently on success; any error prints a traceback with
the exact line.

### "How do I know Colab is running the code I think it's running?"
Without interrupting a live run:
```python
!grep -n "<the setting you changed>" /content/Phase1_DL/<file>.py
```
reads the literal file on Colab's disk — no ambiguity about caching/reload
issues, since each `!python script.py` is a fresh process reading from disk.

### General principle when something looks wrong and I'm not available
1. Read the actual log (`outputs/train_run.log` or the Colab cell output) —
   the real error/traceback is almost always there, don't guess.
2. Check the process is actually alive (`kill -0 ... 2>/dev/null`) before
   assuming it's stuck vs. just slow.
3. Check `outputs/metrics_history.csv` for the real per-epoch numbers rather
   than trusting a vague impression from scrolling logs.
4. Byte-compile after any code edit before relaunching.
5. When restarting after a fix that changes *training behavior* (not just
   cosmetic), clear the checkpoints first (§F) — resuming silently reloads
   whatever was there before the fix, which can look like "the fix didn't
   work" when actually it's still running the old broken state.
