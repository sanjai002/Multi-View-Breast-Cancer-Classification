"""Live training monitor for the NLBS Phase 1 run (standalone, stdlib only).

Run in a separate terminal while training is going:

    python monitor_training.py            # refresh every 15 s
    python monitor_training.py --interval 5
    python monitor_training.py --once      # print one snapshot and exit

Reads outputs/train_run.log, outputs/metrics_history.csv and outputs/train.pid.
Shows process status, current epoch/batch, per-epoch validation metrics
(including per-class recall so you can watch the class imbalance), and an ETA.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
LOG = os.path.join(OUT, "train_run.log")
HIST = os.path.join(OUT, "metrics_history.csv")
PIDF = os.path.join(OUT, "train.pid")

_TS = "%Y-%m-%d %H:%M:%S"


def _ts(line: str):
    try:
        return dt.datetime.strptime(line[:19], _TS)
    except Exception:
        return None


def pid_alive():
    try:
        pid = int(open(PIDF).read().strip())
    except Exception:
        return None, False
    try:
        os.kill(pid, 0)
        return pid, True
    except OSError:
        return pid, False


def read_log():
    try:
        with open(LOG, errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def read_history():
    try:
        with open(HIST) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def parse_progress(lines):
    total_epochs = None
    start = None
    last_batch = None
    epoch_end_ts = []
    for l in lines:
        if "Starting training for" in l:
            m = re.search(r"for (\d+) epochs", l)
            if m:
                total_epochs = int(m.group(1))
            start = _ts(l)
        elif re.search(r"Epoch \d+ \[", l):
            m = re.search(r"Epoch (\d+) \[(\d+)/(\d+)\] loss=([\d.]+)", l)
            if m:
                last_batch = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                              float(m.group(4)), _ts(l))
        elif "| val_loss=" in l:
            epoch_end_ts.append(_ts(l))
    return total_epochs, start, last_batch, epoch_end_ts


def fmt_td(seconds):
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def snapshot() -> str:
    lines = read_log()
    hist = read_history()
    pid, alive = pid_alive()
    total_epochs, start, last_batch, ends = parse_progress(lines)
    now = dt.datetime.now()

    out = []
    out.append("=" * 78)
    out.append(f"  NLBS Phase 1 - training monitor      {now.strftime(_TS)}")
    out.append("=" * 78)

    status = "RUNNING" if alive else ("FINISHED/STOPPED" if pid else "no pid file")
    runtime = fmt_td((now - start).total_seconds()) if start else "?"
    out.append(f"  process   : PID {pid}  [{status}]     runtime: {runtime}")

    completed = len(ends)
    tot = total_epochs or "?"
    if last_batch:
        e, b, nb, loss, _ = last_batch
        pct = 100.0 * b / nb if nb else 0
        out.append(f"  progress  : epoch {e}/{tot}   batch {b}/{nb} ({pct:4.1f}%)   "
                   f"last train loss {loss:.4f}")
    else:
        out.append(f"  progress  : epochs completed {completed}/{tot} (warming up...)")

    # ETA from mean epoch duration.
    if len(ends) >= 1 and start and total_epochs:
        bounds = [start] + ends
        durs = [(bounds[i + 1] - bounds[i]).total_seconds() for i in range(len(ends))]
        mean = sum(durs) / len(durs)
        remaining = max(0, total_epochs - completed)
        eta = mean * remaining
        out.append(f"  pace      : ~{fmt_td(mean)}/epoch   "
                   f"ETA to epoch {total_epochs}: ~{fmt_td(eta)} "
                   f"(sooner if early-stopping triggers)")

    out.append("-" * 78)
    if hist:
        names = [k[len("recall_"):] for k in hist[0] if k.startswith("recall_")]
        head = f"  {'ep':>3} {'val_loss':>9} {'macroF1':>8} {'balAcc':>7}"
        for n in names:
            head += f" {('rec_' + n)[:9]:>9}"
        out.append("  VALIDATION METRICS PER EPOCH")
        out.append(head)
        for r in hist[-12:]:
            row = (f"  {r['epoch']:>3} {float(r['val_loss']):>9.4f} "
                   f"{float(r['val_macro_f1']):>8.4f} {float(r['val_balanced_acc']):>7.4f}")
            for n in names:
                row += f" {float(r['recall_' + n]):>9.3f}"
            out.append(row)
        best = max(hist, key=lambda r: float(r["val_macro_f1"]))
        out.append(f"  best so far: epoch {best['epoch']}  macroF1={float(best['val_macro_f1']):.4f}")
    else:
        out.append("  (no completed epochs yet - metrics_history.csv not written)")

    out.append("-" * 78)
    out.append("  recent log:")
    for l in lines[-6:]:
        out.append("   " + l[:120])
    out.append("=" * 78)
    out.append("  plots: outputs/training_curves.png | outputs/epoch_plots/  "
               "(Ctrl+C to exit monitor)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    try:
        while True:
            print("\033[2J\033[H", end="")   # clear screen
            print(snapshot(), flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor stopped.")


if __name__ == "__main__":
    main()
