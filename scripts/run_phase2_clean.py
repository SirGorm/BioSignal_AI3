"""Phase-2 only launcher for the 5 v2 models that skipped it.

train_optuna.py supports resume — re-running with the SAME --run-dir but
WITHOUT --skip-phase2 will:
  1. Detect the 10 completed Phase-1 trials in optuna.db (instant resume)
  2. Skip Phase 1 (n_remaining = 0)
  3. Run Phase 2: 3 seeds × 5 folds × 50 epochs on the Phase-1 best HPs

Models targeted:
  - feat-lstm
  - raw-cnn1d
  - raw-lstm
  - raw-cnn_lstm
  - raw-tcn

(rf-cpu and feat-mlp from v1 already have full multi-eval results.)

Parallelism: 2 GPU jobs at a time.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
SPLITS = ROOT / "configs" / "splits_clean.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

STEPS = [
    # (slug, arch, variant, run_dir_name)
    ("feat-lstm",      "lstm",         "features", "optuna_clean_v2-features-lstm"),
    ("raw-cnn1d",      "cnn1d_raw",    "raw",      "optuna_clean_v2-raw-cnn1d_raw"),
    ("raw-lstm",       "lstm_raw",     "raw",      "optuna_clean_v2-raw-lstm_raw"),
    ("raw-cnn_lstm",   "cnn_lstm_raw", "raw",      "optuna_clean_v2-raw-cnn_lstm_raw"),
    ("raw-tcn",        "tcn_raw",      "raw",      "optuna_clean_v2-raw-tcn_raw"),
]

MAX_GPU_JOBS = 2


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def make_cmd(arch, variant, run_dir):
    return [
        sys.executable, "scripts/train_optuna.py",
        "--arch", arch, "--variant", variant,
        "--num-workers", "0",
        "--run-dir", str(run_dir),
        "--n-trials", "10",
        "--phase1-epochs", "15",   # ignored on resume
        "--phase2-epochs", "50",
        "--phase2-seeds", "42", "1337", "7",
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits", str(SPLITS),
        # NO --skip-phase2 — that's the whole point.
    ]


def launch(slug, cmd, log_dir):
    log_path = log_dir / f"{slug}.log"
    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    fh.write(f"# {slug}\n# cmd: {' '.join(str(x) for x in cmd)}\n"
             f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return {"slug": slug, "pid": proc.pid, "proc": proc, "fh": fh,
            "log": str(log_path), "started": time.time()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"phase2_clean_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    print(f"=== Phase 2 launcher ({len(STEPS)} models, max {MAX_GPU_JOBS} parallel) ===")
    print(f"Log dir: {log_dir}\n")

    pending = list(STEPS)
    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "running": [{"slug": r["slug"], "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "log": r["log"]} for r in running.values()],
            "pending": [s[0] for s in pending],
            "finished": finished,
        }, indent=2))

    while pending or running:
        # Schedule.
        i = 0
        while i < len(pending):
            slug, arch, variant, dirname = pending[i]
            run_dir = ROOT / "runs" / dirname
            if step_done(run_dir):
                print(f"[{slug}] SKIP — Phase 2 already done")
                finished.append({"slug": slug, "status": "resumed",
                                  "elapsed_s": 0, "rc": 0,
                                  "run_dir": str(run_dir)})
                pending.pop(i); continue
            if len(running) >= MAX_GPU_JOBS:
                i += 1; continue
            cmd = make_cmd(arch, variant, run_dir)
            r = launch(slug, cmd, log_dir)
            running[slug] = r
            print(f"[{slug}] LAUNCHED pid={r['pid']} log={r['log']}")
            pending.pop(i)
        write_status()

        time.sleep(20)

        # Poll.
        done = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is not None:
                r["fh"].flush(); r["fh"].close()
                elapsed = int(time.time() - r["started"])
                status = "ok" if rc == 0 else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": r["log"]})
                done.append(slug)
        for s in done:
            running.pop(s)
        write_status()

    print(f"\n=== ALL PHASE 2 DONE ===")
    for r in finished:
        print(f"  {r['slug']}: {r['status']} ({r['elapsed_s']}s)")
    print(f"\nFull status: {status_path}")


if __name__ == "__main__":
    main()
