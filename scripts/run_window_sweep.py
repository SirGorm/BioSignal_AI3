"""Window-size sweep orchestrator.

Trains the same set of architectures at 4 window sizes (1, 2, 2.5, 4 s) so
we can compare which window length is best for each task. All other settings
(LOSO splits, soft_overlap reps, EMG RMS envelope, uncertainty weighting) are
held fixed.

Run dirs are tagged `optuna_clean_{TAG}-w{WIN}-{ARCH}` where WIN encodes the
window size, e.g. `w1s`, `w2s`, `w2_5s`, `w4s`.

Per-arch budget defaults (sweep, not deep search):
  - n_trials: 50 (vs 200-400 in v7/v8)
  - phase1_epochs: 30
  - phase2_epochs: 150 (vs 300 in v7/v8)
  - patience: 10

Total jobs: len(WINDOWS) × len(ARCHS) NN + RF (RF is window-independent).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
RUN_DIR_BASE = ROOT / "runs"
RF_FEATURES_DIR = ROOT / "runs" / "optuna_clean_v1" / "features"
SPLITS_LOSO = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

# Architectures (kept in v7 layout: 6 multi-task + 2 fatigue-only)
MULTI_ARCHS = [
    ("multi-feat-mlp",      "mlp",          "features"),
    ("multi-feat-lstm",     "lstm",         "features"),
    ("multi-raw-cnn1d",     "cnn1d_raw",    "raw"),
    ("multi-raw-lstm",      "lstm_raw",     "raw"),
    ("multi-raw-cnn_lstm",  "cnn_lstm_raw", "raw"),
    ("multi-raw-tcn",       "tcn_raw",      "raw"),
]
FATIGUE_ARCHS = [
    ("fatigue-raw-tcn",  "tcn_raw",  "raw"),
    ("fatigue-raw-lstm", "lstm_raw", "raw"),
]

WINDOWS = [1.0, 2.0, 2.5, 4.0]


def _wlabel(window_s: float) -> str:
    """Encode window length for directory name: 2.0 → 2s, 2.5 → 2_5s."""
    return f"{window_s:g}s".replace('.', '_')


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def step_done_rf(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def make_steps(n_trials: int, phase1_epochs: int, phase2_epochs: int,
               patience: int, tag: str, windows: list, archs_filter: str
               ) -> list[dict]:
    seeds = [42, 1337, 7]
    steps = []

    archs = []
    if archs_filter in ('all', 'multi'):
        archs += [(s, a, v, ['exercise', 'phase', 'fatigue', 'reps'])
                  for s, a, v in MULTI_ARCHS]
    if archs_filter in ('all', 'fatigue'):
        archs += [(s, a, v, ['fatigue']) for s, a, v in FATIGUE_ARCHS]

    common = [
        "--n-trials", str(n_trials),
        "--phase1-epochs", str(phase1_epochs),
        "--phase2-epochs", str(phase2_epochs),
        "--patience", str(patience),
        "--phase2-seeds", *[str(s) for s in seeds],
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits", str(SPLITS_LOSO),
        "--reps-mode", "soft_overlap",
        "--num-workers", "0",
    ]

    # NN steps for each (window, arch) combo
    for window_s in windows:
        wl = _wlabel(window_s)
        for slug, arch, variant, tasks in archs:
            run_dir_name = f"optuna_clean_{tag}-w{wl}-{slug}"
            run_dir = RUN_DIR_BASE / run_dir_name
            cmd = [sys.executable, "scripts/train_optuna.py",
                   "--arch", arch, "--variant", variant,
                   "--run-dir", str(run_dir),
                   "--tasks", *tasks,
                   "--window-s", str(window_s),
                   *common]
            steps.append({"slug": f"w{wl}-{slug}", "kind": "nn", "cmd": cmd,
                          "run_dir": run_dir, "device": "gpu",
                          "is_done": step_done})

    # RF on CPU — window-independent (uses set_features.parquet for fatigue/reps)
    rf_dir = RUN_DIR_BASE / f"optuna_clean_{tag}-rf"
    rf_dir.mkdir(parents=True, exist_ok=True)
    rf_cmd = [sys.executable, "scripts/train_lgbm.py",
              "--run-dir", str(rf_dir),
              "--features-dir", str(RF_FEATURES_DIR),
              "--exclude-recordings", *EXCLUDE,
              "--splits", str(SPLITS_LOSO)]
    steps.append({"slug": "rf-cpu", "kind": "rf", "cmd": rf_cmd,
                  "run_dir": rf_dir, "device": "cpu",
                  "is_done": step_done_rf})

    return steps


def launch(step: dict, log_dir: Path) -> dict:
    log_path = log_dir / f"{step['slug']}.log"
    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    fh.write(f"# {step['slug']}\n# cmd: {' '.join(str(x) for x in step['cmd'])}\n"
             f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
    fh.flush()
    proc = subprocess.Popen(step["cmd"], stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT))
    return {"pid": proc.pid, "proc": proc, "fh": fh, "log": str(log_path),
            "started": time.time(), "device": step["device"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-gpu-jobs", type=int, default=4)
    ap.add_argument("--tag", default="wsweep")
    ap.add_argument("--windows", nargs='+', type=float, default=WINDOWS,
                    help="Window sizes in seconds (default: 1 2 2.5 4)")
    ap.add_argument("--archs", choices=['all', 'multi', 'fatigue'],
                    default='all',
                    help="Which architecture set to sweep")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"optuna_{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    steps = make_steps(args.n_trials, args.phase1_epochs, args.phase2_epochs,
                        args.patience, args.tag, args.windows, args.archs)
    n_nn = sum(1 for s in steps if s["device"] == "gpu")
    n_rf = sum(1 for s in steps if s["device"] == "cpu")
    print(f"=== {args.tag.upper()}: {n_nn} NN ({len(args.windows)} windows × "
          f"{n_nn // len(args.windows)} archs) + {n_rf} RF ===")
    print(f"  windows: {args.windows}")
    print(f"  n_trials/job: {args.n_trials}")
    print(f"  max GPU jobs: {args.max_gpu_jobs}")
    print(f"Log dir: {log_dir}\n")

    pending = list(steps)
    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "windows": args.windows,
            "n_trials": args.n_trials,
            "running": [{"slug": s, "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "device": r["device"], "log": r["log"]}
                         for s, r in running.items()],
            "pending": [s["slug"] for s in pending],
            "finished": finished,
        }, indent=2))

    while pending or running:
        i = 0
        while i < len(pending):
            step = pending[i]
            if step["is_done"](step["run_dir"]):
                print(f"[{step['slug']}] SKIP — already done")
                finished.append({"slug": step["slug"], "status": "resumed",
                                  "elapsed_s": 0, "rc": 0})
                pending.pop(i); continue
            gpu_count = sum(1 for r in running.values() if r["device"] == "gpu")
            cpu_count = sum(1 for r in running.values() if r["device"] == "cpu")
            if step["device"] == "gpu" and gpu_count >= args.max_gpu_jobs:
                i += 1; continue
            if step["device"] == "cpu" and cpu_count >= 1:
                i += 1; continue
            r = launch(step, log_dir)
            running[step["slug"]] = r
            print(f"[{step['slug']}] LAUNCHED pid={r['pid']} ({step['device']})")
            pending.pop(i)
        write_status()

        time.sleep(20)

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

    print(f"\n=== {args.tag.upper()} ALL DONE ===")
    for r in finished:
        print(f"  {r['slug']}: {r['status']} ({r['elapsed_s']}s)")
    print(f"\nFull status: {status_path}")


if __name__ == "__main__":
    main()
