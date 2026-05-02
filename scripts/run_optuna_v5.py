"""V5 orchestrator: multi-task with soft_overlap+LOSO + fatigue-only specialised runs.

Two parallel tracks:

  TRACK A (multi-task) — 6 NN + RF, all 4 tasks active.
    --reps-mode soft_overlap   (Wang et al. 2026 overlap-fraction labels)
    --splits configs/splits_clean_loso.csv  (7-fold LOSO)
    --tasks exercise phase fatigue reps
    Output: runs/optuna_clean_{TAG}-{features-{mlp,lstm}, raw-{cnn1d,lstm,cnn_lstm,tcn}_raw}/

  TRACK B (fatigue-only) — 4 best fatigue archs from v4-resume.
    Top picks (Pearson r in v4-resume):
      raw-cnn_lstm  +0.28
      raw-cnn1d     +0.17
      raw-tcn       +0.18
      feat-MLP      +0.15
    Same data + LOSO + soft_overlap, but only fatigue contributes to the loss.
    Output: runs/optuna_clean_{TAG}-fatigue-{arch}/

  + RF on CPU (re-run for v5 baseline consistency).

Total: 6 + 4 + 1 = 11 jobs.
Parallelism: 4 GPU + 1 CPU.
Per-arch ETA: 25-50 min (matches v4-resume budget).
Total ETA: ~3-5 hours.
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

# Top fatigue architectures from v4-resume (sorted by Pearson r descending).
FATIGUE_TOP_ARCHS = [
    ("fatigue-raw-cnn_lstm", "cnn_lstm_raw", "raw"),    # r=+0.28
    ("fatigue-raw-cnn1d",    "cnn1d_raw",    "raw"),    # r=+0.17
    ("fatigue-raw-tcn",      "tcn_raw",      "raw"),    # r=+0.18
    ("fatigue-feat-mlp",     "mlp",          "features"),  # r=+0.15
]


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def step_done_rf(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def make_steps(n_trials: int, phase1_epochs: int, phase2_epochs: int,
               patience: int, tag: str = "v5") -> list[dict]:
    TAG = tag
    seeds = [42, 1337, 7]
    common_optuna = [
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

    def opt_step(slug, arch, variant, run_dir_name, tasks):
        run_dir = RUN_DIR_BASE / run_dir_name
        cmd = [sys.executable, "scripts/train_optuna.py",
               "--arch", arch, "--variant", variant,
               "--run-dir", str(run_dir),
               "--tasks", *tasks,
               *common_optuna]
        return {"slug": slug, "kind": "nn", "cmd": cmd, "run_dir": run_dir,
                "device": "gpu", "is_done": step_done}

    steps = []
    # TRACK A — multi-task with all 4 tasks
    multi_archs = [
        ("multi-feat-mlp",      "mlp",          "features"),
        ("multi-feat-lstm",     "lstm",         "features"),
        ("multi-raw-cnn1d",     "cnn1d_raw",    "raw"),
        ("multi-raw-lstm",      "lstm_raw",     "raw"),
        ("multi-raw-cnn_lstm",  "cnn_lstm_raw", "raw"),
        ("multi-raw-tcn",       "tcn_raw",      "raw"),
    ]
    for slug, arch, variant in multi_archs:
        # multi-task run-dir (e.g. optuna_clean_{TAG}-features-mlp)
        rd_name = f"optuna_clean_{TAG}-{variant}-{arch}"
        steps.append(opt_step(slug, arch, variant, rd_name,
                                tasks=["exercise", "phase", "fatigue", "reps"]))

    # TRACK B — fatigue-only specialised
    for slug, arch, variant in FATIGUE_TOP_ARCHS:
        rd_name = f"optuna_clean_{TAG}-{slug}"
        steps.append(opt_step(slug, arch, variant, rd_name,
                                tasks=["fatigue"]))

    # RF on CPU
    rf_run_dir = RUN_DIR_BASE / f"optuna_clean_{TAG}-rf"
    rf_run_dir.mkdir(parents=True, exist_ok=True)
    rf_cmd = [sys.executable, "scripts/train_lgbm.py",
              "--run-dir", str(rf_run_dir),
              "--features-dir", str(RF_FEATURES_DIR),
              "--exclude-recordings", *EXCLUDE,
              "--splits", str(SPLITS_LOSO)]
    steps.append({"slug": "rf-cpu", "kind": "rf", "cmd": rf_cmd,
                  "run_dir": rf_run_dir, "device": "cpu",
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
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--phase1-epochs", type=int, default=50)
    ap.add_argument("--phase2-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-gpu-jobs", type=int, default=4)
    ap.add_argument("--tag", default="v5",
                    help="Run-dir suffix: runs/optuna_clean_<tag>-<arch>/")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"optuna_{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    steps = make_steps(args.n_trials, args.phase1_epochs, args.phase2_epochs,
                        args.patience, args.tag)
    print(f"=== {args.tag.upper()}: {len(steps)} steps ({len(steps) - 1} NN + 1 RF) ===")
    print(f"   max GPU jobs: {args.max_gpu_jobs}")
    print(f"   n_trials={args.n_trials}, phase1_epochs={args.phase1_epochs}, "
          f"phase2_epochs={args.phase2_epochs}, patience={args.patience}")
    print(f"   reps_mode=soft_overlap  splits=splits_clean_loso.csv")
    print(f"Log dir: {log_dir}\n")

    pending = list(steps)
    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_trials": args.n_trials,
            "running": [{"slug": s, "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "device": r["device"], "log": r["log"]}
                         for s, r in running.items()],
            "pending": [s["slug"] for s in pending],
            "finished": finished,
        }, indent=2))

    while pending or running:
        # Schedule
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
            print(f"[{step['slug']}] LAUNCHED pid={r['pid']} ({step['device']}) "
                  f"log={r['log']}")
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

    print(f"\n=== V5 ALL DONE ===")
    for r in finished:
        print(f"  {r['slug']}: {r['status']} ({r['elapsed_s']}s)")
    print(f"\nFull status: {status_path}")


if __name__ == "__main__":
    main()
