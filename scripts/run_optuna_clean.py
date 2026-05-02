"""Parallel orchestrator for Optuna search on dataset_clean (post-renumber).

Runs 7 training jobs (5 NN architectures via train_optuna.py, RF via train_lgbm.py):
  - feat-mlp        (2 NN feature-input)
  - feat-lstm
  - raw-cnn1d       (4 NN raw-input)
  - raw-lstm
  - raw-cnn_lstm
  - raw-tcn
  - rf              (RandomForest, CPU only)

Excluded: recordings 004, 005, 007, 008, 009 (per user 2026-05-01).
Splits:   configs/splits_clean.csv (7 subjects: Vivian, Hytten, kiyomi,
                                      lucas 2, Tias, Juile, Raghild).

Parallelism: 2 NN jobs simultaneously on the GPU + 1 RF on CPU. Feature NNs
are scheduled before raw NNs (lighter, fail-fast). Each step is run as a
detached subprocess so this orchestrator just monitors+logs.

Usage:
    python scripts/run_optuna_clean.py [--n-trials 20] [--smoke-test]
        [--max-gpu-jobs 2]

Logs to logs/optuna_clean_<timestamp>/<step>.log.
Status JSON in logs/optuna_clean_<timestamp>/status.json (live updates).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
RUN_DIR_BASE = ROOT / "runs"
RF_RUN_DIR = ROOT / "runs" / "optuna_clean_v1"  # has features/ already
SPLITS = ROOT / "configs" / "splits_clean.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]


def step_done(run_dir: Path) -> bool:
    """Phase-2 completion signal."""
    if not run_dir.exists():
        return False
    cv = next(iter((run_dir / "phase2").rglob("cv_summary.json")), None)
    return cv is not None


def step_done_rf(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def make_steps(n_trials: int, smoke: bool, gpu_concurrency: int,
                phase1_epochs: int = 30, phase2_epochs: int = 50,
                patience: int = 5, run_dir_tag: str = "v3") -> list[dict]:
    """Return ordered list of step dicts. Order = priority for scheduling.

    v3 config (per user 2026-05-02): full search re-run with the GPU-resident
    fast path landed in src/training/loop.py + datasets.py. Each NN trial is
    now ~10-30× faster than v2, so Phase 2 (3 seeds × 5 folds × 50 epochs) is
    affordable on every architecture.

      - num_workers=0 (irrelevant — DataLoader bypassed entirely on cuda)
      - phase1_epochs=30, phase2_epochs=50 (full default)
      - Phase 2 ENABLED (--skip-phase2 removed)
      - n_trials=20 (full default)
      - max_gpu_jobs=4 (was 2 — GPU-resident dataset frees memory headroom)
    """
    nt = 1 if smoke else n_trials
    p1 = 5 if smoke else phase1_epochs
    p2 = 5 if smoke else phase2_epochs
    seeds = [42] if smoke else [42, 1337, 7]
    nw_feat = 0
    nw_raw = 0

    common_optuna = [
        "--n-trials", str(nt),
        "--phase1-epochs", str(p1),
        "--phase2-epochs", str(p2),
        "--patience", str(patience),
        "--phase2-seeds", *[str(s) for s in seeds],
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits", str(SPLITS),
    ]

    def opt_step(slug, arch, variant, num_workers):
        run_dir = RUN_DIR_BASE / f"optuna_clean_{run_dir_tag}-{variant}-{arch}"
        cmd = [sys.executable, "scripts/train_optuna.py",
               "--arch", arch, "--variant", variant,
               "--num-workers", str(num_workers),
               "--run-dir", str(run_dir),
               *common_optuna]
        return {"slug": slug, "kind": "nn", "cmd": cmd, "run_dir": run_dir,
                "device": "gpu", "is_done": step_done}

    steps = [
        # Feature NN
        opt_step("feat-mlp",       "mlp",          "features", nw_feat),
        opt_step("feat-lstm",      "lstm",         "features", nw_feat),
        # Raw NN
        opt_step("raw-cnn1d",      "cnn1d_raw",    "raw",      nw_raw),
        opt_step("raw-lstm",       "lstm_raw",     "raw",      nw_raw),
        opt_step("raw-cnn_lstm",   "cnn_lstm_raw", "raw",      nw_raw),
        opt_step("raw-tcn",        "tcn_raw",      "raw",      nw_raw),
    ]

    # RF on CPU — independent. Re-runs in ~4 min for consistency with the v3
    # set (uses the same window_features.parquet that the NN feature variants
    # train on, post-stride=100).
    rf_run_dir = RUN_DIR_BASE / f"optuna_clean_{run_dir_tag}-rf"
    rf_run_dir.mkdir(parents=True, exist_ok=True)
    rf_cmd = [sys.executable, "scripts/train_lgbm.py",
              "--run-dir", str(rf_run_dir),
              "--features-dir", str(RF_RUN_DIR / "features"),
              "--exclude-recordings", *EXCLUDE,
              "--splits", str(SPLITS)]
    steps.append({"slug": "rf-cpu", "kind": "rf", "cmd": rf_cmd,
                  "run_dir": rf_run_dir, "device": "cpu",
                  "is_done": step_done_rf})

    return steps


def launch(step: dict, log_dir: Path) -> dict:
    """Start subprocess detached (stdout/stderr to file). Return process info."""
    log_path = log_dir / f"{step['slug']}.log"
    f = open(log_path, "w", encoding="utf-8", buffering=1)
    f.write(f"# {step['slug']}\n# cmd: {' '.join(str(x) for x in step['cmd'])}\n"
            f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
    f.flush()
    proc = subprocess.Popen(step["cmd"], stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(ROOT))
    return {"pid": proc.pid, "proc": proc, "fh": f, "log": str(log_path),
            "started": time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--smoke-test", action="store_true",
                    help="1 trial, 5 epochs each")
    ap.add_argument("--max-gpu-jobs", type=int, default=4,
                    help="Concurrent NN training jobs on GPU (default 4)")
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tag", default="v3",
                    help="Run-dir suffix: runs/optuna_clean_<tag>-<arch>/")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"optuna_clean_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    steps = make_steps(args.n_trials, args.smoke_test, args.max_gpu_jobs,
                        phase1_epochs=args.phase1_epochs,
                        phase2_epochs=args.phase2_epochs,
                        patience=args.patience, run_dir_tag=args.tag)
    print(f"=== Optuna orchestrator: {len(steps)} steps "
          f"({'SMOKE' if args.smoke_test else 'FULL'}) ===")
    print(f"Log dir: {log_dir}")
    print(f"Excluded: {EXCLUDE}")
    print(f"Splits: {SPLITS}")
    print()

    pending = list(steps)
    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "smoke": args.smoke_test, "n_trials": args.n_trials,
            "max_gpu_jobs": args.max_gpu_jobs,
            "running": [{"slug": s, "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "log": r["log"]} for s, r in running.items()],
            "pending": [s["slug"] for s in pending],
            "finished": finished,
        }, indent=2))

    while pending or running:
        # Try to schedule new jobs.
        gpu_count = sum(1 for s in running.values() if s.get("device") == "gpu")
        cpu_count = sum(1 for s in running.values() if s.get("device") == "cpu")
        i = 0
        while i < len(pending):
            step = pending[i]
            if step["is_done"](step["run_dir"]):
                print(f"[{step['slug']}] SKIP — already done at {step['run_dir']}")
                finished.append({"slug": step["slug"], "status": "resumed",
                                  "elapsed_s": 0, "log": None,
                                  "run_dir": str(step["run_dir"])})
                pending.pop(i)
                continue
            if step["device"] == "gpu" and gpu_count >= args.max_gpu_jobs:
                i += 1; continue
            if step["device"] == "cpu" and cpu_count >= 1:
                i += 1; continue
            # Launch.
            proc_info = launch(step, log_dir)
            proc_info["device"] = step["device"]
            running[step["slug"]] = proc_info
            print(f"[{step['slug']}] LAUNCHED pid={proc_info['pid']} "
                  f"({step['device']}) log={proc_info['log']}")
            if step["device"] == "gpu": gpu_count += 1
            if step["device"] == "cpu": cpu_count += 1
            pending.pop(i)
        write_status()

        # Poll running jobs.
        time.sleep(20)
        done_slugs = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is not None:
                r["fh"].flush(); r["fh"].close()
                elapsed = int(time.time() - r["started"])
                status = "ok" if rc == 0 else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": r["log"]})
                done_slugs.append(slug)
        for s in done_slugs:
            running.pop(s)
        write_status()

    print("\n=== ALL DONE ===")
    for r in finished:
        print(f"  {r['slug']}: {r['status']} ({r['elapsed_s']}s) — {r.get('log','-')}")
    print(f"\nFull status: {status_path}")


if __name__ == "__main__":
    main()
