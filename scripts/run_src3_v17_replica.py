"""Replicate v17 single-task feature-NN sweep on src3.

Matrix:
    archs   = mlp, lstm
    windows = 1.0, 2.0, 5.0
    tasks   = exercise, phase, fatigue, reps
    folds   = 7 LOSO (configs/splits_clean_loso.csv)
    seeds   = 3 (cfg.training.seeds)
    -> 2 × 3 × 4 = 24 jobs (each does 21 fold/seed runs internally)

Best-HP-source: runs/v17_fst-{task}-{arch}-w{wlabel}s/best_hps.json
Soft-mode:      reps=soft_overlap, phase=soft (matches v17 run_window_sweep.py)
Excludes:       recording_004,005,007,008,009 (matches v17 EXCLUDE list)
Output:         runs/src3_v17_replica_<ts>/<task>-<arch>-w<wlabel>s/

Parallelism: --max-jobs N spawns N subprocesses on the GPU concurrently.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
SPLITS_LOSO = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

ARCHS = ["mlp", "lstm"]
WINDOWS = [1.0, 2.0, 5.0]
TASKS = ["exercise", "phase", "fatigue", "reps"]


def _wlabel(w: float) -> str:
    """Encode window length for v17 dir name: 1.0->1s, 2.0->2s, 5.0->5s."""
    return f"{w:g}s"


def _v17_best_hp(task: str, arch: str, w: float) -> dict:
    """Load v17 single-task best_hps for this (task, arch, window)."""
    d = ROOT / "runs" / f"v17_fst-{task}-{arch}-w{_wlabel(w)}"
    p = d / "best_hps.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing v17 HP file: {p}")
    return json.loads(p.read_text())["best_hps"]


def _hp_to_overrides(arch: str, hp: dict) -> list[str]:
    """Convert v17 HP dict -> OmegaConf dotted overrides for src3 train.

    src3 reads training.{lr, weight_decay} and models_nn.features.{arch}.{...}.
    v17 uses lstm_hidden/lstm_layers — remap to src3's hidden/n_layers for
    the LSTM feature encoder.
    """
    ov: list[str] = []
    if "lr" in hp:
        ov.append(f"training.lr={hp['lr']}")
    if "weight_decay" in hp:
        ov.append(f"training.weight_decay={hp['weight_decay']}")
    if "_patience" in hp:
        ov.append(f"training.patience={int(hp['_patience'])}")
    # Encoder kwargs — translate v17 keys to src3 schema.
    base = f"models_nn.features.{arch}"
    if arch == "mlp":
        for k in ("hidden_dim", "repr_dim", "dropout"):
            if k in hp:
                ov.append(f"{base}.{k}={hp[k]}")
    elif arch == "lstm":
        if "lstm_hidden" in hp:
            ov.append(f"{base}.hidden={hp['lstm_hidden']}")
        if "lstm_layers" in hp:
            ov.append(f"{base}.n_layers={hp['lstm_layers']}")
        for k in ("repr_dim", "dropout"):
            if k in hp:
                ov.append(f"{base}.{k}={hp[k]}")
    return ov


def _build_cmd(task: str, arch: str, w: float, seeds: list[int],
               epochs: int, batch_size: int, run_dir: Path) -> list[str]:
    overrides = _hp_to_overrides(arch, _v17_best_hp(task, arch, w))
    overrides += [
        "soft_targets.default_modes.phase=soft",
        "soft_targets.default_modes.reps=soft_overlap",
    ]
    cmd = [
        sys.executable, "-u", "-m", "src3.pipeline.train",
        "--variant", "features",
        "--arch", arch,
        "--tasks", task,
        "--labeled-root", str(LABELED_ROOT),
        "--exclude-recordings", *EXCLUDE,
        "--window-s", str(w),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seeds", *(str(s) for s in seeds),
        "--n-trials", "0",                 # use v17 HPs as-is
        "--out", str(run_dir),
        "cv.scheme=loso",
        f"paths.splits_csv={SPLITS_LOSO}",
        *overrides,
    ]
    return cmd


def _job_done(run_dir: Path) -> bool:
    """Treat presence of summary.json as 'job done' (resume-friendly)."""
    return (run_dir / "summary.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-jobs", type=int, default=4,
                    help="Concurrent training jobs on the same GPU.")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    ap.add_argument("--tag", default="v17rep")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root or ROOT / "runs" / f"src3_{args.tag}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / f"src3_{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "status.json"

    # Build the full job list. Verify every v17 HP file exists up front so
    # we don't crash 3 hours in.
    jobs: list[dict] = []
    missing: list[str] = []
    for task in TASKS:
        for arch in ARCHS:
            for w in WINDOWS:
                slug = f"{task}-{arch}-w{_wlabel(w)}"
                hp_path = ROOT / "runs" / f"v17_fst-{task}-{arch}-w{_wlabel(w)}/best_hps.json"
                if not hp_path.exists():
                    missing.append(str(hp_path))
                    continue
                run_dir = out_root / slug
                jobs.append({
                    "slug": slug, "task": task, "arch": arch, "window": w,
                    "run_dir": run_dir,
                })
    if missing:
        print(f"[err] {len(missing)} missing v17 HP files. First few:")
        for p in missing[:5]:
            print(f"  {p}")
        return 1

    print(f"[runner] {len(jobs)} jobs in matrix; max-jobs={args.max_jobs}")
    print(f"[runner] out-root: {out_root}")
    print(f"[runner] log-dir : {log_dir}")
    print(f"[runner] excludes: {EXCLUDE}")
    print(f"[runner] seeds   : {args.seeds}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}")

    pending = list(jobs)
    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "running": [{"slug": s, "pid": r["proc"].pid,
                          "elapsed_s": int(time.time() - r["t0"]),
                          "log": str(r["log"])}
                         for s, r in running.items()],
            "pending": [j["slug"] for j in pending],
            "finished": finished,
        }, indent=2))

    t_start = time.time()
    while pending or running:
        # Spawn while under cap and there's pending work.
        while pending and len(running) < args.max_jobs:
            job = pending.pop(0)
            if _job_done(job["run_dir"]):
                print(f"[runner][skip] {job['slug']} (already done)")
                finished.append({"slug": job["slug"], "status": "resumed", "rc": 0})
                continue
            if job["run_dir"].exists():
                shutil.rmtree(job["run_dir"], ignore_errors=True)
            cmd = _build_cmd(job["task"], job["arch"], job["window"],
                              args.seeds, args.epochs, args.batch_size,
                              job["run_dir"])
            log = log_dir / f"{job['slug']}.log"
            fh = open(log, "w", encoding="utf-8", buffering=1)
            fh.write(f"# {job['slug']}\n# cmd: {' '.join(str(c) for c in cmd)}\n"
                     f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     cwd=str(ROOT))
            running[job["slug"]] = {"proc": proc, "fh": fh, "log": log,
                                     "t0": time.time(), "job": job}
            print(f"[runner][spawn] {job['slug']} (pid={proc.pid}, log={log.name})")
            write_status()

        # Reap completed.
        done = []
        for slug, rec in running.items():
            rc = rec["proc"].poll()
            if rc is not None:
                rec["fh"].close()
                elapsed = int(time.time() - rec["t0"])
                status = "ok" if rc == 0 else f"rc={rc}"
                print(f"[runner][done ] {slug} {status} ({elapsed}s)")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": str(rec["log"])})
                done.append(slug)
        for s in done:
            running.pop(s)
        if done:
            write_status()
        time.sleep(2)

    total = int(time.time() - t_start)
    print(f"\n[runner] done — {len(finished)} jobs in {total // 60}m {total % 60}s")
    print(f"[runner] summary: {status_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
