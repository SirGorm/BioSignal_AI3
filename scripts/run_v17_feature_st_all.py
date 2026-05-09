"""Run Feature single-task training for all (arch, window, task) combinations,
mirroring how v20 raw_st covered the full grid.

For each of the 24 combinations
    arch    in {mlp, lstm}
    window  in {1.0, 2.0, 5.0}
    task    in {exercise, phase, fatigue, reps}
this launches scripts/train_phase2_only.py using the matching v17 multi-task
Optuna best_hps as the HP source:
    --src-run-dir runs/v17multi-{arch}-w{w}s
    --out-run-dir runs/v17_fst-{task}-{arch}-w{w}s
    --tasks {task}

Phase-2 settings match the v17 campaign: 150 epochs, patience 10, 3 seeds, LOSO
10-fold. Resumes by skipping any run that already has phase2/.../cv_summary.json.

Usage:
    python scripts/run_v17_feature_st_all.py --max-gpu-jobs 3
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

ARCHS = ["mlp", "lstm"]
WINDOWS = [1.0, 2.0, 5.0]
TASKS = ["exercise", "phase", "fatigue", "reps"]


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def make_jobs(phase2_epochs: int, patience: int, splits: str):
    seeds = [42, 1337, 7]
    jobs = []
    for arch in ARCHS:
        for w in WINDOWS:
            ws = str(int(w))
            src_run = ROOT / "runs" / f"v17multi-{arch}-w{ws}s"
            for task in TASKS:
                slug = f"v17_fst-{task}-{arch}-w{ws}s"
                out_run = ROOT / "runs" / slug
                cmd = [
                    sys.executable, "scripts/train_phase2_only.py",
                    "--arch", arch,
                    "--variant", "features",
                    "--src-run-dir", str(src_run),
                    "--out-run-dir", str(out_run),
                    "--tasks", task,
                    "--window-s", str(w),
                    "--phase2-epochs", str(phase2_epochs),
                    "--patience", str(patience),
                    "--phase2-seeds", *[str(s) for s in seeds],
                    "--splits", splits,
                    "--num-workers", "0",
                ]
                jobs.append({"slug": slug, "cmd": cmd, "run_dir": out_run,
                              "src_run": src_run})
    return jobs


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gpu-jobs", type=int, default=3,
                    help="Concurrent jobs (Phase 2 only is GPU-light).")
    ap.add_argument("--phase2-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--splits", default="configs/splits_loso.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print jobs and exit without launching.")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"v17_fst_all_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    jobs = make_jobs(args.phase2_epochs, args.patience, args.splits)

    pending, skipped, missing_src = [], [], []
    for j in jobs:
        if not (j["src_run"] / "best_hps.json").exists():
            missing_src.append(j["slug"])
            continue
        if step_done(j["run_dir"]):
            skipped.append(j["slug"])
        else:
            pending.append(j)

    print(f"=== v17 FEATURE ST — full grid ({len(jobs)} jobs) ===")
    print(f"  pending: {len(pending)}  cached: {len(skipped)}  "
          f"missing-src: {len(missing_src)}")
    print(f"  phase2_epochs={args.phase2_epochs}  patience={args.patience}  "
          f"max_parallel={args.max_gpu_jobs}")
    for j in pending:
        print(f"  PENDING  {j['slug']}")
    for s in skipped:
        print(f"  CACHED   {s}")
    for s in missing_src:
        print(f"  NO-HPS   {s}")
    print(f"Logs: {log_dir}\n")

    if args.dry_run:
        print("Dry-run: exiting without launching.")
        return

    running = {}
    finished = []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "running": [{"slug": s, "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "log": str(r["log"])}
                         for s, r in running.items()],
            "pending": [j["slug"] for j in pending],
            "finished": finished,
            "skipped_cached": skipped,
            "missing_src": missing_src,
        }, indent=2))

    while pending or running:
        while pending and len(running) < args.max_gpu_jobs:
            j = pending.pop(0)
            log = log_dir / f"{j['slug']}.log"
            fh = open(log, "w", encoding="utf-8", buffering=1)
            fh.write(f"# {j['slug']}\n# cmd: {' '.join(str(x) for x in j['cmd'])}\n"
                     f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            proc = subprocess.Popen(j["cmd"], stdout=fh,
                                     stderr=subprocess.STDOUT, cwd=str(ROOT))
            running[j["slug"]] = {"pid": proc.pid, "proc": proc, "fh": fh,
                                    "log": log, "started": time.time()}
            print(f"[{j['slug']}] LAUNCHED pid={proc.pid}")
        write_status()
        time.sleep(30)
        done = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is not None:
                r["fh"].flush(); r["fh"].close()
                elapsed = int(time.time() - r["started"])
                run_dir = next(j["run_dir"] for j in jobs if j["slug"] == slug)
                ok = (rc == 0
                      or (rc in (3221226505, -1073741510)
                          and step_done(run_dir)))
                status = "ok" if ok else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s status={status}")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": str(r["log"])})
                done.append(slug)
        for s in done:
            running.pop(s)
        write_status()

    print("\n=== v17 FEATURE ST — DONE ===")
    ok = sum(1 for f in finished if f["status"] == "ok")
    fail = [f for f in finished if f["status"] != "ok"]
    print(f"  ok: {ok}  failed: {len(fail)}  cached: {len(skipped)}  "
          f"missing-src: {len(missing_src)}")
    for f in fail:
        print(f"  FAIL  {f['slug']}  log={f['log']}")


if __name__ == "__main__":
    main()
