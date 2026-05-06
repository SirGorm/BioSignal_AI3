"""V13 single-task PHASE only, but with soft phase target (KL-div).

Mirror of the four phase-only configs in run_v13_singletask.py
(feat-mlp/feat-lstm @ 1s/2s), but trained with --phase-mode soft so the
result is apples-to-apples with v13soft. Same dataset, splits, exclusion
list, 50 Optuna trials, phase1 + phase2 with 3 seeds.

Output dirs use the prefix `optuna_clean_v13singlesoft-` to keep them
separate from the hard baseline runs.
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
SPLITS = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

# (task, slug-suffix, arch, variant, window_s)  -- same 4 phase-only configs
JOBS_DEF = [
    ('phase', 'feat-mlp',  'mlp',  'features', 1.0),
    ('phase', 'feat-mlp',  'mlp',  'features', 2.0),
    ('phase', 'feat-lstm', 'lstm', 'features', 1.0),
    ('phase', 'feat-lstm', 'lstm', 'features', 2.0),
]


def _wlabel(w: float) -> str:
    return f"{w:g}s".replace('.', '_')


def step_done(d: Path) -> bool:
    return (d / "phase2").exists() and \
           next(iter((d / "phase2").rglob("cv_summary.json")), None) is not None


def make_jobs(n_trials, p1_epochs, p2_epochs, patience, tag):
    seeds = [42, 1337, 7]
    common = [
        "--n-trials",     str(n_trials),
        "--phase1-epochs", str(p1_epochs),
        "--phase2-epochs", str(p2_epochs),
        "--patience",      str(patience),
        "--phase2-seeds", *[str(s) for s in seeds],
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits",       str(SPLITS),
        "--reps-mode",    "soft_overlap",
        "--phase-mode",   "soft",      # <-- the only difference vs v13single
        "--num-workers",  "0",
    ]
    jobs = []
    for task, slug_suffix, arch, variant, ws in JOBS_DEF:
        wl = _wlabel(ws)
        slug = f"{task}-only-w{wl}-{slug_suffix}"
        run_dir = ROOT / "runs" / f"optuna_clean_{tag}-{slug}"
        cmd = [sys.executable, "scripts/train_optuna.py",
               "--arch", arch, "--variant", variant,
               "--run-dir", str(run_dir),
               "--tasks", task,
               "--window-s", str(ws),
               *common]
        jobs.append({"slug": slug, "cmd": cmd, "run_dir": run_dir})
    return jobs


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials",      type=int, default=50)
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=300)
    ap.add_argument("--patience",      type=int, default=20)
    ap.add_argument("--max-gpu-jobs",  type=int, default=4)
    ap.add_argument("--tag", default="v13singlesoft")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    jobs = make_jobs(args.n_trials, args.phase1_epochs, args.phase2_epochs,
                      args.patience, args.tag)
    pending, skipped = [], []
    for j in jobs:
        (skipped if step_done(j["run_dir"]) else pending).append(
            j["slug"] if step_done(j["run_dir"]) else j)
    pending = [j for j in pending if isinstance(j, dict)]

    print(f"=== {args.tag.upper()}: phase-only, soft KL-div phase target ===")
    print(f"  configs: {len(jobs)}  pending={len(pending)}  cached={len(skipped)}")
    print(f"  n_trials={args.n_trials}  p1_epochs={args.phase1_epochs}  "
          f"p2_epochs={args.phase2_epochs}  patience={args.patience}")
    print(f"  max GPU jobs: {args.max_gpu_jobs}")
    print(f"  logs: {log_dir}\n")

    running, finished = {}, []

    def write_status():
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "running": [{"slug": s, "pid": r["pid"],
                          "elapsed_s": int(time.time() - r["started"]),
                          "log": str(r["log"])}
                         for s, r in running.items()],
            "pending": [j["slug"] for j in pending],
            "finished": finished,
            "skipped_cached": skipped,
        }, indent=2))

    while pending or running:
        while pending and len(running) < args.max_gpu_jobs:
            j = pending.pop(0)
            log = log_dir / f"{j['slug']}.log"
            fh = open(log, "w", encoding="utf-8", buffering=1)
            fh.write(f"# {j['slug']}\n# cmd: "
                      f"{' '.join(str(x) for x in j['cmd'])}\n"
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
                ok = (rc == 0 or
                      (rc == 3221226505 and step_done(Path([
                          j['run_dir'] for j in jobs if j['slug'] == slug
                      ][0]))))
                status = "ok" if ok else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s "
                      f"status={status}")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": str(r["log"])})
                done.append(slug)
        for s in done:
            running.pop(s)
        write_status()

    print(f"\n=== {args.tag.upper()} DONE ===")
    ok = sum(1 for f in finished if f['status'] == 'ok')
    fail = [f for f in finished if f['status'] != 'ok']
    print(f"  ok: {ok}  failed: {len(fail)}  cached: {len(skipped)}")
    for f in fail:
        print(f"    FAIL {f['slug']}  rc={f['rc']}  log={f['log']}")
    print(f"\nFull status: {status_path}")


if __name__ == '__main__':
    main()
