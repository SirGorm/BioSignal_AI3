"""V12: full Optuna search (75 trials) + phase 2 with the new
equal-weight ``score_summary``. Runs 8 archs × 3 windows (1, 2, 5 s) on
active-only data (the v9 default; rest-window experiment lives in v10/v11).

Each job calls scripts/train_optuna.py end-to-end (phase 1 + phase 2). The
new score normalises each task into [0,1] error space before averaging, so
fatigue MAE no longer dominates over F1.
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

WINDOWS = [1.0, 2.0, 5.0]

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


def _wlabel(window_s: float) -> str:
    return f"{window_s:g}s".replace('.', '_')


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def make_jobs(n_trials, p1_epochs, p2_epochs, patience, tag,
              arch_filter=None, window_filter=None):
    seeds = [42, 1337, 7]
    common = [
        "--n-trials", str(n_trials),
        "--phase1-epochs", str(p1_epochs),
        "--phase2-epochs", str(p2_epochs),
        "--patience", str(patience),
        "--phase2-seeds", *[str(s) for s in seeds],
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits", str(SPLITS),
        "--reps-mode", "soft_overlap",
        "--num-workers", "0",
    ]
    archs = [(s, a, v, ['exercise','phase','fatigue','reps'])
             for s, a, v in MULTI_ARCHS]
    archs += [(s, a, v, ['fatigue']) for s, a, v in FATIGUE_ARCHS]
    if arch_filter:
        archs = [t for t in archs if t[0] in arch_filter]
    windows = WINDOWS if window_filter is None else window_filter

    jobs = []
    for w in windows:
        wl = _wlabel(w)
        for slug, arch, variant, tasks in archs:
            run_dir = ROOT / "runs" / f"optuna_clean_{tag}-w{wl}-{slug}"
            cmd = [sys.executable, "scripts/train_optuna.py",
                   "--arch", arch, "--variant", variant,
                   "--run-dir", str(run_dir),
                   "--tasks", *tasks,
                   "--window-s", str(w),
                   *common]
            jobs.append({"slug": f"w{wl}-{slug}", "cmd": cmd,
                         "run_dir": run_dir})
    return jobs


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=75)
    ap.add_argument("--phase1-epochs", type=int, default=30)
    ap.add_argument("--phase2-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--max-gpu-jobs", type=int, default=4)
    ap.add_argument("--tag", default="v12eqw")
    ap.add_argument("--archs", nargs="*", default=None,
                    help="Restrict to these arch slugs (e.g. multi-raw-tcn). "
                         "Default: all 8 archs.")
    ap.add_argument("--windows", nargs="*", type=float, default=None,
                    help="Restrict to these window lengths in seconds "
                         "(e.g. 1 2 5). Default: 1 2 5.")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    jobs = make_jobs(args.n_trials, args.phase1_epochs, args.phase2_epochs,
                      args.patience, args.tag,
                      arch_filter=args.archs, window_filter=args.windows)
    pending = []
    skipped = []
    for j in jobs:
        if step_done(j["run_dir"]):
            skipped.append(j["slug"])
        else:
            pending.append(j)

    print(f"=== V12 ({args.tag.upper()}): full Optuna w/ equal-weight score ===")
    print(f"Windows: {WINDOWS}  archs: {len(MULTI_ARCHS)+len(FATIGUE_ARCHS)}  "
          f"total jobs: {len(jobs)}")
    print(f"  pending: {len(pending)}  cached: {len(skipped)}")
    print(f"  n_trials: {args.n_trials}  p1_epochs: {args.phase1_epochs}  "
          f"p2_epochs: {args.phase2_epochs}  patience: {args.patience}")
    print(f"  max GPU jobs: {args.max_gpu_jobs}")
    print(f"Logs: {log_dir}\n")

    running: dict[str, dict] = {}
    finished: list[dict] = []

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
                # rc=0xC0000409 (3221226505) is the known Windows torch+CUDA
                # cleanup bug; cv_summary.json is still saved successfully.
                ok = (rc == 0
                      or (rc == 3221226505
                          and step_done(Path([
                              j['run_dir'] for j in (jobs)
                              if j['slug'] == slug
                          ][0]))))
                status = "ok" if ok else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s  status={status}")
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
        print(f"    FAIL  {f['slug']}  rc={f['rc']}  log={f['log']}")
    print(f"\nFull status: {status_path}")


if __name__ == '__main__':
    main()
