"""V14: modality ablation for the top-3 v12 multi-task architectures.

For each of {feat-mlp, feat-lstm, raw-tcn} @ 1 s, runs:
  - Leave-one-out (LOO): keep 3 of 4 modalities, drop the named one (4 jobs)
  - Leave-one-in (LOI):  keep ONLY the named modality (4 jobs)

= 3 archs × 8 ablations = 24 phase-2-only refits with v12 best HPs.

Modalities: EMG, acc, PPG-green, temp. (ECG and EDA are excluded upstream.)

For features-variant archs we filter with ``--feature-prefixes``. For raw
archs we filter with ``--raw-channels``. v12 best HPs are loaded from
``--src-run-dir`` and reused unchanged.
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

# Modality definitions
FEATURE_PREFIXES = {
    'emg':  ['emg_'],
    'acc':  ['acc_'],
    'ppg':  ['ppg_'],
    'temp': ['temp_'],
}
RAW_CHANNEL_OF = {
    'emg':  'emg',
    'acc':  'acc_mag',
    'ppg':  'ppg_green',
    'temp': 'temp',
}
ALL_MODALITIES = ['emg', 'acc', 'ppg', 'temp']

# (slug, arch, variant, window_s, src v12 dir)
BASE_ARCHS = [
    ('multi-feat-mlp',  'mlp',     'features',
     'optuna_clean_v12eqw-w1s-multi-feat-mlp'),
    ('multi-feat-lstm', 'lstm',    'features',
     'optuna_clean_v12eqw-w1s-multi-feat-lstm'),
    ('multi-raw-tcn',   'tcn_raw', 'raw',
     'optuna_clean_v12eqw-w1s-multi-raw-tcn'),
]


def step_done(run_dir: Path) -> bool:
    return (run_dir / "phase2").exists() and \
           next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) is not None


def make_jobs():
    """Yield 24 ablation jobs — 3 archs × (4 LOO + 4 LOI)."""
    jobs = []
    for arch_slug, arch, variant, src_dir in BASE_ARCHS:
        for kind in ('loo', 'loi'):  # leave-one-out / leave-one-in
            for mod in ALL_MODALITIES:
                if kind == 'loo':
                    kept = [m for m in ALL_MODALITIES if m != mod]
                else:
                    kept = [mod]
                slug = f"{arch_slug}-{kind}-{mod}"
                run_dir = ROOT / "runs" / f"optuna_clean_v14ablate-{slug}"
                # Build CLI for either feature- or raw-variant ablation
                ablation_args = []
                if variant == 'features':
                    prefixes = sum((FEATURE_PREFIXES[m] for m in kept), [])
                    ablation_args = ["--feature-prefixes", *prefixes]
                else:
                    chans = [RAW_CHANNEL_OF[m] for m in kept]
                    ablation_args = ["--raw-channels", *chans]
                cmd = [
                    sys.executable, "scripts/train_phase2_only.py",
                    "--arch", arch, "--variant", variant,
                    "--src-run-dir", str(ROOT / "runs" / src_dir),
                    "--out-run-dir", str(run_dir),
                    "--phase2-epochs", "300", "--patience", "20",
                    "--phase2-seeds", "42", "1337", "7",
                    "--tasks", "exercise", "phase", "fatigue", "reps",
                    "--reps-mode", "soft_overlap",
                    "--window-s", "1.0",
                    "--num-workers", "0",
                    "--exclude-recordings", *EXCLUDE,
                    "--labeled-root", str(LABELED_ROOT),
                    "--splits", str(SPLITS),
                    *ablation_args,
                ]
                jobs.append({"slug": slug, "cmd": cmd, "run_dir": run_dir,
                             "kept_modalities": kept})
    return jobs


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gpu-jobs", type=int, default=4)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"v14ablate_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    jobs = make_jobs()
    pending, skipped = [], []
    for j in jobs:
        (skipped if step_done(j["run_dir"]) else pending).append(j)
    skipped = [j['slug'] for j in skipped]

    print(f"=== V14 modality ablation ===")
    print(f"Total: {len(jobs)} (3 archs × 8 ablations)")
    print(f"Pending: {len(pending)}  cached: {len(skipped)}")
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
            fh.write(f"# {j['slug']}\n# kept: {j['kept_modalities']}\n"
                     f"# cmd: {' '.join(str(x) for x in j['cmd'])}\n"
                     f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            proc = subprocess.Popen(j["cmd"], stdout=fh,
                                     stderr=subprocess.STDOUT, cwd=str(ROOT))
            running[j["slug"]] = {"pid": proc.pid, "proc": proc, "fh": fh,
                                    "log": log, "started": time.time()}
            print(f"[{j['slug']}] LAUNCHED pid={proc.pid}  kept={j['kept_modalities']}")
        write_status()

        time.sleep(30)

        done = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is not None:
                r["fh"].flush(); r["fh"].close()
                elapsed = int(time.time() - r["started"])
                ok = (rc == 0 or
                      (rc == 3221226505 and step_done([j['run_dir']
                       for j in jobs if j['slug'] == slug][0])))
                status = "ok" if ok else f"failed_rc{rc}"
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s  status={status}")
                finished.append({"slug": slug, "status": status, "rc": rc,
                                  "elapsed_s": elapsed, "log": str(r["log"])})
                done.append(slug)
        for s in done:
            running.pop(s)
        write_status()

    print(f"\n=== V14 ablation DONE ===")
    ok = sum(1 for f in finished if f['status'] == 'ok')
    fail = [f for f in finished if f['status'] != 'ok']
    print(f"  ok: {ok}  failed: {len(fail)}  cached: {len(skipped)}")
    for f in fail:
        print(f"    FAIL  {f['slug']}  rc={f['rc']}")


if __name__ == '__main__':
    main()
