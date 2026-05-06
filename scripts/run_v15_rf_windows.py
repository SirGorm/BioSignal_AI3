"""V15: Random Forest baselines at NN window sizes (1, 2, 5 s).

train_lgbm.py decimates window_features.parquet by ``--stride`` (samples on
the 100 Hz feature grid). To match v12 NN windows with 50% overlap:

  window 1 s → hop 0.5 s → stride = 50
  window 2 s → hop 1.0 s → stride = 100
  window 5 s → hop 2.5 s → stride = 250

Fatigue + reps use ``set_features.parquet`` (per-set, window-independent),
so those metrics will be identical across the 3 runs. Exercise + phase
classification metrics will differ because the per-window training set
size and label balance change.

Sequential — RF uses all CPU cores via ``n_jobs=-1`` so parallel runs
would just thrash.
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
RF_FEATURES_DIR = ROOT / "runs" / "optuna_clean_v1" / "features"
SPLITS = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

# (window_label, stride)
WINDOWS = [('1s', 50), ('2s', 100), ('5s', 250)]


def step_done(run_dir: Path) -> bool:
    return (run_dir / "metrics.json").exists()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v15rf")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"{args.tag}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"

    print(f"=== V15 RF baselines at windows {[w for w, _ in WINDOWS]} ===")
    print(f"Logs: {log_dir}\n")

    finished = []
    for wlabel, stride in WINDOWS:
        run_dir = ROOT / "runs" / f"optuna_clean_{args.tag}-w{wlabel}"
        run_dir.mkdir(parents=True, exist_ok=True)
        slug = f"rf-w{wlabel}"
        if step_done(run_dir):
            print(f"[{slug}] SKIP — already done ({run_dir/'metrics.json'})")
            finished.append({"slug": slug, "status": "cached", "rc": 0,
                              "elapsed_s": 0})
            continue
        cmd = [
            sys.executable, "scripts/train_lgbm.py",
            "--run-dir", str(run_dir),
            "--features-dir", str(RF_FEATURES_DIR),
            "--exclude-recordings", *EXCLUDE,
            "--splits", str(SPLITS),
            "--stride", str(stride),
        ]
        log = log_dir / f"{slug}.log"
        fh = open(log, "w", encoding="utf-8", buffering=1)
        fh.write(f"# {slug}\n# stride={stride}  window={wlabel}\n"
                 f"# cmd: {' '.join(str(x) for x in cmd)}\n"
                 f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        fh.flush()
        print(f"[{slug}] starting (stride={stride}, log={log})")
        t0 = time.time()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(ROOT))
        fh.flush(); fh.close()
        elapsed = int(time.time() - t0)
        status = "ok" if rc == 0 else f"failed_rc{rc}"
        finished.append({"slug": slug, "status": status, "rc": rc,
                          "elapsed_s": elapsed, "log": str(log)})
        print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s  status={status}")
        status_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "finished": finished,
        }, indent=2))

    print(f"\n=== {args.tag.upper()} DONE ===")
    for f in finished:
        print(f"  {f['slug']}: {f['status']} ({f['elapsed_s']}s)")


if __name__ == '__main__':
    main()
