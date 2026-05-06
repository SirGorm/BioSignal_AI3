"""Wait for v7 to finish, then launch v8 with 400 Optuna trials.

Chain:
  Poll  logs/optuna_v7_<ts>/status.json  every 60 s
  When  all 9 steps in `finished` and no `running`/`pending`
   →    copy runs/optuna_clean_v7-* → runs/optuna_clean_v8-*  (preserve optuna.db, drop phase2)
   →    launch  scripts/run_optuna_v5.py --tag v8 --n-trials 400 ...

Background-friendly: no interactive output, writes its own log.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# v7 run-dir names that need migration to v8
MIGRATE = [
    "optuna_clean_v7-features-mlp",
    "optuna_clean_v7-features-lstm",
    "optuna_clean_v7-raw-cnn1d_raw",
    "optuna_clean_v7-raw-lstm_raw",
    "optuna_clean_v7-raw-cnn_lstm_raw",
    "optuna_clean_v7-raw-tcn_raw",
    "optuna_clean_v7-fatigue-raw-tcn",
    "optuna_clean_v7-fatigue-raw-lstm",
]

# v8 launch config
V8_CMD = [
    sys.executable, "scripts/run_optuna_v5.py",
    "--n-trials", "400",
    "--phase1-epochs", "50",
    "--phase2-epochs", "300",
    "--patience", "20",
    "--max-gpu-jobs", "4",
    "--tag", "v8",
]

POLL_INTERVAL = 60  # seconds


def find_v7_status() -> Path | None:
    """Return path to the most recent v7 status.json, or None."""
    candidates = sorted((ROOT / "logs").glob("optuna_v7_*/status.json"))
    return candidates[-1] if candidates else None


def v7_done(status_path: Path) -> bool:
    """v7 is done when there are 0 running, 0 pending, and >= 9 finished."""
    try:
        d = json.loads(status_path.read_text())
    except Exception:
        return False
    if d.get("running") or d.get("pending"):
        return False
    fin = d.get("finished", [])
    # 9 steps total: 6 multi-task NN + 2 fatigue-only NN + 1 RF
    return len(fin) >= 9


def migrate_v7_to_v8(log_path: Path) -> None:
    """Copy v7 dirs to v8, preserving optuna.db, dropping phase2/."""
    for v7_name in MIGRATE:
        src = RUNS / v7_name
        dst = RUNS / v7_name.replace("v7", "v8")
        if not src.exists():
            log_path.write_text(log_path.read_text() + f"  SKIP {v7_name} (no source)\n")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        p2 = dst / "phase2"
        if p2.exists():
            shutil.rmtree(p2)
        msg = f"  migrated {dst.name} (phase2 dropped)\n"
        log_path.write_text(log_path.read_text() + msg)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"chain_v7_v8_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"# chain_v7_to_v8 started {datetime.now().isoformat()}\n\n")

    print(f"Chain log: {log_path}")
    print(f"Polling every {POLL_INTERVAL} s for v7 completion...")

    # Wait for v7 status to appear
    status = find_v7_status()
    while status is None:
        time.sleep(POLL_INTERVAL)
        status = find_v7_status()
    log_path.write_text(log_path.read_text() + f"Found v7 status: {status}\n")

    # Poll until done
    last_log_t = time.time()
    while not v7_done(status):
        time.sleep(POLL_INTERVAL)
        # Heartbeat every 10 min
        if time.time() - last_log_t > 600:
            try:
                d = json.loads(status.read_text())
                running = [r["slug"] for r in d.get("running", [])]
                pending = d.get("pending", [])
                finished = len(d.get("finished", []))
                msg = (f"[{datetime.now().strftime('%H:%M:%S')}] "
                       f"running={running} pending={len(pending)} finished={finished}\n")
                log_path.write_text(log_path.read_text() + msg)
                last_log_t = time.time()
            except Exception:
                pass

    log_path.write_text(log_path.read_text()
                         + f"\n=== v7 DONE at {datetime.now().isoformat()} ===\n")

    # Migrate
    log_path.write_text(log_path.read_text() + "Migrating v7 → v8...\n")
    migrate_v7_to_v8(log_path)

    # Launch v8
    log_path.write_text(log_path.read_text()
                         + f"\nLaunching v8: {' '.join(V8_CMD)}\n")
    proc = subprocess.Popen(V8_CMD, cwd=str(ROOT))
    log_path.write_text(log_path.read_text() + f"v8 launched, pid={proc.pid}\n")

    # Wait for v8 to finish (just keeps this chain process alive for tracking)
    rc = proc.wait()
    log_path.write_text(log_path.read_text()
                         + f"\n=== v8 DONE rc={rc} at {datetime.now().isoformat()} ===\n")


if __name__ == "__main__":
    main()
