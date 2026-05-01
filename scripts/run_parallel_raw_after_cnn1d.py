"""Watcher + parallel dispatcher.

1. Poll until raw-cnn1d phase2 cv_summary.json exists.
2. Kill the sequential orchestrator (so it doesn't start raw-lstm).
3. Launch raw-lstm + raw-cnn_lstm + raw-tcn in parallel as 3 background
   subprocesses, each logging to its own file.

Each parallel arch uses num_workers=3 (down from 8) to keep CPU within
20-core budget when running 3 simultaneously: 3 procs x 3 workers = 9
plus 3 mains = 12, well under the limit.
"""

from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOGS = Path('logs')
LOGS.mkdir(exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
WATCH_LOG = LOGS / f"parallel_dispatcher_{ts}.log"

# Sequential orchestrator PID (passed via env or auto-detect)
SEQ_ORCHESTRATOR_PID = int(os.environ.get('SEQ_ORCH_PID', '15584'))

# Phase 2 cv_summary indicating raw-cnn1d is done
CNN1D_DONE_MARKER = Path('runs/optuna-raw-cnn1d_raw/phase2/cnn1d_raw/cv_summary.json')


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(WATCH_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def wait_for_cnn1d():
    log(f"Waiting for {CNN1D_DONE_MARKER} ...")
    while not CNN1D_DONE_MARKER.exists():
        time.sleep(120)  # 2 min polling
        log(f"  still waiting (cnn1d not yet done)")
    log(f"raw-cnn1d phase 2 cv_summary detected. Proceeding.")


def kill_sequential_orchestrator():
    """Kill the sequential orchestrator so it doesn't launch raw-lstm.
    Uses Windows taskkill with /T to take down the process tree."""
    log(f"Killing sequential orchestrator PID {SEQ_ORCHESTRATOR_PID} ...")
    try:
        result = subprocess.run(
            ['taskkill', '/F', '/PID', str(SEQ_ORCHESTRATOR_PID), '/T'],
            capture_output=True, text=True
        )
        log(f"  taskkill output: {result.stdout.strip()}")
        if result.returncode != 0:
            log(f"  taskkill stderr: {result.stderr.strip()}")
    except Exception as e:
        log(f"  EXCEPTION: {e}")


COMMON = [
    '--phase1-epochs', '30', '--phase2-epochs', '50',
    '--exclude-recordings', 'recording_003',
    '--phase-whitelist', 'configs/phase_quality_sets.csv',
    '--num-workers', '3',
]
PARALLEL_STEPS = [
    ('raw-lstm',     'lstm_raw',     '10'),
    ('raw-cnn_lstm', 'cnn_lstm_raw', '15'),
    ('raw-tcn',      'tcn_raw',      '25'),
]


def launch_parallel():
    log(f"Launching {len(PARALLEL_STEPS)} parallel raw-arch jobs...")
    procs = []
    for name, arch, n_trials in PARALLEL_STEPS:
        run_dir = Path('runs') / f"optuna-raw-{arch}"
        per_log = LOGS / f"parallel_{name}_{ts}.log"
        cmd = [
            sys.executable, 'scripts/train_optuna.py',
            '--arch', arch, '--variant', 'raw',
            '--n-trials', n_trials,
            '--run-dir', str(run_dir),
            *COMMON,
        ]
        log(f"  spawn {name}: {' '.join(cmd)}")
        log(f"    -> log: {per_log}")
        f = open(per_log, 'a', encoding='utf-8', buffering=1)
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                              if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0)
        procs.append((name, p, f))
        log(f"    -> PID {p.pid}")
        time.sleep(15)  # stagger startup so dataset loading doesn't collide
    return procs


def wait_for_all(procs):
    log(f"Waiting for all {len(procs)} parallel jobs to complete...")
    while procs:
        for i, (name, p, f) in list(enumerate(procs)):
            ret = p.poll()
            if ret is not None:
                log(f"  {name} (PID {p.pid}) finished with rc={ret}")
                f.close()
                procs.pop(i)
        time.sleep(60)
    log(f"All parallel jobs done.")


def main():
    log(f"=== PARALLEL DISPATCHER STARTED ===")
    log(f"Will: 1) wait for {CNN1D_DONE_MARKER} | "
        f"2) kill PID {SEQ_ORCHESTRATOR_PID} | "
        f"3) launch 3 parallel raw arch jobs")

    wait_for_cnn1d()
    kill_sequential_orchestrator()
    time.sleep(10)
    procs = launch_parallel()
    wait_for_all(procs)
    log(f"=== PARALLEL DISPATCHER COMPLETE ===")


if __name__ == '__main__':
    main()
