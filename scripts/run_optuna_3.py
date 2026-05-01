"""Sequentially run Optuna for: feat-mlp, raw-tcn, raw-cnn_lstm.

Logs to logs/optuna_3_<timestamp>.log and logs/optuna_3_status_<ts>.json.
Each step retried once on failure, then skipped.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = LOGS_DIR / f"optuna_3_{ts}.log"
STATUS_PATH = LOGS_DIR / f"optuna_3_status_{ts}.json"


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def step_already_done(run_dir: Path) -> bool:
    """Resume detection: a step is done if its phase2 has cv_summary.json."""
    cv = next(iter((run_dir / 'phase2').rglob('cv_summary.json')), None) \
            if run_dir.exists() else None
    return cv is not None


def run_step(name: str, cmd: list, run_dir: Path = None) -> dict:
    if run_dir is not None and step_already_done(run_dir):
        log(f"=== STEP: {name} — RESUME: already complete at {run_dir} ===")
        return {'name': name, 'status': 'resumed', 'attempts': 0,
                 'elapsed_s': 0, 'run_dir': str(run_dir)}
    log(f"=== STEP: {name} ===")
    log(f"  cmd: {' '.join(cmd)}")
    started = time.time()
    for attempt in (1, 2):
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    log(f"  [out] {line}")
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    log(f"  [err] {line}")
            elapsed = time.time() - started
            if proc.returncode == 0:
                log(f"  ok ({elapsed:.0f}s, attempt {attempt})")
                return {'name': name, 'status': 'ok',
                         'attempts': attempt, 'elapsed_s': elapsed}
            log(f"  FAILED rc={proc.returncode} on attempt {attempt}")
            if attempt == 2:
                return {'name': name, 'status': 'failed',
                         'attempts': 2, 'elapsed_s': elapsed,
                         'returncode': proc.returncode}
            time.sleep(5)
        except Exception as e:
            log(f"  EXCEPTION on attempt {attempt}: {e}")
            if attempt == 2:
                return {'name': name, 'status': 'crashed',
                         'error': str(e),
                         'elapsed_s': time.time() - started}
            time.sleep(5)


COMMON = [
    '--phase1-epochs', '30', '--phase2-epochs', '50',
    '--exclude-recordings', 'recording_003',
    '--phase-whitelist', 'configs/phase_quality_sets.csv',
]
# Modality subset: EMG, ACC, PPG, TEMP (drop ECG, EDA)
FEATURE_PREFIXES = ['emg_', 'acc_', 'ppg_', 'temp_']
RUNS = Path('runs')


def step_spec(slug: str, arch: str, variant: str, n_trials: int, n_workers: int):
    """Returns (name, cmd, run_dir) for one step. Stable run_dir enables resume."""
    run_dir = RUNS / f"optuna-{variant}-{arch}"
    cmd = [
        sys.executable, 'scripts/train_optuna.py',
        '--arch', arch, '--variant', variant,
        '--n-trials', str(n_trials),
        '--num-workers', str(n_workers),
        '--run-dir', str(run_dir),
        *COMMON,
    ]
    if variant == 'features':
        cmd += ['--feature-prefixes', *FEATURE_PREFIXES]
    return slug, cmd, run_dir


steps = [
    step_spec("feat-mlp",       "mlp",          "features", 20, 2),
    step_spec("feat-lstm",      "lstm",         "features", 10, 2),
    step_spec("raw-cnn1d",      "cnn1d_raw",    "raw",      25, 8),
    step_spec("raw-lstm",       "lstm_raw",     "raw",      10, 8),
    step_spec("raw-cnn_lstm",   "cnn_lstm_raw", "raw",      15, 8),
    step_spec("raw-tcn",        "tcn_raw",      "raw",      25, 8),
]

results = []
log(f"=== OPTUNA SEQUENTIAL RUN STARTED ===")
log(f"Steps: {[s[0] for s in steps]}")
STATUS_PATH.write_text(json.dumps({'started': ts, 'steps_planned': [s[0] for s in steps]}, indent=2))

t0_total = time.time()
for name, cmd, run_dir in steps:
    res = run_step(name, cmd, run_dir=run_dir)
    results.append(res)
    STATUS_PATH.write_text(json.dumps({
        'started': ts,
        'last_updated': datetime.now().isoformat(timespec='seconds'),
        'results': results,
        'total_elapsed_s': time.time() - t0_total,
    }, indent=2))

log(f"=== ALL DONE in {(time.time() - t0_total) / 3600:.1f}h ===")
for r in results:
    log(f"  {r['name']}: {r['status']} ({r.get('elapsed_s', 0)/60:.1f} min)")
