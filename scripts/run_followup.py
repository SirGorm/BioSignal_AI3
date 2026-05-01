"""Wait for the in-progress optuna_3 pipeline to finish, then run two
follow-up trainings:
  - full TCN on raw signals  (3 seeds x 5 folds x 50 ep, uncertainty ON)
  - full LSTM on features    (3 seeds x 5 folds x 50 ep, uncertainty ON)

Polls the latest logs/optuna_3_status_*.json until len(results) == 4.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOGS = Path('logs')
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = LOGS / f"followup_{ts}.log"
STATUS_PATH = LOGS / f"followup_status_{ts}.json"


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def wait_for_optuna_done():
    log("Polling for optuna_3 pipeline completion...")
    expected_total = 4
    while True:
        candidates = sorted(LOGS.glob('optuna_3_status_*.json'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            log("  no optuna_3_status_*.json found yet")
            time.sleep(60)
            continue
        latest = candidates[0]
        try:
            data = json.loads(latest.read_text())
        except Exception as e:
            log(f"  status JSON not parseable yet: {e}")
            time.sleep(60)
            continue
        results = data.get('results', [])
        log(f"  {latest.name}: {len(results)}/{expected_total} steps complete")
        if len(results) >= expected_total:
            log(f"  ALL DONE: {[r['name'] + '/' + r['status'] for r in results]}")
            return latest
        time.sleep(300)  # 5 min polling cadence


RUNS = Path('runs')


def latest_complete_run(slug_pattern: str) -> Path | None:
    """Find newest runs/*_<slug_pattern> dir that contains a cv_summary.json."""
    candidates = sorted(RUNS.glob(f"*_{slug_pattern}"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if any(c.rglob('cv_summary.json')):
            return c
    return None


def run_step(name: str, cmd: list, slug_pattern: str = None) -> dict:
    if slug_pattern:
        already = latest_complete_run(slug_pattern)
        if already is not None:
            log(f"=== STEP: {name} — RESUME: already complete at {already} ===")
            return {'name': name, 'status': 'resumed', 'attempts': 0,
                     'elapsed_s': 0, 'run_dir': str(already)}
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


COMMON_FEAT = [
    '--seeds', '42', '1337', '7',
    '--epochs', '50',
    '--uncertainty-weighting',
    '--phase-whitelist', 'configs/phase_quality_sets.csv',
]
COMMON_RAW = [
    '--seeds', '42', '1337', '7',
    '--epochs', '50',
    '--n-folds', '5',
    '--num-workers', '8',
    '--uncertainty-weighting',
    '--phase-whitelist', 'configs/phase_quality_sets.csv',
    '--exclude-recordings', 'recording_003',
    '--run-slug-prefix', 'full-uncw',
]

steps = [
    ("full-tcn-raw-uncw",
        [sys.executable, 'scripts/train_raw_short.py',
          '--archs', 'tcn_raw', *COMMON_RAW],
        'full-uncw-tcn_raw'),  # slug_pattern matches train_raw_short output
    ("full-lstm-feat-uncw",
        [sys.executable, 'scripts/train_lstm.py',
          '--run-slug', 'full-lstm-features-uncw', *COMMON_FEAT],
        'full-lstm-features-uncw'),
]


def main():
    log(f"=== FOLLOWUP RUN STARTED ===")
    log(f"Will run after optuna_3 pipeline completes: "
        f"{[s[0] for s in steps]}")
    STATUS_PATH.write_text(json.dumps({
        'started': ts, 'state': 'waiting',
        'steps_planned': [s[0] for s in steps],
    }, indent=2))

    completed_status = wait_for_optuna_done()
    log(f"Optuna pipeline finished (status: {completed_status.name}). "
        f"Starting follow-up trainings.")

    results = []
    t0 = time.time()
    for name, cmd, slug in steps:
        res = run_step(name, cmd, slug_pattern=slug)
        results.append(res)
        STATUS_PATH.write_text(json.dumps({
            'started': ts,
            'last_updated': datetime.now().isoformat(timespec='seconds'),
            'optuna_status_file': str(completed_status),
            'results': results,
            'total_elapsed_s': time.time() - t0,
        }, indent=2))

    log(f"=== FOLLOWUP DONE in {(time.time() - t0) / 3600:.1f}h ===")
    for r in results:
        log(f"  {r['name']}: {r['status']} ({r.get('elapsed_s', 0)/60:.1f} min)")


if __name__ == '__main__':
    main()
