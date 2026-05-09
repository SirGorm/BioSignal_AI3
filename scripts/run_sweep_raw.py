"""Sweep over raw-input architectures (1D-CNN, TCN, CNN-LSTM).

Standalone orchestrator — designed to be moved to a different machine and run
on its own. Calls scripts/train_optuna.py as subprocess for each
(arch, mode, window) cell, then runs phase 2 with 3 seeds. All results are
appended to results/sweep_raw_results.csv (one row per arch×mode×window×seed).
Per-fold rows go to results/sweep_raw_per_fold.csv.

Workflow per (arch, mode):
  1. Optuna phase 1 on BASE window (--window-s 2.0): 100 trials, 100 epochs
  2. Optuna phase 1 on OTHER windows (1, 5): 20 trials each, seeded from
     base via --seed-hps-from
  3. Phase 2 refit per window with 3 seeds × 5 GKF folds × 200 epochs

Modes:
  multi          — all 4 tasks
  exercise       — single-task exercise
  phase          — single-task phase
  fatigue        — single-task fatigue
  reps           — single-task reps

Concurrency: max MAX_CONCURRENT subprocesses at any time (default 3 — meant
for 1× RTX 5070 Ti). Resumable: each (arch, mode, window) writes to a
deterministic run-dir; existing cv_summary.json files cause skip.

Usage:
    python scripts/run_sweep_raw.py
    python scripts/run_sweep_raw.py --max-concurrent 2 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parent.parent

ARCHS = ['cnn1d_raw', 'tcn_raw', 'cnn_lstm_raw']
MODES = ['multi', 'exercise', 'phase', 'fatigue', 'reps']
WINDOWS = [2.0, 1.0, 5.0]   # base = WINDOWS[0]
SEEDS = [42, 7, 1337]
SPLITS = ROOT / 'configs' / 'splits.csv'
RESULTS_CSV = ROOT / 'results' / 'sweep_raw_results.csv'
PER_FOLD_CSV = ROOT / 'results' / 'sweep_raw_per_fold.csv'
LOG_DIR = ROOT / 'logs' / 'sweep_raw'
RUNS_ROOT = ROOT / 'runs' / 'sweep_raw'


def run_dir_for(arch: str, mode: str, window_s: float) -> Path:
    return (RUNS_ROOT / f"{arch}__{mode}__w{int(window_s)}s")


def phase2_dir_for(arch: str, mode: str, window_s: float, seed: int) -> Path:
    return run_dir_for(arch, mode, window_s) / 'phase2_seeds' / f"seed_{seed}"


def tasks_for(mode: str) -> list[str]:
    return ['exercise', 'phase', 'fatigue', 'reps'] if mode == 'multi' else [mode]


# ─────────────────────────────────────────────────────────────────────────────
# Job queue with bounded concurrency
# ─────────────────────────────────────────────────────────────────────────────

class JobQueue:
    """Thread-safe submit/wait pool with at most `max_concurrent` subprocesses."""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._slots = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._active: list[tuple[subprocess.Popen, Path]] = []

    def submit(self, cmd: list[str], log_path: Path, label: str) -> threading.Thread:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def runner():
            self._slots.acquire()
            with self._lock:
                print(f"[queue] START {label}\n         cmd: {' '.join(cmd)}\n         log: {log_path}")
            t0 = time.time()
            with open(log_path, 'w', encoding='utf-8') as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=str(ROOT), env={**os.environ,
                                      'PYTHONIOENCODING': 'utf-8'})
                with self._lock:
                    self._active.append((p, log_path))
                ret = p.wait()
                with self._lock:
                    self._active = [(pp, lp) for pp, lp in self._active
                                     if pp.pid != p.pid]
            elapsed = (time.time() - t0) / 60
            with self._lock:
                status = 'OK' if ret == 0 else f'FAIL({ret})'
                print(f"[queue] DONE  {label}  {status}  {elapsed:.1f} min")
            self._slots.release()

        t = threading.Thread(target=runner, daemon=False)
        t.start()
        return t

    def wait_all(self, threads: list[threading.Thread]):
        for t in threads:
            t.join()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess command builders
# ─────────────────────────────────────────────────────────────────────────────

def build_optuna_cmd(arch: str, mode: str, window_s: float, n_trials: int,
                      seed_hps_from: Path | None, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable, 'scripts/train_optuna.py',
        '--arch', arch, '--variant', 'raw',
        '--n-trials', str(n_trials),
        '--phase1-epochs', '100', '--phase2-epochs', '200',
        '--patience', '15',
        '--tasks', *tasks_for(mode),
        '--window-s', str(window_s),
        '--splits', str(SPLITS),
        '--exercise-aggregation', 'both',
        '--wide-arch-search',
        '--repr-dim-choices', '64', '128', '256',
        '--skip-phase2',
        '--run-dir', str(run_dir),
    ]
    if seed_hps_from is not None:
        cmd += ['--seed-hps-from', str(seed_hps_from)]
    return cmd


def build_phase2_cmd(arch: str, mode: str, window_s: float, seed: int,
                      src_run_dir: Path, out_run_dir: Path) -> list[str]:
    return [
        sys.executable, 'scripts/train_phase2_only.py',
        '--arch', arch, '--variant', 'raw',
        '--src-run-dir', str(src_run_dir),
        '--out-run-dir', str(out_run_dir),
        '--phase2-seeds', str(seed),
        '--phase2-epochs', '200',
        '--patience', '15',
        '--tasks', *tasks_for(mode),
        '--window-s', str(window_s),
        '--splits', str(SPLITS),
        '--exercise-aggregation', 'both',
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction → CSV
# ─────────────────────────────────────────────────────────────────────────────

def _safe(d: dict, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def gather_phase2_to_csv(arch: str, mode: str, window_s: float, seed: int,
                          phase2_dir: Path, results_csv: Path,
                          per_fold_csv: Path, src_run_dir: Path):
    """Append rows to results CSV (one per seed) + per-fold CSV (one per fold)."""
    cv_summary_path = next(phase2_dir.rglob('cv_summary.json'), None)
    if cv_summary_path is None:
        print(f"[csv] WARN: no cv_summary.json under {phase2_dir}")
        return
    summary = json.loads(cv_summary_path.read_text())['summary']
    all_results = json.loads(cv_summary_path.read_text()).get('all_results', [])

    best_hps_path = src_run_dir / 'best_hps.json'
    best_hps = (json.loads(best_hps_path.read_text())['best_hps']
                if best_hps_path.exists() else {})

    base_row = {
        'script': 'sweep_raw',
        'arch': arch, 'variant': 'raw',
        'mode': mode, 'window_s': window_s, 'seed': seed,
        'best_hps_json': json.dumps(best_hps),
    }

    seed_row = dict(base_row)
    seed_row.update({
        'fold': 'aggregate',
        'exercise_pw_f1':       _safe(summary, 'exercise', 'f1_macro', 'mean'),
        'exercise_pw_balacc':   _safe(summary, 'exercise', 'balanced_accuracy', 'mean'),
        'exercise_ps_f1':       _safe(summary, 'exercise_per_set', 'f1_macro', 'mean'),
        'exercise_ps_balacc':   _safe(summary, 'exercise_per_set', 'balanced_accuracy', 'mean'),
        'phase_f1':             _safe(summary, 'phase', 'f1_macro', 'mean'),
        'phase_balacc':         _safe(summary, 'phase', 'balanced_accuracy', 'mean'),
        'fatigue_mae':          _safe(summary, 'fatigue', 'mae', 'mean'),
        'fatigue_r':            _safe(summary, 'fatigue', 'pearson_r', 'mean'),
        'reps_mae':             _safe(summary, 'reps', 'mae', 'mean'),
        'val_total':            _safe(summary, 'val_total', 'mean'),
        'n_test_windows': '', 'n_test_sets': '', 'test_subjects': '',
    })
    _append_csv(results_csv, seed_row)

    for r in all_results:
        m = r.get('metrics', {})
        fold_row = dict(base_row)
        fold_row.update({
            'fold': r.get('fold'),
            'test_subjects': '|'.join(r.get('test_subjects', [])),
            'exercise_pw_f1':     _safe(m, 'exercise', 'f1_macro'),
            'exercise_pw_balacc': _safe(m, 'exercise', 'balanced_accuracy'),
            'exercise_ps_f1':     _safe(m, 'exercise_per_set', 'f1_macro'),
            'exercise_ps_balacc': _safe(m, 'exercise_per_set', 'balanced_accuracy'),
            'phase_f1':           _safe(m, 'phase', 'f1_macro'),
            'phase_balacc':       _safe(m, 'phase', 'balanced_accuracy'),
            'fatigue_mae':        _safe(m, 'fatigue', 'mae'),
            'fatigue_r':          _safe(m, 'fatigue', 'pearson_r'),
            'reps_mae':           _safe(m, 'reps', 'mae'),
            'val_total':          m.get('val_total'),
            'n_test_windows':     _safe(m, 'exercise', 'n'),
            'n_test_sets':        _safe(m, 'exercise_per_set', 'n_sets'),
        })
        _append_csv(per_fold_csv, fold_row)


CSV_COLS = [
    'script', 'arch', 'variant', 'mode', 'window_s', 'seed', 'fold',
    'test_subjects',
    'exercise_pw_f1', 'exercise_pw_balacc',
    'exercise_ps_f1', 'exercise_ps_balacc',
    'phase_f1', 'phase_balacc',
    'fatigue_mae', 'fatigue_r', 'reps_mae',
    'val_total', 'n_test_windows', 'n_test_sets', 'best_hps_json',
]
_CSV_LOCK = threading.Lock()


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with _CSV_LOCK:
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in CSV_COLS})


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-concurrent', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true',
                     help='Print what would run, do not launch')
    ap.add_argument('--archs', nargs='+', default=ARCHS)
    ap.add_argument('--modes', nargs='+', default=MODES)
    ap.add_argument('--windows', type=float, nargs='+', default=WINDOWS,
                     help=f'First entry is base. Default: {WINDOWS}')
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = ap.parse_args()

    base_w = args.windows[0]
    other_ws = args.windows[1:]
    pool = JobQueue(args.max_concurrent)

    print(f"[sweep_raw] archs={args.archs} modes={args.modes} "
          f"base_window={base_w}s other_windows={other_ws} seeds={args.seeds} "
          f"max_concurrent={args.max_concurrent}")

    # ─── PHASE A: Optuna at base window for every (arch, mode) ────────────
    print("\n=== PHASE A: Optuna 100 trials at base window per (arch, mode) ===")
    threads_a = []
    for arch in args.archs:
        for mode in args.modes:
            rd = run_dir_for(arch, mode, base_w)
            if (rd / 'best_hps.json').exists():
                print(f"[skip] base done: {rd}")
                continue
            cmd = build_optuna_cmd(arch, mode, base_w, n_trials=100,
                                    seed_hps_from=None, run_dir=rd)
            log = LOG_DIR / f"A_{arch}__{mode}__w{int(base_w)}s.log"
            label = f"A {arch}/{mode}/w{int(base_w)}s"
            if args.dry_run:
                print(f"[dry] {label}  ->  {' '.join(cmd)}")
            else:
                threads_a.append(pool.submit(cmd, log, label))
    if not args.dry_run:
        pool.wait_all(threads_a)

    # ─── PHASE B: Optuna 20 trials at other windows, warm-started ─────────
    print("\n=== PHASE B: Optuna 20 trials at other windows (warm-start) ===")
    threads_b = []
    for arch in args.archs:
        for mode in args.modes:
            base_dir = run_dir_for(arch, mode, base_w)
            base_hps = base_dir / 'best_hps.json'
            if not base_hps.exists():
                print(f"[skip] base HPs missing for {arch}/{mode}: {base_hps}")
                continue
            for w in other_ws:
                rd = run_dir_for(arch, mode, w)
                if (rd / 'best_hps.json').exists():
                    print(f"[skip] window done: {rd}")
                    continue
                cmd = build_optuna_cmd(arch, mode, w, n_trials=20,
                                        seed_hps_from=base_hps, run_dir=rd)
                log = LOG_DIR / f"B_{arch}__{mode}__w{int(w)}s.log"
                label = f"B {arch}/{mode}/w{int(w)}s"
                if args.dry_run:
                    print(f"[dry] {label}  ->  {' '.join(cmd)}")
                else:
                    threads_b.append(pool.submit(cmd, log, label))
    if not args.dry_run:
        pool.wait_all(threads_b)

    # ─── PHASE C: Phase 2 refit, 3 seeds × 3 windows ──────────────────────
    print("\n=== PHASE C: Phase 2 refit (3 seeds × all windows) ===")
    threads_c = []
    for arch in args.archs:
        for mode in args.modes:
            for w in args.windows:
                src = run_dir_for(arch, mode, w)
                if not (src / 'best_hps.json').exists():
                    print(f"[skip] no HPs for phase 2: {src}")
                    continue
                for seed in args.seeds:
                    out = phase2_dir_for(arch, mode, w, seed)
                    if next(out.rglob('cv_summary.json'), None) is not None:
                        print(f"[skip] phase 2 done: {out}")
                        continue
                    cmd = build_phase2_cmd(arch, mode, w, seed, src, out)
                    log = LOG_DIR / f"C_{arch}__{mode}__w{int(w)}s__s{seed}.log"
                    label = f"C {arch}/{mode}/w{int(w)}s/seed{seed}"
                    if args.dry_run:
                        print(f"[dry] {label}  ->  {' '.join(cmd)}")
                    else:
                        threads_c.append(pool.submit(cmd, log, label))
    if not args.dry_run:
        pool.wait_all(threads_c)

    # ─── Gather all phase 2 outputs into CSV ──────────────────────────────
    print("\n=== Gathering results into CSV ===")
    if not args.dry_run:
        # Wipe before re-gather so resumed runs don't duplicate rows.
        for csvp in (RESULTS_CSV, PER_FOLD_CSV):
            if csvp.exists():
                csvp.unlink()
        for arch in args.archs:
            for mode in args.modes:
                for w in args.windows:
                    src = run_dir_for(arch, mode, w)
                    for seed in args.seeds:
                        p2 = phase2_dir_for(arch, mode, w, seed)
                        if next(p2.rglob('cv_summary.json'), None) is None:
                            continue
                        gather_phase2_to_csv(arch, mode, w, seed, p2,
                                              RESULTS_CSV, PER_FOLD_CSV,
                                              src_run_dir=src)
        print(f"[csv] wrote {RESULTS_CSV}")
        print(f"[csv] wrote {PER_FOLD_CSV}")

    print("\n[sweep_raw] DONE")


if __name__ == '__main__':
    main()
