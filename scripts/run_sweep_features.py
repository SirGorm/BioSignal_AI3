"""Sweep over feature-input architectures (MLP, LSTM).

Standalone orchestrator — designed to be moved to a different machine and run
on its own. Steps:

  STEP 0  LDA / ANOVA F-score selection on the 32 candidate features for
          the exercise target → write top-15 to results/sweep_features_top15.json
  PHASE A Optuna 100 trials at BASE window (--window-s 2.0) per (arch, mode)
  PHASE B Optuna 20 trials at OTHER windows (1, 5), warm-started from base
          via --seed-hps-from
  PHASE C Phase 2 refit per window with 3 seeds × 5 GKF folds × 200 epochs

Modes:
  multi          — all 4 tasks
  exercise       — single-task exercise
  phase          — single-task phase
  fatigue        — single-task fatigue
  reps           — single-task reps

LSTM tunes n_layers ∈ [1, 2]; MLP uses defaults. Both run with
--exercise-aggregation both → metrics CSV has both per-window and per-set.

All results go to:
  results/sweep_features_results.csv     — one row per (arch, mode, win, seed)
  results/sweep_features_per_fold.csv    — one row per (arch, mode, win,
                                            seed, fold)
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.datasets import EXCLUDED_FEATURE_PREFIXES  # noqa: E402

ARCHS = ['mlp', 'lstm']
MODES = ['multi', 'exercise', 'phase', 'fatigue', 'reps']
WINDOWS = [2.0, 1.0, 5.0]   # base = WINDOWS[0]
SEEDS = [42, 7, 1337]
SPLITS = ROOT / 'configs' / 'splits.csv'
LABELED_ROOT = ROOT / 'data' / 'labeled'
TOP_K = 15
TOP_K_JSON = ROOT / 'results' / 'sweep_features_top15.json'
RESULTS_CSV = ROOT / 'results' / 'sweep_features_results.csv'
PER_FOLD_CSV = ROOT / 'results' / 'sweep_features_per_fold.csv'
LOG_DIR = ROOT / 'logs' / 'sweep_features'
RUNS_ROOT = ROOT / 'runs' / 'sweep_features'


def run_dir_for(arch, mode, w):
    return RUNS_ROOT / f"{arch}__{mode}__w{int(w)}s"


def phase2_dir_for(arch, mode, w, seed):
    return run_dir_for(arch, mode, w) / 'phase2_seeds' / f"seed_{seed}"


def tasks_for(mode):
    return ['exercise', 'phase', 'fatigue', 'reps'] if mode == 'multi' else [mode]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LDA / ANOVA F-score top-15 selection
# ─────────────────────────────────────────────────────────────────────────────

def select_top15_features(verbose: bool = True) -> list[str]:
    """ANOVA F-score on exercise label, top-15. Reads window_features.parquet
    from data/labeled, filters to active+valid-exercise rows, computes F-score
    per candidate feature column.

    Mirrors the auto-selection in train_rf_exercise_perset.py (strip ECG/EDA,
    soft targets, metadata, label cols) so the candidate pool is the 32-ish
    feature columns we have been using throughout the project.
    """
    from sklearn.feature_selection import f_classif

    paths = sorted(LABELED_ROOT.rglob('window_features.parquet'))
    if not paths:
        raise FileNotFoundError(f"No window_features.parquet under {LABELED_ROOT}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df[df['in_active_set'] & df['exercise'].isin(
        ['squat', 'deadlift', 'benchpress', 'pullup'])].copy()
    if verbose:
        print(f"[lda] loaded {len(df)} active+valid windows from "
              f"{len(paths)} parquets")

    META = {'recording_id', 'subject_id', 'session_id', 'set_number',
            'rep_index', 't_unix', 't_session_s',
            't_window_start', 't_window_end', 'in_active_set', 'set_phase',
            't_window_center_s'}
    LABEL = {'exercise', 'phase_label', 'rep_count_in_set',
             'rpe_for_this_set', 'rpe', 'reps_in_window_2s',
             'soft_overlap_reps', 'has_rep_intervals', 'rep_density_hz'}
    cands = [c for c in df.columns
             if c not in META and c not in LABEL
             and not c.startswith(EXCLUDED_FEATURE_PREFIXES)
             and not c.startswith('phase_frac_')
             and not c.startswith('soft_overlap_reps_')
             and pd.api.types.is_numeric_dtype(df[c])]
    if verbose:
        print(f"[lda] candidate features ({len(cands)}): {cands[:5]} ...")

    X = df[cands].fillna(df[cands].median()).to_numpy(dtype=float)
    y = df['exercise'].values

    f_scores, p_values = f_classif(X, y)
    ranking = sorted(zip(cands, f_scores, p_values), key=lambda r: -r[1])
    top = ranking[:TOP_K]

    TOP_K_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'method': 'ANOVA F-score on exercise label',
        'n_candidates': len(cands),
        'top_k': TOP_K,
        'rows_used': int(len(df)),
        'features': [c for c, _, _ in top],
        'f_score': [float(f) for _, f, _ in top],
        'p_value': [float(p) for _, _, p in top],
        'all_ranked': [{'feature': c, 'f': float(f), 'p': float(p)}
                       for c, f, p in ranking],
    }
    TOP_K_JSON.write_text(json.dumps(payload, indent=2))
    if verbose:
        print(f"[lda] top {TOP_K} (F-score):")
        for c, f, p in top:
            print(f"   {c:30s}  F={f:9.2f}  p={p:.2e}")
        print(f"[lda] wrote {TOP_K_JSON}")
    return [c for c, _, _ in top]


# ─────────────────────────────────────────────────────────────────────────────
# Job queue (same as run_sweep_raw.py — kept inline to remain standalone)
# ─────────────────────────────────────────────────────────────────────────────

class JobQueue:
    def __init__(self, max_concurrent: int):
        self._slots = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()

    def submit(self, cmd, log_path, label):
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def runner():
            self._slots.acquire()
            with self._lock:
                print(f"[queue] START {label}\n         log: {log_path}")
            t0 = time.time()
            with open(log_path, 'w', encoding='utf-8') as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=str(ROOT),
                                      env={**os.environ,
                                            'PYTHONIOENCODING': 'utf-8'})
                ret = p.wait()
            elapsed = (time.time() - t0) / 60
            with self._lock:
                status = 'OK' if ret == 0 else f'FAIL({ret})'
                print(f"[queue] DONE  {label}  {status}  {elapsed:.1f} min")
            self._slots.release()

        t = threading.Thread(target=runner, daemon=False)
        t.start()
        return t

    @staticmethod
    def wait_all(threads):
        for t in threads:
            t.join()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess command builders
# ─────────────────────────────────────────────────────────────────────────────

def build_optuna_cmd(arch, mode, w, n_trials, top_features,
                      seed_hps_from, run_dir):
    cmd = [
        sys.executable, 'scripts/train_optuna.py',
        '--arch', arch, '--variant', 'features',
        '--n-trials', str(n_trials),
        '--phase1-epochs', '100', '--phase2-epochs', '200',
        '--patience', '15',
        '--tasks', *tasks_for(mode),
        '--window-s', str(w),
        '--splits', str(SPLITS),
        '--exercise-aggregation', 'both',
        '--wide-arch-search',
        '--repr-dim-choices', '16', '32', '64',
        '--feature-cols', *top_features,
        '--skip-phase2',
        '--run-dir', str(run_dir),
    ]
    if seed_hps_from is not None:
        cmd += ['--seed-hps-from', str(seed_hps_from)]
    return cmd


def build_phase2_cmd(arch, mode, w, seed, top_features,
                      src_run_dir, out_run_dir):
    cmd = [
        sys.executable, 'scripts/train_phase2_only.py',
        '--arch', arch, '--variant', 'features',
        '--src-run-dir', str(src_run_dir),
        '--out-run-dir', str(out_run_dir),
        '--phase2-seeds', str(seed),
        '--phase2-epochs', '200',
        '--patience', '15',
        '--tasks', *tasks_for(mode),
        '--window-s', str(w),
        '--splits', str(SPLITS),
        '--exercise-aggregation', 'both',
    ]
    # train_phase2_only.py uses --feature-prefixes (startswith match). The
    # exact column names also work as prefixes when none of the chosen
    # columns are prefix-substrings of others (true for our top-15).
    cmd += ['--feature-prefixes', *top_features]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Result extraction → CSV
# ─────────────────────────────────────────────────────────────────────────────

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


def _safe(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with _CSV_LOCK:
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, '') for k in CSV_COLS})


def gather_phase2_to_csv(arch, mode, w, seed, phase2_dir, src_run_dir):
    cv_summary_path = next(phase2_dir.rglob('cv_summary.json'), None)
    if cv_summary_path is None:
        print(f"[csv] WARN: no cv_summary.json under {phase2_dir}")
        return
    payload = json.loads(cv_summary_path.read_text())
    summary = payload['summary']
    all_results = payload.get('all_results', [])

    best_hps_path = src_run_dir / 'best_hps.json'
    best_hps = (json.loads(best_hps_path.read_text()).get('best_hps', {})
                if best_hps_path.exists() else {})

    base = {
        'script': 'sweep_features',
        'arch': arch, 'variant': 'features',
        'mode': mode, 'window_s': w, 'seed': seed,
        'best_hps_json': json.dumps(best_hps),
    }

    seed_row = dict(base)
    seed_row.update({
        'fold': 'aggregate',
        'exercise_pw_f1':     _safe(summary, 'exercise', 'f1_macro', 'mean'),
        'exercise_pw_balacc': _safe(summary, 'exercise', 'balanced_accuracy', 'mean'),
        'exercise_ps_f1':     _safe(summary, 'exercise_per_set', 'f1_macro', 'mean'),
        'exercise_ps_balacc': _safe(summary, 'exercise_per_set', 'balanced_accuracy', 'mean'),
        'phase_f1':           _safe(summary, 'phase', 'f1_macro', 'mean'),
        'phase_balacc':       _safe(summary, 'phase', 'balanced_accuracy', 'mean'),
        'fatigue_mae':        _safe(summary, 'fatigue', 'mae', 'mean'),
        'fatigue_r':          _safe(summary, 'fatigue', 'pearson_r', 'mean'),
        'reps_mae':           _safe(summary, 'reps', 'mae', 'mean'),
        'val_total':          _safe(summary, 'val_total', 'mean'),
        'n_test_windows': '', 'n_test_sets': '', 'test_subjects': '',
    })
    _append_csv(RESULTS_CSV, seed_row)

    for r in all_results:
        m = r.get('metrics', {})
        fold_row = dict(base)
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
        _append_csv(PER_FOLD_CSV, fold_row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-concurrent', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--archs', nargs='+', default=ARCHS)
    ap.add_argument('--modes', nargs='+', default=MODES)
    ap.add_argument('--windows', type=float, nargs='+', default=WINDOWS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--skip-lda', action='store_true',
                     help='Skip LDA selection; assume top-15 JSON exists')
    args = ap.parse_args()

    base_w = args.windows[0]
    other_ws = args.windows[1:]
    pool = JobQueue(args.max_concurrent)

    print(f"[sweep_features] archs={args.archs} modes={args.modes} "
          f"base_window={base_w}s other_windows={other_ws} seeds={args.seeds} "
          f"max_concurrent={args.max_concurrent}")

    # ─── STEP 0: LDA selection ─────────────────────────────────────────────
    if args.skip_lda and TOP_K_JSON.exists():
        top_features = json.loads(TOP_K_JSON.read_text())['features']
        print(f"[sweep_features] using cached top-15: {top_features}")
    else:
        print("\n=== STEP 0: LDA / ANOVA top-15 selection ===")
        top_features = select_top15_features()

    # ─── PHASE A: Optuna at base window ────────────────────────────────────
    print("\n=== PHASE A: Optuna 100 trials at base window per (arch, mode) ===")
    threads_a = []
    for arch in args.archs:
        for mode in args.modes:
            rd = run_dir_for(arch, mode, base_w)
            if (rd / 'best_hps.json').exists():
                print(f"[skip] base done: {rd}")
                continue
            cmd = build_optuna_cmd(arch, mode, base_w, n_trials=100,
                                    top_features=top_features,
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
                                        top_features=top_features,
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
                    cmd = build_phase2_cmd(arch, mode, w, seed, top_features,
                                            src, out)
                    log = LOG_DIR / f"C_{arch}__{mode}__w{int(w)}s__s{seed}.log"
                    label = f"C {arch}/{mode}/w{int(w)}s/seed{seed}"
                    if args.dry_run:
                        print(f"[dry] {label}  ->  {' '.join(cmd)}")
                    else:
                        threads_c.append(pool.submit(cmd, log, label))
    if not args.dry_run:
        pool.wait_all(threads_c)

    # ─── Gather results ───────────────────────────────────────────────────
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
                                              src_run_dir=src)
        print(f"[csv] wrote {RESULTS_CSV}")
        print(f"[csv] wrote {PER_FOLD_CSV}")

    print("\n[sweep_features] DONE")


if __name__ == '__main__':
    main()
