"""Random Forest baseline for single-task exercise classification, with both
per-window and per-set evaluation (Rute A — mirrors the NN setup in
runs/v18single-exercise-mlp-perset-w5s/).

Reads window_features.parquet from data/labeled/, strides to match a given
window_s on the 100 Hz feature grid, fits RandomForestClassifier with LOSO
10-fold GroupKFold, and reports:
  - per-window F1 / balanced-acc (one prediction per window)
  - per-set     F1 / balanced-acc (mean of predict_proba over all valid
    windows in a (recording_id, set_number) group → argmax)

Usage:
    python scripts/train_rf_exercise_perset.py --window-s 5.0 \
        --out-dir runs/v18single-exercise-rf-perset-w5s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.datasets import EXCLUDED_FEATURE_PREFIXES  # ECG/EDA stripped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-s', type=float, default=5.0)
    ap.add_argument('--labeled-root', type=Path,
                     default=ROOT / 'data' / 'labeled')
    ap.add_argument('--splits', type=Path,
                     default=ROOT / 'configs' / 'splits_loso.csv')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--n-trees', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=12)
    ap.add_argument('--min-samples-leaf', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-trials', type=int, default=0,
                     help='Optuna trials. 0 (default) = use --n-trees / '
                          '--max-depth / --min-samples-leaf as-is. >0 = '
                          'tune n_estimators, max_depth, min_samples_split, '
                          'min_samples_leaf, max_features by minimizing '
                          'negative per-window F1 over the CV folds (same '
                          'objective as scripts/train_lgbm.py).')
    ap.add_argument('--feature-cols', nargs='+', default=None,
                     help='Restrict input to this exact list of column '
                          'names (skips the default modality+strip-prefix '
                          'auto-selection). Example: --feature-cols '
                          'emg_lscore emg_mfl emg_msr emg_wamp acc_lscore '
                          'acc_mfl acc_msr acc_wamp')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load + concat ------------------------------------------------------
    paths = sorted(args.labeled_root.rglob('window_features.parquet'))
    print(f"[rf-perset] loading {len(paths)} parquets")
    wf = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    print(f"[rf-perset] concat shape: {wf.shape}")

    # Stride to match a (window_s/2) hop on the 100 Hz grid — same convention
    # as src/data/datasets.py:WindowFeatureDataset.
    stride = max(1, int(round(args.window_s / 2 * 100)))
    n0 = len(wf)
    wf = (wf.groupby('recording_id', sort=False, group_keys=False)
            .apply(lambda g: g.iloc[::stride])
            .reset_index(drop=True))
    print(f"[rf-perset] stride={stride}  rows {n0} -> {len(wf)}")

    # Active + valid exercise label only
    valid_classes = ['squat', 'deadlift', 'benchpress', 'pullup']
    wf = wf[wf['in_active_set'] & wf['exercise'].isin(valid_classes)].copy()
    print(f"[rf-perset] active+valid: {len(wf)} windows")
    print("[rf-perset] class balance:",
          wf['exercise'].value_counts().to_dict())

    # --- Feature columns ----------------------------------------------------
    if args.feature_cols:
        missing = [c for c in args.feature_cols if c not in wf.columns]
        if missing:
            raise KeyError(
                f"--feature-cols references missing columns: {missing}. "
                f"Available: {sorted(wf.columns.tolist())[:20]} ..."
            )
        feat_cols = list(args.feature_cols)
        print(f"[rf-perset] features (--feature-cols): {feat_cols}")
    else:
        META = {'recording_id', 'subject_id', 'session_id', 'set_number',
                'rep_index', 't_unix', 't_session_s',
                't_window_start', 't_window_end', 'in_active_set', 'set_phase'}
        LABEL = {'exercise', 'phase_label', 'rep_count_in_set',
                 'rpe_for_this_set', 'rpe', 'reps_in_window_2s',
                 'soft_overlap_reps', 'has_rep_intervals'}
        feat_cols = [c for c in wf.columns
                     if c not in META and c not in LABEL
                     and not c.startswith(EXCLUDED_FEATURE_PREFIXES)
                     and not c.startswith('phase_frac_')
                     and not c.startswith('soft_overlap_reps_')]
        feat_cols = [c for c in feat_cols
                     if pd.api.types.is_numeric_dtype(wf[c])]
        print(f"[rf-perset] features: {len(feat_cols)}")

    X = wf[feat_cols].to_numpy(dtype=float)
    le = LabelEncoder()
    y = le.fit_transform(wf['exercise'].values)
    classes = list(le.classes_)
    print(f"[rf-perset] classes: {classes}")

    # --- LOSO splits --------------------------------------------------------
    splits_df = pd.read_csv(args.splits)  # cols: subject_id, fold, split
    n_folds = int(splits_df['fold'].nunique())
    print(f"[rf-perset] LOSO {n_folds} folds")

    subj = wf['subject_id'].astype(str).values
    rec = wf['recording_id'].astype(str).values
    set_n = pd.to_numeric(wf['set_number'], errors='coerce').values

    # Build folds: fold f → test = subjects whose splits_df.fold == f
    fold_subjects = (splits_df.groupby('fold')['subject_id']
                              .apply(lambda s: list(s.astype(str))).to_dict())
    all_subjects = set(splits_df['subject_id'].astype(str).unique())

    # Pre-compute (train_idx, test_idx) per fold; reused for Optuna + final.
    fold_indices = []
    for fold in sorted(fold_subjects.keys()):
        test_subj = set(fold_subjects[fold])
        train_subj = all_subjects - test_subj
        tr = np.flatnonzero(np.isin(subj, list(train_subj)))
        te = np.flatnonzero(np.isin(subj, list(test_subj)))
        if len(te) and len(tr):
            fold_indices.append((fold, sorted(test_subj), tr, te))

    # --- Optuna (optional) --------------------------------------------------
    best_params = None
    if args.n_trials > 0:
        print(f"\n[rf-perset] === Optuna ({args.n_trials} trials, "
              f"objective = -mean(per-window F1) over LOSO folds) ===")

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int('n_estimators', 100, 600),
                max_depth=trial.suggest_int('max_depth', 4, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical(
                    'max_features', ['sqrt', 'log2', 0.5, 1.0]),
                class_weight='balanced',
                n_jobs=-1,
                random_state=args.seed,
            )
            f1s = []
            for _, _, tr, te in fold_indices:
                pipe = Pipeline([
                    ('imp', SimpleImputer(strategy='median')),
                    ('rf', RandomForestClassifier(**params)),
                ])
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                f1s.append(f1_score(y[te], pred, labels=list(range(4)),
                                     average='macro', zero_division=0))
            return -float(np.mean(f1s))

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        t0 = time.time()
        study.optimize(objective, n_trials=args.n_trials,
                        show_progress_bar=False)
        elapsed = time.time() - t0
        best_params = dict(study.best_params)
        print(f"[rf-perset] Optuna done in {elapsed:.0f}s. "
              f"Best -F1={study.best_value:.4f}  "
              f"(F1={-study.best_value:.4f})")
        print(f"[rf-perset] Best params: {best_params}")
        (args.out_dir / 'best_hps.json').write_text(json.dumps({
            'best_value_neg_f1': float(study.best_value),
            'best_value_f1': float(-study.best_value),
            'best_params': best_params,
            'n_trials': len(study.trials),
            'optuna_elapsed_s': float(elapsed),
        }, indent=2))

    fold_pw_f1 = []
    fold_pw_bacc = []
    fold_ps_f1 = []
    fold_ps_bacc = []
    cm_pw = np.zeros((4, 4), dtype=int)
    cm_ps = np.zeros((4, 4), dtype=int)
    fold_records = []

    if best_params is not None:
        rf_kwargs = dict(best_params,
                          class_weight='balanced',
                          n_jobs=-1,
                          random_state=args.seed)
    else:
        rf_kwargs = dict(
            n_estimators=args.n_trees,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight='balanced',
            n_jobs=-1,
            random_state=args.seed,
        )

    print(f"\n[rf-perset] === Final LOSO eval ===")
    for fold, test_subj, tr, te in fold_indices:
        t0 = time.time()
        pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('rf', RandomForestClassifier(**rf_kwargs)),
        ])
        pipe.fit(X[tr], y[tr])
        # Per-window predictions
        proba = pipe.predict_proba(X[te])
        pred_pw = np.argmax(proba, axis=1)

        f1_pw = f1_score(y[te], pred_pw, labels=list(range(4)),
                          average='macro', zero_division=0)
        bacc_pw = balanced_accuracy_score(y[te], pred_pw)
        cm_pw += confusion_matrix(y[te], pred_pw, labels=list(range(4)))

        # Per-set aggregation: mean(proba) over (recording, set), argmax
        rec_te = rec[te]
        sn_te = set_n[te]
        keys = np.array([
            f"{r}__{int(s)}" if s == s and s >= 0 else ""
            for r, s in zip(rec_te, sn_te)
        ], dtype=object)

        ps_true, ps_pred = [], []
        # Group by key
        groups = {}
        for i, k in enumerate(keys):
            if k == "":
                continue
            groups.setdefault(k, []).append(i)
        for k, idxs in groups.items():
            mean_p = proba[idxs].mean(axis=0)
            ps_pred.append(int(np.argmax(mean_p)))
            ps_true.append(int(y[te][idxs[0]]))  # broadcast within set
        if ps_true:
            ps_true = np.asarray(ps_true)
            ps_pred = np.asarray(ps_pred)
            f1_ps = f1_score(ps_true, ps_pred, labels=list(range(4)),
                              average='macro', zero_division=0)
            bacc_ps = balanced_accuracy_score(ps_true, ps_pred)
            cm_ps += confusion_matrix(ps_true, ps_pred, labels=list(range(4)))
            n_sets = len(ps_true)
        else:
            f1_ps = bacc_ps = float('nan')
            n_sets = 0

        elapsed = time.time() - t0
        fold_pw_f1.append(float(f1_pw))
        fold_pw_bacc.append(float(bacc_pw))
        fold_ps_f1.append(float(f1_ps))
        fold_ps_bacc.append(float(bacc_ps))
        fold_records.append({
            'fold': fold,
            'test_subjects': sorted(test_subj),
            'n_train_windows': int(len(tr)),
            'n_test_windows': int(len(te)),
            'n_test_sets': int(n_sets),
            'per_window': {'f1_macro': float(f1_pw),
                            'balanced_accuracy': float(bacc_pw)},
            'per_set':    {'f1_macro': float(f1_ps),
                            'balanced_accuracy': float(bacc_ps),
                            'n_sets': int(n_sets)},
            'fit_seconds': round(elapsed, 1),
        })
        print(f"[rf-perset] fold={fold}  test={sorted(test_subj)}  "
              f"pw_F1={f1_pw:.3f} bal={bacc_pw:.3f}  "
              f"ps_F1={f1_ps:.3f} bal={bacc_ps:.3f}  "
              f"n_sets={n_sets}  ({elapsed:.0f}s)")

    summary = {
        'window_s': args.window_s,
        'stride': stride,
        'n_features': len(feat_cols),
        'n_folds': len(fold_records),
        'classes': classes,
        'rf_params': {k: v for k, v in rf_kwargs.items() if k != 'n_jobs'},
        'optuna_n_trials': int(args.n_trials),
        'optuna_best_params': best_params,
        'per_window': {
            'f1_macro_mean': float(np.mean(fold_pw_f1)),
            'f1_macro_std':  float(np.std(fold_pw_f1)),
            'balanced_accuracy_mean': float(np.mean(fold_pw_bacc)),
            'balanced_accuracy_std':  float(np.std(fold_pw_bacc)),
        },
        'per_set': {
            'f1_macro_mean': float(np.nanmean(fold_ps_f1)),
            'f1_macro_std':  float(np.nanstd(fold_ps_f1)),
            'balanced_accuracy_mean': float(np.nanmean(fold_ps_bacc)),
            'balanced_accuracy_std':  float(np.nanstd(fold_ps_bacc)),
        },
        'confusion_matrix_per_window': cm_pw.tolist(),
        'confusion_matrix_per_set':     cm_ps.tolist(),
        'folds': fold_records,
    }
    out_json = args.out_dir / 'cv_summary.json'
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[rf-perset] === SUMMARY ===")
    print(f"  per-window  F1={summary['per_window']['f1_macro_mean']:.3f} "
          f"± {summary['per_window']['f1_macro_std']:.3f}  "
          f"bal-acc={summary['per_window']['balanced_accuracy_mean']:.3f}")
    print(f"  per-set     F1={summary['per_set']['f1_macro_mean']:.3f} "
          f"± {summary['per_set']['f1_macro_std']:.3f}  "
          f"bal-acc={summary['per_set']['balanced_accuracy_mean']:.3f}")
    print(f"\nWrote {out_json}")


if __name__ == '__main__':
    main()
