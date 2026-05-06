"""Per-session RPE plots — actual vs predicted per set, validation only.

For each of 3 NN models, generates 1 figure with 3 subplots:
  - Best subject  (highest avg Pearson r across all 3 models)
  - Worst subject (lowest avg Pearson r across all 3 models)
  - Average across all 7 subjects (mean actual vs mean predicted per set)

Models:
  feat-MLP @ 1s              (best phase F1 = 0.518)
  fatigue-raw-tcn @ 3s       (best fatigue r = +0.41)
  feat-LSTM @ 1s             (lowest composite loss)

Predictions come from LOSO 7-fold validation (test_preds.pt × 3 seeds × 5 folds).
Per-window predictions are aggregated to per-set by mean over windows belonging
to that set.

Output: runs/comparison_v9/per_session_rpe/
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v9" / "per_session_rpe"
OUT.mkdir(parents=True, exist_ok=True)

# 7 subjects in LOSO order (configs/splits_clean_loso.csv)
FOLD_SUBJECTS = {
    0: 'Vivian', 1: 'Hytten', 2: 'kiyomi', 3: 'lucas 2',
    4: 'Tias', 5: 'Juile', 6: 'Raghild',
}
SUBJECT_TO_REC = {
    'Vivian': 'recording_003', 'Hytten': 'recording_006',
    'kiyomi': 'recording_010', 'lucas 2': 'recording_011',
    'Tias': 'recording_012', 'Juile': 'recording_013',
    'Raghild': 'recording_014',
}

# Model definitions: (label, run_dir_name, variant, window_s)
MODELS = [
    ('feat-MLP @ 1s',           'optuna_clean_v9-w1s-multi-feat-mlp',     'features', 1.0),
    ('fatigue-raw-tcn @ 3s',    'optuna_clean_v9-w3s-fatigue-raw-tcn',    'raw',      3.0),
    ('feat-LSTM @ 1s',          'optuna_clean_v9-w1s-multi-feat-lstm',    'features', 1.0),
]

EXCLUDE = ['recording_004', 'recording_005', 'recording_007',
           'recording_008', 'recording_009']


def build_dataset_indexed(variant: str, window_s: float):
    """Re-construct dataset and return parallel arrays:
       (subject_ids, set_numbers, rpe_targets) — one per windowed sample.
    The order matches dataset[i] for i in 0..len(dataset)-1.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.datasets import WindowFeatureDataset
    from src.data.raw_window_dataset import RawMultimodalWindowDataset

    if variant == 'features':
        files = sorted((ROOT / 'data/labeled_clean').rglob('window_features.parquet'))
        files = [p for p in files if not any(e in str(p) for e in EXCLUDE)]
        ds = WindowFeatureDataset(window_parquets=files, active_only=True,
                                    target_modes={'reps':'soft_overlap','phase':'hard'},
                                    window_s=window_s, verbose=False)
        # Re-derive set_number per dataset row by replaying the loader logic.
        dfs = [pd.read_parquet(p) for p in files]
        df = pd.concat(dfs, ignore_index=True)
        # stride
        stride = max(1, int(round(window_s / 2 * 100)))
        df = df.groupby('recording_id', sort=False, group_keys=False)\
                .apply(lambda g: g.iloc[::stride])\
                .reset_index(drop=True)
        df = df[df['in_active_set'].astype(bool)].reset_index(drop=True)
        subj = df['subject_id'].astype(str).to_numpy()
        sets = df['set_number'].to_numpy()
        rpe = df['rpe_for_this_set'].to_numpy()
        assert len(subj) == len(ds), (len(subj), len(ds))
        return subj, sets, rpe

    else:
        files = sorted((ROOT / 'data/labeled_clean').rglob('aligned_features.parquet'))
        files = [p for p in files if not any(e in str(p) for e in EXCLUDE)]
        ds = RawMultimodalWindowDataset(parquet_paths=files, active_only=True,
                                          target_modes={'reps':'soft_overlap','phase':'hard'},
                                          window_s=window_s, verbose=False)
        # _window_idx[i] = (file_idx, start) → end_sample = start + window_size - 1
        # subject + set_number live in dfs[file_idx] at row end_sample
        WS = ds.window_size
        subj_arr = []
        set_arr = []
        rpe_arr = []
        for file_idx, start in ds._window_idx:
            df = ds._dfs[file_idx]
            row = df.iloc[start + WS - 1]
            subj_arr.append(str(row['subject_id']))
            set_arr.append(row.get('set_number', np.nan))
            rpe_arr.append(row.get('rpe_for_this_set', np.nan))
        return np.array(subj_arr), np.array(set_arr), np.array(rpe_arr)


def collect_per_set_predictions(rd: Path, variant: str, window_s: float):
    """Return dict {subject: {set_num: (mean_actual, mean_pred)}} aggregated
    across 3 seeds × LOSO folds.
    """
    print(f'  Building dataset for {variant} @ {window_s}s ...')
    subj_per_idx, set_per_idx, rpe_per_idx = build_dataset_indexed(variant, window_s)
    print(f'  Dataset has {len(subj_per_idx)} active windows')

    # Aggregate predictions per (subject, set_num)
    by_subj_set = defaultdict(lambda: defaultdict(lambda: {'pred': [], 'true': []}))
    p2 = rd / 'phase2'
    n_files = 0
    for seed_dir in p2.glob('*/seed_*'):
        for fold_dir in sorted(seed_dir.glob('fold_*')):
            try:
                d = torch.load(fold_dir / 'test_preds.pt', weights_only=False, map_location='cpu')
            except Exception:
                continue
            ti = d['test_idx']
            preds = d['preds']['fatigue'].numpy()
            targs = d['targets']['fatigue'].numpy()
            mask = d['masks']['fatigue'].numpy().astype(bool)
            ti = ti[mask]; preds = preds[mask]; targs = targs[mask]
            for i, idx in enumerate(ti):
                idx = int(idx)
                if idx >= len(subj_per_idx): continue
                subj = subj_per_idx[idx]
                sn = set_per_idx[idx]
                if pd.isna(sn): continue
                sn = int(sn)
                by_subj_set[subj][sn]['pred'].append(float(preds[i]))
                by_subj_set[subj][sn]['true'].append(float(targs[i]))
            n_files += 1
    print(f'  Processed {n_files} (fold,seed) prediction files')

    # Reduce to mean per (subject, set)
    out = {}
    for subj, set_dict in by_subj_set.items():
        out[subj] = {}
        for sn, vals in set_dict.items():
            actual = float(np.mean(vals['true']))   # constant per set anyway
            pred = float(np.mean(vals['pred']))     # mean across windows × seeds
            out[subj][sn] = (actual, pred)
    return out


def per_subject_pearson(per_set):
    """Given {subj: {set: (act, pred)}}, return {subj: r}."""
    out = {}
    for subj, sd in per_set.items():
        if len(sd) < 3: continue
        a = np.array([v[0] for v in sd.values()])
        p = np.array([v[1] for v in sd.values()])
        if np.std(a) == 0 or np.std(p) == 0: continue
        out[subj] = float(np.corrcoef(a, p)[0, 1])
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print(f'Loading per-set predictions for {len(MODELS)} models...\n')

    model_data = {}  # label → {subj: {set: (act, pred)}}
    for label, dir_name, variant, ws in MODELS:
        print(f'== {label} ==')
        rd = ROOT / 'runs' / dir_name
        model_data[label] = collect_per_set_predictions(rd, variant, ws)
        n_subj = len(model_data[label])
        avg_sets = np.mean([len(v) for v in model_data[label].values()])
        print(f'  → {n_subj} subjects, avg {avg_sets:.1f} sets/subject\n')

    # Compute average Pearson r per subject across all 3 models
    rs_by_subj = defaultdict(list)
    for label, d in model_data.items():
        for subj, r in per_subject_pearson(d).items():
            rs_by_subj[subj].append(r)
    avg_r = {s: float(np.mean(rs)) for s, rs in rs_by_subj.items() if len(rs) == len(MODELS)}
    sorted_subj = sorted(avg_r.items(), key=lambda x: -x[1])
    print('Per-subject AVG Pearson r across 3 models:')
    for s, r in sorted_subj:
        print(f'  {s:<12} {r:+.3f}')
    if not sorted_subj:
        print('No subjects ranked — aborting'); return
    best_subj = sorted_subj[0][0]
    worst_subj = sorted_subj[-1][0]
    print(f'\nBest:  {best_subj} (r={sorted_subj[0][1]:+.3f})')
    print(f'Worst: {worst_subj} (r={sorted_subj[-1][1]:+.3f})')

    # Build plots
    for label, d in model_data.items():
        # Best subject
        best_sets = sorted(d.get(best_subj, {}).items())
        worst_sets = sorted(d.get(worst_subj, {}).items())
        # Average across all subjects per set
        avg_actual_per_set = defaultdict(list)
        avg_pred_per_set = defaultdict(list)
        for subj, sd in d.items():
            for sn, (act, pred) in sd.items():
                avg_actual_per_set[sn].append(act)
                avg_pred_per_set[sn].append(pred)
        avg_sets_xs = sorted(avg_actual_per_set.keys())
        avg_actual = [np.mean(avg_actual_per_set[sn]) for sn in avg_sets_xs]
        avg_pred = [np.mean(avg_pred_per_set[sn]) for sn in avg_sets_xs]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        def _plot(ax, sets_data, title):
            if not sets_data:
                ax.text(0.5, 0.5, 'NO DATA', ha='center', va='center',
                         transform=ax.transAxes); return
            xs = [s[0] for s in sets_data]
            ys_act = [s[1][0] for s in sets_data]
            ys_pred = [s[1][1] for s in sets_data]
            ax.plot(xs, ys_act, '-o', color='#2c3e50', label='Actual RPE',
                     linewidth=2, markersize=8)
            ax.plot(xs, ys_pred, '--s', color='#e74c3c', label='Predicted RPE',
                     linewidth=2, markersize=7, markerfacecolor='white')
            ax.set_xlabel('Set number')
            ax.set_xticks(xs)
            ax.set_ylim(0, 11)
            ax.set_yticks(range(1, 11))
            ax.set_title(title, fontsize=11)
            ax.grid(linestyle=':', alpha=0.5)
            # Compute MAE + r for this subset
            a = np.array(ys_act); p = np.array(ys_pred)
            mae = float(np.mean(np.abs(a-p)))
            r = float(np.corrcoef(a,p)[0,1]) if np.std(a)>0 and np.std(p)>0 else 0
            ax.text(0.02, 0.98,
                     f'MAE={mae:.2f}\nr={r:+.2f}',
                     transform=ax.transAxes, va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        _plot(axes[0], best_sets,  f'Best: {best_subj}')
        _plot(axes[1], worst_sets, f'Worst: {worst_subj}')
        _plot(axes[2],
               list(zip(avg_sets_xs, list(zip(avg_actual, avg_pred)))),
               'Average across all 7 subjects')

        axes[0].set_ylabel('RPE (1-10)')
        axes[0].legend(loc='lower right', fontsize=10)
        fig.suptitle(f'{label} — per-set RPE (validation, LOSO)', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = OUT / f'rpe_per_set_{label.replace(" @ ", "_").replace(" ", "_")}.png'
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'\nWrote {out_path}')

    print(f'\nAll plots in {OUT}')


if __name__ == '__main__':
    main()
