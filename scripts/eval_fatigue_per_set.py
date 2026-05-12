"""Re-evaluate fatigue (RPE) per-set from existing test_preds.pt files.

The training loss + reported MAE are per-window (every window inside a set
shares the same RPE label). With 108 sets but ~10–30k windows per recording,
per-window MAE under-states the real difficulty. This script aggregates each
fold's per-window predictions into one prediction per (recording, set) via
mean, then computes MAE + Pearson r at the set level.

Default scope: all (arch, multi, window) cells under runs/sweep_raw/.
Output: results/sweep_raw_fatigue_per_set.csv (one row per arch×window×seed
+ aggregate-across-seeds rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.raw_window_dataset import RawMultimodalWindowDataset  # noqa: E402


def build_dataset(window_s: float, labeled_root: Path):
    files = sorted(labeled_root.rglob('aligned_features.parquet'))
    target_modes = {'reps': 'soft_overlap', 'phase': 'soft'}
    return RawMultimodalWindowDataset(
        parquet_paths=files, active_only=True,
        phase_whitelist=None, target_modes=target_modes,
        window_s=window_s, channels=None,
    )


def per_set_metrics(rec_ids, set_nums, pred, target):
    df = pd.DataFrame({
        'rec': rec_ids, 'set': set_nums.astype(int),
        'pred': pred, 'target': target,
    })
    per_set = df.groupby(['rec', 'set']).agg(
        pred=('pred', 'mean'),
        target=('target', 'mean'),
        n_windows=('pred', 'size'),
    ).reset_index()
    mae = float(np.abs(per_set['pred'] - per_set['target']).mean())
    if per_set['target'].std() < 1e-9 or per_set['pred'].std() < 1e-9:
        r = np.nan
    else:
        r = float(np.corrcoef(per_set['pred'], per_set['target'])[0, 1])
    return mae, r, len(per_set)


def eval_cell(arch: str, window_s: float, sweep_root: Path,
              labeled_root: Path):
    """Return list of dicts: one per seed, aggregating across all folds."""
    cell_dir = sweep_root / f"{arch}__multi__w{int(window_s)}s"
    if not cell_dir.exists():
        return []

    ds = build_dataset(window_s, labeled_root)
    rec_ids = np.asarray(ds.recording_ids)
    set_nums = np.asarray(ds.set_numbers)

    rows = []
    for seed_dir in sorted(cell_dir.glob('phase2_seeds/seed_*')):
        seed = int(seed_dir.name.split('_')[-1])
        pred_files = sorted(seed_dir.rglob('test_preds.pt'))
        if not pred_files:
            continue

        all_pred, all_tgt, all_rec, all_set = [], [], [], []
        for pf in pred_files:
            d = torch.load(pf, map_location='cpu', weights_only=False)
            fat_pred = d['preds']['fatigue'].float().numpy()
            fat_tgt = d['targets']['fatigue'].float().numpy()
            fat_mask = d['masks']['fatigue'].bool().numpy()
            ti = d['test_idx']
            test_idx = ti.numpy() if hasattr(ti, 'numpy') else np.asarray(ti)

            valid = fat_mask
            all_pred.append(fat_pred[valid])
            all_tgt.append(fat_tgt[valid])
            all_rec.append(rec_ids[test_idx[valid]])
            all_set.append(set_nums[test_idx[valid]])

        if not all_pred:
            continue
        mae, r, n_sets = per_set_metrics(
            np.concatenate(all_rec), np.concatenate(all_set),
            np.concatenate(all_pred), np.concatenate(all_tgt))
        # Also per-window for sanity-check
        pw = np.concatenate(all_pred)
        tw = np.concatenate(all_tgt)
        pw_mae = float(np.abs(pw - tw).mean())
        if pw.std() > 1e-9:
            pw_r = float(np.corrcoef(pw, tw)[0, 1])
        else:
            pw_r = np.nan

        rows.append({
            'arch': arch, 'window_s': window_s, 'seed': seed,
            'per_set_mae': mae, 'per_set_r': r, 'n_sets': n_sets,
            'per_window_mae': pw_mae, 'per_window_r': pw_r,
            'n_windows': int(len(pw)),
        })
        print(f"  {arch}/w{int(window_s)}s/seed{seed}: "
              f"per_set MAE={mae:.3f} r={r:.3f} (n={n_sets} sets)   "
              f"per_window MAE={pw_mae:.3f} r={pw_r:.3f} (n={len(pw)} win)")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-root', type=Path,
                     default=ROOT / 'runs' / 'sweep_raw')
    ap.add_argument('--labeled-root', type=Path,
                     default=ROOT / 'data' / 'labeled')
    ap.add_argument('--archs', nargs='+',
                     default=['cnn1d_raw', 'tcn_raw', 'cnn_lstm_raw'])
    ap.add_argument('--windows', type=float, nargs='+',
                     default=[1.0, 2.0, 5.0])
    ap.add_argument('--out-csv', type=Path,
                     default=ROOT / 'results' /
                             'sweep_raw_fatigue_per_set.csv')
    args = ap.parse_args()

    all_rows = []
    for arch in args.archs:
        for w in args.windows:
            print(f"\n=== {arch} / multi / w{int(w)}s ===")
            all_rows.extend(eval_cell(arch, w, args.sweep_root,
                                       args.labeled_root))

    if not all_rows:
        print("No predictions found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[csv] wrote {args.out_csv}  ({len(df)} rows)")

    print("\n=== Mean across 3 seeds per (arch, window) ===")
    agg = df.groupby(['arch', 'window_s']).agg(
        per_set_mae_mean=('per_set_mae', 'mean'),
        per_set_mae_std=('per_set_mae', 'std'),
        per_set_r_mean=('per_set_r', 'mean'),
        per_set_r_std=('per_set_r', 'std'),
        per_window_mae_mean=('per_window_mae', 'mean'),
        per_window_r_mean=('per_window_r', 'mean'),
        n_sets=('n_sets', 'first'),
    ).reset_index().sort_values('per_set_mae_mean')
    print(agg.to_string(index=False, float_format='%.3f'))


if __name__ == '__main__':
    main()
