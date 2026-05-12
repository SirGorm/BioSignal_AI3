"""Time-series fatigue prediction for full sessions of the best/worst recording.

Per recording, plots:
  - Actual RPE per set as a step function (one value per set, broadcast)
  - Model prediction per window (line, mean ± std across 3 seeds)
  - Mean-RPE baseline (constant)
  - Vertical lines for set boundaries

"Best" / "worst" recording chosen by mean per-set MAE on the held-out fold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.raw_window_dataset import RawMultimodalWindowDataset  # noqa: E402


def gather_per_window_long(cell_dir: Path, ds):
    """Build a long-form DataFrame of every (seed, fold)'s test window with
    pred + target + (rec, set) + dataset_idx."""
    rec_ids = np.asarray(ds.recording_ids)
    set_nums = np.asarray(ds.set_numbers)

    chunks = []
    for seed_dir in sorted(cell_dir.glob('phase2_seeds/seed_*')):
        seed = int(seed_dir.name.split('_')[-1])
        for pf in sorted(seed_dir.rglob('test_preds.pt')):
            d = torch.load(pf, map_location='cpu', weights_only=False)
            mask = d['masks']['fatigue'].bool().numpy()
            ti = d['test_idx']
            test_idx = ti.numpy() if hasattr(ti, 'numpy') else np.asarray(ti)
            pred = d['preds']['fatigue'].float().numpy()[mask]
            tgt = d['targets']['fatigue'].float().numpy()[mask]
            idx = test_idx[mask]
            chunks.append(pd.DataFrame({
                'seed': seed,
                'idx': idx,
                'rec': rec_ids[idx],
                'set': set_nums[idx].astype(int),
                'pred': pred,
                'target': tgt,
            }))
    return pd.concat(chunks, ignore_index=True)


def plot_session(rec: str, df: pd.DataFrame, mean_rpe: float, out_path: Path,
                  model_label: str):
    """One plot for one recording: x = window-order index within the
    recording, y = RPE. Overlay model preds (mean+std across seeds),
    actual RPE step, mean-baseline line, and set boundaries."""
    sub = df[df['rec'] == rec].copy()
    # Per-window mean+std across 3 seeds (each window's idx appears once per
    # seed since recording is in one fold)
    pw = sub.groupby('idx').agg(
        pred_mean=('pred', 'mean'),
        pred_std=('pred', 'std'),
        target=('target', 'first'),
        set=('set', 'first'),
    ).reset_index().sort_values('idx').reset_index(drop=True)
    if len(pw) == 0:
        print(f"  {rec}: no windows, skip")
        return None

    # Per-set metrics: mean-predicted vs actual
    per_set = sub.groupby(['seed', 'set']).agg(
        pred=('pred', 'mean'), target=('target', 'first')).reset_index()
    per_set_agg = per_set.groupby('set').agg(
        pred=('pred', 'mean'), target=('target', 'first')).reset_index()
    mae_model = float(np.abs(per_set_agg['pred'] - per_set_agg['target']).mean())
    mae_mean = float(np.abs(mean_rpe - per_set_agg['target']).mean())

    # Pearson r per recording
    if per_set_agg['target'].std() > 1e-9:
        r_model = float(np.corrcoef(per_set_agg['pred'],
                                     per_set_agg['target'])[0, 1])
    else:
        r_model = np.nan

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(pw))

    # Model predictions (line + ribbon)
    ax.plot(x, pw['pred_mean'], color='C0', linewidth=1.4,
            label=f'model pred (mean±std over 3 seeds)')
    ax.fill_between(x, pw['pred_mean'] - pw['pred_std'].fillna(0),
                    pw['pred_mean'] + pw['pred_std'].fillna(0),
                    alpha=0.25, color='C0')

    # Actual RPE per set: step (use set boundaries)
    set_changes = np.where(np.diff(pw['set'].to_numpy(), prepend=-9) != 0)[0]
    for i, start in enumerate(set_changes):
        end = set_changes[i + 1] if i + 1 < len(set_changes) else len(pw)
        s = pw['set'].iloc[start]
        tgt = pw['target'].iloc[start]
        ax.hlines(tgt, x[start], x[end - 1], color='C3',
                  linewidth=2.5, label=('actual RPE per set'
                                          if i == 0 else None))
        ax.axvline(x[start], color='gray', linewidth=0.5, alpha=0.5)
        # Set number label
        ax.text((x[start] + x[end - 1]) / 2, ax.get_ylim()[1] if False
                 else tgt + 0.15, f'set {s}',
                 ha='center', va='bottom', fontsize=7, color='gray')

    # Mean-RPE baseline
    ax.axhline(mean_rpe, color='C2', linestyle='--', linewidth=1.3,
                label=f'mean-RPE baseline = {mean_rpe:.2f}')

    ax.set_xlabel('test-window index (in held-out session)')
    ax.set_ylabel('RPE (1–10)')
    ax.set_title(
        f"{rec} — {model_label}\n"
        f"per-set MAE: model={mae_model:.3f}  mean-baseline={mae_mean:.3f}   "
        f"per-set Pearson r (model) = {r_model:.3f}   "
        f"n_sets={len(per_set_agg)}  n_windows={len(pw)}"
    )
    ax.set_ylim(0, 11)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.85, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}  (MAE model={mae_model:.3f}  baseline={mae_mean:.3f})")
    return mae_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cnn_lstm_raw')
    ap.add_argument('--mode', default='multi',
                     choices=['multi', 'fatigue', 'exercise', 'phase', 'reps'])
    ap.add_argument('--window-s', type=float, default=5.0)
    ap.add_argument('--sweep-root', type=Path,
                     default=ROOT / 'runs' / 'sweep_raw')
    ap.add_argument('--labeled-root', type=Path,
                     default=ROOT / 'data' / 'labeled')
    ap.add_argument('--out-dir', type=Path, default=None,
                     help='Default: <cell-dir>/plots/')
    args = ap.parse_args()

    cell_dir = (args.sweep_root /
                f"{args.arch}__{args.mode}__w{int(args.window_s)}s")
    model_label = f"{args.arch} / {args.mode} / w{int(args.window_s)}s"
    out_dir = args.out_dir if args.out_dir else cell_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.labeled_root.rglob('aligned_features.parquet'))
    ds = RawMultimodalWindowDataset(
        parquet_paths=files, active_only=True,
        phase_whitelist=None,
        target_modes={'reps': 'soft_overlap', 'phase': 'soft'},
        window_s=args.window_s, channels=None,
    )

    df = gather_per_window_long(cell_dir, ds)
    if df.empty:
        print("No test preds found.")
        return

    # Per-recording per-set MAE (mean over seeds)
    per_seed_set = df.groupby(['seed', 'rec', 'set']).agg(
        pred=('pred', 'mean'), target=('target', 'first')).reset_index()
    per_set = per_seed_set.groupby(['rec', 'set']).agg(
        pred=('pred', 'mean'), target=('target', 'first')).reset_index()
    per_set['abs_err'] = (per_set['pred'] - per_set['target']).abs()
    rec_mae = per_set.groupby('rec')['abs_err'].mean().sort_values()
    print("\nPer-recording per-set MAE:")
    for r, m in rec_mae.items():
        print(f"  {r}:  MAE={m:.3f}  n_sets={(per_set['rec']==r).sum()}")

    best_rec = rec_mae.idxmin()
    worst_rec = rec_mae.idxmax()
    print(f"\nBest:  {best_rec}  ({rec_mae.loc[best_rec]:.3f})")
    print(f"Worst: {worst_rec} ({rec_mae.loc[worst_rec]:.3f})")

    # Mean-RPE baseline = overall mean across all sets (subject-CV-mean
    # would differ trivially with 9 train subjects)
    mean_rpe = float(per_set['target'].mean())
    print(f"\nMean-RPE baseline (all sets): {mean_rpe:.3f}")

    plot_session(best_rec, df, mean_rpe,
                  out_dir / f'session_best_{best_rec}.png', model_label)
    plot_session(worst_rec, df, mean_rpe,
                  out_dir / f'session_worst_{worst_rec}.png', model_label)


if __name__ == '__main__':
    main()
