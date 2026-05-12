"""Per-set predicted vs actual RPE scatter for one multitask model.

Default: cnn_lstm_raw/multi/w5s (highest per-set Pearson r in sweep_raw).
Aggregates per-window predictions to per-(recording, set) means, averages
across 3 seeds, colors by recording_id, overlays y=x and metrics.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='cnn_lstm_raw')
    ap.add_argument('--window-s', type=float, default=5.0)
    ap.add_argument('--sweep-root', type=Path,
                     default=ROOT / 'runs' / 'sweep_raw')
    ap.add_argument('--labeled-root', type=Path,
                     default=ROOT / 'data' / 'labeled')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    cell_dir = args.sweep_root / f"{args.arch}__multi__w{int(args.window_s)}s"
    label = f"{args.arch} / multi / w{int(args.window_s)}s"
    out_path = (args.out or cell_dir / 'plots' / 'fatigue_scatter.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataset to map test_idx → (rec, set)
    files = sorted(args.labeled_root.rglob('aligned_features.parquet'))
    ds = RawMultimodalWindowDataset(
        parquet_paths=files, active_only=True,
        phase_whitelist=None,
        target_modes={'reps': 'soft_overlap', 'phase': 'soft'},
        window_s=args.window_s, channels=None,
    )
    rec_ids = np.asarray(ds.recording_ids)
    set_nums = np.asarray(ds.set_numbers)

    # Collect per-window preds across all (seed, fold) → tidy long-form df
    rows = []
    for seed_dir in sorted(cell_dir.glob('phase2_seeds/seed_*')):
        seed = int(seed_dir.name.split('_')[-1])
        for pf in sorted(seed_dir.rglob('test_preds.pt')):
            d = torch.load(pf, map_location='cpu', weights_only=False)
            mask = d['masks']['fatigue'].bool().numpy()
            ti = d['test_idx']
            test_idx = ti.numpy() if hasattr(ti, 'numpy') else np.asarray(ti)
            pred = d['preds']['fatigue'].float().numpy()[mask]
            tgt = d['targets']['fatigue'].float().numpy()[mask]
            r = rec_ids[test_idx[mask]]
            s = set_nums[test_idx[mask]].astype(int)
            rows.append(pd.DataFrame({'seed': seed, 'rec': r, 'set': s,
                                       'pred': pred, 'target': tgt}))
    df = pd.concat(rows, ignore_index=True)

    # First aggregate per (seed, rec, set) — mean pred across windows.
    # Then average across seeds → one point per (rec, set).
    per_seed_set = df.groupby(['seed', 'rec', 'set']).agg(
        pred=('pred', 'mean'), target=('target', 'mean')).reset_index()
    per_set = per_seed_set.groupby(['rec', 'set']).agg(
        pred=('pred', 'mean'),
        pred_std=('pred', 'std'),
        target=('target', 'first'),
    ).reset_index()

    # Metrics
    mae = float(np.abs(per_set['pred'] - per_set['target']).mean())
    r_corr = float(np.corrcoef(per_set['pred'], per_set['target'])[0, 1])
    print(f"[{label}] n_sets={len(per_set)}  per-set MAE={mae:.3f}  "
          f"Pearson r={r_corr:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    recs = sorted(per_set['rec'].unique())
    cmap = plt.get_cmap('tab10', len(recs))
    for i, rec in enumerate(recs):
        sub = per_set[per_set['rec'] == rec]
        # Tiny x-jitter so integer RPE targets don't overlap
        jit = (np.random.RandomState(int(rec[-3:]) if rec[-3:].isdigit() else i)
                .uniform(-0.15, 0.15, size=len(sub)))
        ax.errorbar(sub['target'] + jit, sub['pred'],
                    yerr=sub['pred_std'],
                    fmt='o', markersize=6, alpha=0.75,
                    color=cmap(i), ecolor=cmap(i), elinewidth=0.8,
                    capsize=2, label=rec.split('/')[-1])

    lo = min(per_set['target'].min(), per_set['pred'].min()) - 0.5
    hi = max(per_set['target'].max(), per_set['pred'].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')

    ax.set_xlabel('actual RPE (per set, 1–10)')
    ax.set_ylabel('predicted RPE (mean over windows in set)')
    ax.set_title(
        f"{label}\n"
        f"n={len(per_set)} sets across {len(recs)} recordings  "
        f"MAE={mae:.3f}   Pearson r={r_corr:.3f}\n"
        f"error bars = std across 3 seeds   x-jitter for visibility"
    )
    ax.legend(title='recording', fontsize=8, loc='lower right',
               framealpha=0.8)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 11))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"[plot] wrote {out_path}")


if __name__ == '__main__':
    main()
