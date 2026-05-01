"""Exploratory feature relevance analysis (NOT used inside training).

Loads window_features.parquet, computes LDA / ANOVA / mutual info per task on
the full dataset, and writes a report to runs/<timestamp>_feature-analysis/.

This is for human inspection — to see which features dominate, sanity-check
feature engineering, and pick a top-K shortlist for ablation.

For leakage-safe feature selection inside the CV loop, the pipeline must
re-run the analysis per fold via select_features_within_fold(). See
scripts/train_with_top_k.py.

Run:
    python scripts/analyze_features.py
    python scripts/analyze_features.py --top-k 30 --top-n 20
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.datasets import WindowFeatureDataset
from src.eval.feature_analysis import (
    per_task_feature_table, aggregate_across_tasks,
    write_report, write_top_k_list,
)
from scripts._common import find_window_feature_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--top-k', type=int, default=30,
                    help='Number of top features to write to top_k.json')
    p.add_argument('--top-n', type=int, default=20,
                    help='How many features to show in the per-task tables')
    args = p.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.runs_root / f"{timestamp}_feature-analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] Output: {out_dir}")

    win_paths = find_window_feature_files(args.labeled_root)
    dataset = WindowFeatureDataset(window_parquets=win_paths, active_only=True)
    print(f"[analyze] {len(dataset)} windows × {dataset.n_features} features")

    X = dataset.X.numpy()
    y_dict = {
        'exercise': dataset.t_exercise.numpy().astype(np.int64),
        'phase':    dataset.t_phase.numpy().astype(np.int64),
        'fatigue':  np.where(dataset.m_fatigue.numpy(),
                              dataset.t_fatigue.numpy().astype(float),
                              np.nan),
        'reps':     np.where(dataset.m_reps.numpy(),
                              dataset.t_reps.numpy().astype(float),
                              np.nan),
    }
    # Mask invalid classification labels as NaN-equivalent (-1)
    for k in ('exercise', 'phase'):
        y_dict[k] = np.where(y_dict[k] >= 0, y_dict[k], -1).astype(np.int64)

    print(f"[analyze] Computing LDA, ANOVA, mutual info per task...")
    per_task = per_task_feature_table(X, y_dict, dataset.feature_cols)
    combined = aggregate_across_tasks(per_task)

    write_report(per_task, combined, out_dir, top_n=args.top_n)
    write_top_k_list(combined, out_dir / f'top_{args.top_k}_features.json',
                       k=args.top_k)

    print(f"\n[analyze] Top {min(10, args.top_k)} features (combined ranking):")
    for i, row in combined.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:<40s}  rank_min={row['rank_min']:.0f}  "
              f"rank_mean={row['rank_mean']:.1f}")

    print(f"\n[analyze] Open {out_dir}/feature_relevance_report.md for full report.")
    print(f"          Use {out_dir}/top_{args.top_k}_features.json with "
          f"scripts/train_with_top_k.py to retrain on the shortlist.")


if __name__ == '__main__':
    main()
