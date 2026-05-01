"""Sweep top-K feature counts and plot accuracy-vs-K to find the elbow.

Trains the chosen architecture with K = 5, 10, 20, 30, 50, 75, 100, all-features
(if you have that many) and plots performance per task vs K. The elbow tells
you the smallest K that retains most of the signal.

CRITICAL: this script uses leakage-safe selection (per-fold) by default. Lazy
mode is available for quick iteration but should not be used for reported
numbers (Saeb et al. 2017).

Run:
    python scripts/sweep_top_k.py --arch tcn --ks 10 20 30 50 100
    python scripts/sweep_top_k.py --arch cnn_lstm --ks 10 20 30 50 --epochs 30

References:
- Saeb et al. 2017 — leakage-safe CV
- Guyon & Elisseeff 2003 — feature selection methodology
- Hastie, Tibshirani & Friedman 2009 — performance-vs-complexity trade-off
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data.datasets import WindowFeatureDataset
from src.eval.feature_analysis import select_features_within_fold
from src.training.cv import load_or_generate_splits
from src.training.loop import TrainConfig, train_one_fold
from scripts._common import (
    ARCH_REGISTRY, find_window_feature_files, make_model_factory,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=list(ARCH_REGISTRY), required=True)
    p.add_argument('--ks', type=int, nargs='+',
                    default=[10, 20, 30, 50, 100],
                    help='K values to sweep (top-K features)')
    p.add_argument('--seeds', type=int, nargs='+', default=[42],
                    help='Single seed by default for sweep speed; rerun '
                         'best K with multi-seed afterwards')
    p.add_argument('--run-slug', type=str, default=None)
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    return p.parse_args()


def find_elbow(ks, scores):
    """Heuristic elbow detection: smallest K where adding more features
    yields < 1% relative improvement in best-so-far. Returns recommended K."""
    ks = np.asarray(ks)
    scores = np.asarray(scores, dtype=float)
    best_so_far = np.maximum.accumulate(scores)
    improvements = np.diff(best_so_far) / np.maximum(best_so_far[:-1], 1e-9)
    # Find first k where subsequent improvements all stay < 1%
    for i in range(len(improvements)):
        if all(imp < 0.01 for imp in improvements[i:]):
            return int(ks[i])
    return int(ks[-1])  # no plateau detected; return largest tested


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slug = args.run_slug or f"sweep_k_{args.arch}"
    run_dir = args.runs_root / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] Output: {run_dir}")

    win_paths = find_window_feature_files(args.labeled_root)
    full_dataset = WindowFeatureDataset(window_parquets=win_paths,
                                         active_only=True)
    print(f"[sweep] Full dataset: {len(full_dataset)} windows × "
          f"{full_dataset.n_features} features")

    # Cap K values at the actual feature count
    ks = sorted({k for k in args.ks if k <= full_dataset.n_features})
    if full_dataset.n_features not in ks:
        ks.append(full_dataset.n_features)  # always include "all features"
        ks.sort()
    print(f"[sweep] K values: {ks}")

    subject_ids = np.array(full_dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=args.splits)

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                       lr=args.lr, weight_decay=1e-4, grad_clip=1.0,
                       patience=8, mixed_precision=True, num_workers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Pre-compute per-fold feature rankings ONCE (independent of K)
    print("[sweep] Pre-computing per-fold feature rankings...")
    X_full = full_dataset.X.numpy()
    feature_names = full_dataset.feature_cols
    per_fold_rankings = {}
    for fold in folds:
        train_idx = fold['train_idx']
        y_dict = {
            'exercise': full_dataset.t_exercise[train_idx].numpy().astype(np.int64),
            'phase':    full_dataset.t_phase[train_idx].numpy().astype(np.int64),
            'fatigue':  np.where(
                full_dataset.m_fatigue[train_idx].numpy(),
                full_dataset.t_fatigue[train_idx].numpy().astype(float),
                np.nan),
            'reps':     np.where(
                full_dataset.m_reps[train_idx].numpy(),
                full_dataset.t_reps[train_idx].numpy().astype(float),
                np.nan),
        }
        # Get full ranking once; we'll subset to top-K per K below
        names_max, _ = select_features_within_fold(
            X_full[train_idx], y_dict, feature_names,
            top_k=full_dataset.n_features,
        )
        per_fold_rankings[fold['fold']] = names_max

    # Sweep K
    results = {k: [] for k in ks}
    for k in ks:
        print(f"\n[sweep] === K = {k} ===")
        for seed in args.seeds:
            for fold in folds:
                fold_id = fold['fold']
                top_k_names = per_fold_rankings[fold_id][:k]
                sub_ds = WindowFeatureDataset(
                    window_parquets=win_paths, active_only=True,
                    feature_cols=top_k_names,
                    exercise_encoder=full_dataset.exercise_encoder,
                    phase_encoder=full_dataset.phase_encoder,
                    verbose=False,
                )
                factory = make_model_factory(
                    arch_name=args.arch,
                    n_features=sub_ds.n_features,
                    n_exercise=sub_ds.n_exercise,
                    n_phase=sub_ds.n_phase,
                )
                fold_dir = (run_dir / f"k_{k}" / f"seed_{seed}"
                             / f"fold_{fold_id}")
                _, metrics = train_one_fold(
                    model_factory=factory, dataset=sub_ds,
                    train_idx=fold['train_idx'],
                    test_idx=fold['test_idx'],
                    cfg=cfg, device=device, out_dir=fold_dir,
                    n_exercise=sub_ds.n_exercise, n_phase=sub_ds.n_phase,
                )
                results[k].append({
                    'seed': seed, 'fold': fold_id, 'metrics': metrics,
                })

    # Aggregate per K
    summary = {}
    for k, runs in results.items():
        agg = {
            'exercise_f1': np.mean([r['metrics']['exercise']['f1_macro']
                                      for r in runs]),
            'phase_f1':    np.mean([r['metrics']['phase']['f1_macro']
                                      for r in runs]),
            'fatigue_mae': np.mean([r['metrics']['fatigue']['mae']
                                      for r in runs]),
            'reps_mae':    np.mean([r['metrics']['reps']['mae']
                                      for r in runs]),
        }
        summary[k] = agg

    # Find elbow per task
    ks_arr = sorted(summary.keys())
    elbows = {
        'exercise': find_elbow(ks_arr,
                                [summary[k]['exercise_f1'] for k in ks_arr]),
        'phase':    find_elbow(ks_arr,
                                [summary[k]['phase_f1'] for k in ks_arr]),
        # For regression (lower is better), invert sign
        'fatigue':  find_elbow(ks_arr,
                                [-summary[k]['fatigue_mae'] for k in ks_arr]),
        'reps':     find_elbow(ks_arr,
                                [-summary[k]['reps_mae'] for k in ks_arr]),
    }
    recommended_k = max(elbows.values())

    # Write report
    lines = [
        f"# Top-K sweep — {args.arch}",
        "",
        f"Architecture: {args.arch}",
        f"K values tested: {ks_arr}",
        f"Seeds: {list(args.seeds)}",
        f"Folds: {len(folds)}",
        "",
        "## Performance vs K (mean across folds × seeds)",
        "",
        "| K | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |",
        "|---|-------------|----------|-------------|----------|",
    ]
    for k in ks_arr:
        s = summary[k]
        lines.append(
            f"| {k} | {s['exercise_f1']:.3f} | {s['phase_f1']:.3f} | "
            f"{s['fatigue_mae']:.3f} | {s['reps_mae']:.3f} |"
        )

    lines += [
        "",
        "## Elbow detection (smallest K with <1% subsequent improvement)",
        "",
        f"- Exercise: K={elbows['exercise']}",
        f"- Phase:    K={elbows['phase']}",
        f"- Fatigue:  K={elbows['fatigue']}",
        f"- Reps:     K={elbows['reps']}",
        "",
        f"**Recommended K = {recommended_k}** (max across tasks — keeps the "
        f"most demanding task happy).",
        "",
        "Run final multi-seed comparison:",
        "```bash",
        f"python scripts/train_with_top_k.py --arch {args.arch} \\",
        f"    --top-k {recommended_k} --leakage-safe \\",
        f"    --seeds 42 1337 7 --run-slug {args.arch}_top{recommended_k}_final",
        "```",
        "",
        "## Methodological notes",
        "",
        "- Feature selection re-run inside each CV fold (Saeb et al. 2017)",
        "- Single-seed sweep for speed; final run should use 3 seeds",
        "- Elbow heuristic: first K where all subsequent steps yield <1% "
          "relative improvement",
        "",
        "## References",
        "",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. "
          "(2017). The need to approximate the use-case in clinical machine "
          "learning. *GigaScience*, 6(5), gix019.",
        "- Guyon, I., & Elisseeff, A. (2003). An introduction to variable "
          "and feature selection. *JMLR*, 3, 1157–1182.",
        "- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements "
          "of Statistical Learning* (2nd ed.). Springer.",
    ]
    (run_dir / 'sweep_report.md').write_text('\n'.join(lines))
    (run_dir / 'sweep_summary.json').write_text(json.dumps({
        'arch': args.arch, 'ks': ks_arr, 'summary': summary,
        'elbows': elbows, 'recommended_k': recommended_k,
    }, indent=2, default=str))

    print("\n[sweep] Done. Summary:")
    print(f"  Recommended K = {recommended_k}")
    print(f"  Elbows per task: {elbows}")
    print(f"  Open {run_dir}/sweep_report.md for full report.")


if __name__ == '__main__':
    main()
