"""Comprehensive comparison: aggregates all NN runs (full + top-K + raw if exists)
plus baselines (LightGBM, XGBoost) into one big report.

Reads `cv_summary.json` from each architecture run directory and produces:
- Master comparison table (all models × all tasks)
- Statistical significance vs LightGBM and vs XGBoost (paired tests)
- Per-modality ablation summary if those runs exist
- Latency ranking
- Best-per-task recommendation

Usage:
    python scripts/compare_all.py \\
        --baseline-run runs/<lgbm_xgb_run>/ \\
        --full-feature-runs runs/<ts>_cnn1d-full runs/<ts>_lstm-full ... \\
        --topk-runs runs/<ts>_cnn1d-top30 runs/<ts>_lstm-top30 ... \\
        --raw-runs runs/<ts>_cnn1d-raw runs/<ts>_lstm-raw ...  # optional
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.eval.significance import (
    compare_models_across_tasks, render_significance_table,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-run', type=Path, required=True,
                    help='Run dir from /train (LightGBM + XGBoost baselines)')
    p.add_argument('--full-feature-runs', type=Path, nargs='+', default=[],
                    help='NN runs trained on ALL features')
    p.add_argument('--topk-runs', type=Path, nargs='+', default=[],
                    help='NN runs trained on TOP-K features')
    p.add_argument('--raw-runs', type=Path, nargs='+', default=[],
                    help='NN runs trained on RAW signals (Phase 2, optional)')
    p.add_argument('--ablation-runs', type=Path, nargs='+', default=[],
                    help='Per-modality ablation runs (optional)')
    p.add_argument('--top-k', type=int, default=None,
                    help='K used in topk runs (for labelling only)')
    p.add_argument('--output-slug', type=str, default='all-comparison')
    return p.parse_args()


def load_cv_summary(run_dir: Path) -> Optional[Dict]:
    """Find <run_dir>/<arch>/cv_summary.json. Returns None if missing."""
    candidates = list(run_dir.glob('*/cv_summary.json'))
    if not candidates:
        print(f"[compare_all] WARNING: no cv_summary.json under {run_dir}")
        return None
    return json.loads(candidates[0].read_text())


def load_baseline(baseline_run: Path) -> Dict[str, Optional[Dict]]:
    """Load both LightGBM and XGBoost metrics + per-fold values from baseline run."""
    out = {'lgbm': None, 'xgb': None}
    for name in ('lgbm', 'xgb'):
        for cand in (baseline_run / name / 'cv_summary.json',
                      baseline_run / name / 'metrics.json',
                      baseline_run / f'metrics_{name}.json'):
            if cand.exists():
                out[name] = json.loads(cand.read_text())
                break
        if out[name] is None:
            print(f"[compare_all] WARNING: {name} metrics not found under {baseline_run}")
    return out


def extract_summary_metric(summary: Dict, task: str, metric: str) -> Dict:
    """Pull mean/std for a (task, metric) pair. Returns {'mean':..., 'std':...}."""
    if summary is None:
        return {'mean': float('nan'), 'std': float('nan'), 'n': 0}
    s = summary.get('summary', summary).get(task, {}).get(metric, {})
    if isinstance(s, dict) and 'mean' in s:
        return s
    if isinstance(s, (int, float)):
        return {'mean': float(s), 'std': float('nan'), 'n': 1}
    return {'mean': float('nan'), 'std': float('nan'), 'n': 0}


def extract_fold_values(summary: Dict, task: str, metric: str) -> List[float]:
    """Pull per-fold values from a summary's 'all_results' list (for paired tests).

    NN summaries contain 3 seeds × 5 folds = 15 entries; LightGBM has 5 folds.
    Aggregate seeds → per-fold means so paired tests get equal-length arrays.
    """
    if summary is None:
        return []
    all_res = summary.get('all_results', [])
    by_fold: Dict[int, List[float]] = {}
    for r in all_res:
        m = r.get('metrics', {}).get(task, {}).get(metric)
        if m is None or (isinstance(m, float) and np.isnan(m)):
            continue
        fold = int(r.get('fold', len(by_fold)))
        by_fold.setdefault(fold, []).append(float(m))
    return [float(np.mean(vals)) for _, vals in sorted(by_fold.items())]


def fmt(d: Dict) -> str:
    if d is None or d.get('n', 0) == 0:
        return '—'
    m, s = d['mean'], d.get('std', float('nan'))
    if np.isnan(s):
        return f"{m:.3f}"
    return f"{m:.3f} ± {s:.3f}"


def collect_models(args) -> Dict[str, Dict]:
    """Build a dict of {label: cv_summary_dict} for all loaded runs."""
    out: Dict[str, Dict] = {}

    baselines = load_baseline(args.baseline_run)
    if baselines['lgbm']:
        out['LightGBM'] = baselines['lgbm']
    if baselines['xgb']:
        out['XGBoost'] = baselines['xgb']

    for run_dir in args.full_feature_runs:
        s = load_cv_summary(run_dir)
        if s:
            arch = s.get('arch', run_dir.name)
            out[f'{arch}_full'] = s

    k_label = f'top{args.top_k}' if args.top_k else 'topK'
    for run_dir in args.topk_runs:
        s = load_cv_summary(run_dir)
        if s:
            arch = s.get('arch', run_dir.name)
            out[f'{arch}_{k_label}'] = s

    for run_dir in args.raw_runs:
        s = load_cv_summary(run_dir)
        if s:
            arch = s.get('arch', run_dir.name)
            out[f'{arch}_raw'] = s

    for run_dir in args.ablation_runs:
        s = load_cv_summary(run_dir)
        if s:
            arch = s.get('arch', run_dir.name)
            out[f'{arch}_ablation_{run_dir.name}'] = s

    print(f"[compare_all] Loaded {len(out)} models: {list(out.keys())}")
    return out


def build_main_table(models: Dict[str, Dict]) -> str:
    """Render the master comparison table as markdown."""
    tasks_metrics = [
        ('exercise', 'f1_macro', 'Exercise F1'),
        ('exercise', 'balanced_accuracy', 'Exercise Bal-Acc'),
        ('phase',    'f1_macro', 'Phase F1'),
        ('phase',    'balanced_accuracy', 'Phase Bal-Acc'),
        ('fatigue',  'mae', 'Fatigue MAE'),
        ('fatigue',  'pearson_r', 'Fatigue Pearson r'),
        ('reps',     'mae', 'Reps MAE'),
    ]

    header_cells = ['Task / Metric'] + list(models.keys())
    sep = ['---'] * len(header_cells)
    lines = ['| ' + ' | '.join(header_cells) + ' |',
             '| ' + ' | '.join(sep) + ' |']

    for task, metric, label in tasks_metrics:
        row = [label]
        for name, summary in models.items():
            d = extract_summary_metric(summary, task, metric)
            row.append(fmt(d))
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def build_significance_section(models: Dict[str, Dict]) -> str:
    """Paired comparisons vs LightGBM and vs XGBoost."""
    fold_metrics: Dict[str, Dict[str, List[float]]] = {}
    metrics_to_test = [
        ('exercise', 'f1_macro'),
        ('phase',    'f1_macro'),
        ('fatigue',  'mae'),
        ('reps',     'mae'),
    ]
    for name, summary in models.items():
        if summary is None:
            continue
        m: Dict[str, List[float]] = {}
        for task, metric in metrics_to_test:
            vals = extract_fold_values(summary, task, metric)
            if vals:
                m[f'{task}_{metric}'] = vals
        if m:
            fold_metrics[name] = m

    parts = ["", "## Statistical significance (paired tests across folds × seeds)", ""]

    if 'LightGBM' in fold_metrics:
        df = compare_models_across_tasks(fold_metrics, reference_model='LightGBM')
        parts += ["### vs LightGBM", "", render_significance_table(df), ""]

    if 'XGBoost' in fold_metrics:
        df = compare_models_across_tasks(fold_metrics, reference_model='XGBoost')
        parts += ["### vs XGBoost", "", render_significance_table(df), ""]

    parts += [
        "Notes:",
        "- Test auto-selected: paired t-test (n≥10, normal residuals) or Wilcoxon "
          "signed-rank otherwise.",
        "- p (Bonf) is Bonferroni-adjusted for the number of comparisons in this "
          "table. Significance threshold = 0.05.",
        "- Cohen's d gives effect size; |d|>0.5 = medium, |d|>0.8 = large.",
        "- With small n, lack of significance does NOT mean equivalence.",
        "",
    ]
    return '\n'.join(parts)


def best_per_task(models: Dict[str, Dict]) -> str:
    """For each task, identify the best model."""
    tasks = [
        ('exercise', 'f1_macro', 'higher_better'),
        ('phase',    'f1_macro', 'higher_better'),
        ('fatigue',  'mae',      'lower_better'),
        ('reps',     'mae',      'lower_better'),
    ]
    lines = ["## Best model per task", "",
             "| Task | Best model | Metric value |",
             "|------|------------|--------------|"]
    for task, metric, direction in tasks:
        best_name, best_val = None, None
        for name, summary in models.items():
            d = extract_summary_metric(summary, task, metric)
            if d.get('n', 0) == 0 or np.isnan(d['mean']):
                continue
            if best_val is None:
                best_name, best_val = name, d['mean']
            elif (direction == 'higher_better' and d['mean'] > best_val) or \
                 (direction == 'lower_better' and d['mean'] < best_val):
                best_name, best_val = name, d['mean']
        if best_name:
            lines.append(f"| {task} | **{best_name}** | {best_val:.3f} ({metric}) |")
        else:
            lines.append(f"| {task} | — | — |")
    return '\n'.join(lines)


def main():
    args = parse_args()
    models = collect_models(args)
    if not models:
        raise SystemExit("No models loaded. Provide at least one of "
                          "--full-feature-runs / --topk-runs / --raw-runs.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('runs') / f"{timestamp}_{args.output_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_table = build_main_table(models)
    significance = build_significance_section(models)
    best = best_per_task(models)

    md_lines = [
        f"# Comprehensive Comparison — {timestamp}",
        "",
        f"Models compared: {len(models)}",
        f"  - Baselines: {sum(1 for k in models if k in ('LightGBM', 'XGBoost'))}",
        f"  - NN full-features: {sum(1 for k in models if k.endswith('_full'))}",
        f"  - NN top-K: {sum(1 for k in models if '_top' in k)}",
        f"  - NN raw signals: {sum(1 for k in models if k.endswith('_raw'))}",
        f"  - Ablation runs: {sum(1 for k in models if '_ablation_' in k)}",
        "",
        "## Master comparison table",
        "",
        main_table,
        "",
        best,
        "",
        significance,
        "",
        "## References",
        "- Ke, G., et al. (2017). LightGBM. *NeurIPS*, 30.",
        "- Chen, T., & Guestrin, C. (2016). XGBoost. *KDD*, 785–794.",
        "- Saeb, S., et al. (2017). The need to approximate the use-case in "
          "clinical machine learning. *GigaScience*, 6(5), gix019.",
        "- Demšar, J. (2006). Statistical comparisons of classifiers over "
          "multiple data sets. *JMLR*, 7, 1–30.",
        "- Wilcoxon, F. (1945). Individual comparisons by ranking methods. "
          "*Biometrics Bulletin*, 1(6), 80–83.",
    ]
    (out_dir / 'comparison.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"[compare_all] Wrote {out_dir / 'comparison.md'}")
    print(f"[compare_all] Done.")


if __name__ == '__main__':
    main()
