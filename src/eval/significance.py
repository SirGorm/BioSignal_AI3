"""Paired statistical tests for model comparison across CV folds.

When comparing two models trained on identical CV folds, paired tests are
appropriate (the same fold is the experimental unit). We use:

- Paired t-test for continuous metrics (MAE, F1)
- Wilcoxon signed-rank for ordinal RPE (more robust to non-normality)

CRITICAL — interpretation cautions:
- With 5 folds × 3 seeds = 15 paired observations, statistical power is
  modest. p-values should be reported with effect sizes (mean diff +/- std).
- Multiple comparisons (4 NN architectures × 4 tasks = 16 tests against
  baseline) inflate Type I error. Apply Bonferroni or report q-values
  alongside raw p-values.

References:
- Demšar 2006 — guidelines for comparing classifiers across multiple datasets
- Wilcoxon 1945 — original signed-rank test
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def paired_test(
    a: List[float],
    b: List[float],
    test: str = 'auto',
    alternative: str = 'two-sided',
) -> Dict[str, float]:
    """Paired comparison of two models on identical folds.

    a, b: per-fold metric values, same order, same length.
    test: 'ttest', 'wilcoxon', or 'auto' (Wilcoxon if n<10 or non-normal).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired test needs equal-length arrays: {len(a)} vs {len(b)}")
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    if n < 3:
        return {'n': n, 'p_value': float('nan'), 'test': 'insufficient_n',
                'mean_diff': float('nan'), 'effect_size': float('nan')}

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else float('nan')

    if test == 'auto':
        # Use Wilcoxon if n small or Shapiro-Wilk rejects normality of diff
        if n < 10:
            test = 'wilcoxon'
        else:
            try:
                _, p_norm = stats.shapiro(diff)
                test = 'wilcoxon' if p_norm < 0.05 else 'ttest'
            except Exception:
                test = 'wilcoxon'

    try:
        if test == 'ttest':
            stat, p = stats.ttest_rel(a, b, alternative=alternative)
        elif test == 'wilcoxon':
            stat, p = stats.wilcoxon(a, b, alternative=alternative,
                                      zero_method='zsplit')
        else:
            raise ValueError(f"Unknown test: {test}")
    except Exception as e:
        return {'n': n, 'p_value': float('nan'), 'test': f'failed: {e}',
                'mean_diff': mean_diff, 'effect_size': cohens_d}

    return {
        'n': n,
        'test': test,
        'statistic': float(stat),
        'p_value': float(p),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'cohens_d': cohens_d,
    }


def bonferroni_correction(p_values: List[float], n_comparisons: int = None) -> List[float]:
    """Adjust p-values for multiple comparisons. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    if n_comparisons is None:
        n_comparisons = (~np.isnan(p)).sum()
    return list(np.minimum(p * n_comparisons, 1.0))


def compare_models_across_tasks(
    fold_metrics: Dict[str, Dict[str, List[float]]],
    reference_model: str,
) -> pd.DataFrame:
    """Compare every model against the reference model across tasks.

    fold_metrics: {model_name: {task_metric: [per-fold values]}}
                  e.g. {'lgbm': {'exercise_f1': [0.85, 0.86, ...], ...},
                        'tcn':  {'exercise_f1': [0.88, 0.89, ...], ...}}
    reference_model: key into fold_metrics; everyone else is compared to it.

    Returns a DataFrame with one row per (model, task_metric) pair.
    """
    if reference_model not in fold_metrics:
        raise KeyError(f"reference_model '{reference_model}' not in fold_metrics")
    ref = fold_metrics[reference_model]
    rows = []
    for model, metrics in fold_metrics.items():
        if model == reference_model:
            continue
        for task_metric, vals in metrics.items():
            if task_metric not in ref:
                continue
            res = paired_test(vals, ref[task_metric])
            rows.append({
                'model': model,
                'reference': reference_model,
                'task_metric': task_metric,
                **res,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['p_bonferroni'] = bonferroni_correction(df['p_value'].tolist())
        df['significant_05'] = df['p_bonferroni'] < 0.05
    return df


def render_significance_table(df: pd.DataFrame) -> str:
    """Render the comparison DataFrame as a markdown table."""
    if df.empty:
        return "_(no comparisons available)_"
    lines = [
        "| Model | vs | Task/Metric | n | Mean Δ | Cohen's d | p (raw) | p (Bonf) | Sig? |",
        "|-------|-----|-------------|---|--------|-----------|---------|----------|------|",
    ]
    for _, r in df.iterrows():
        sig = '✓' if r.get('significant_05') else '–'
        lines.append(
            f"| {r['model']} | {r['reference']} | {r['task_metric']} | "
            f"{r['n']} | {r['mean_diff']:+.3f} | "
            f"{r.get('cohens_d', float('nan')):.2f} | "
            f"{r['p_value']:.4f} | {r['p_bonferroni']:.4f} | {sig} |"
        )
    return '\n'.join(lines)
