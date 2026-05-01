"""Feature relevance analysis for the strength-RT project.

Three complementary tests, all per-task:
- Fisher LDA (linear discriminant): which features separate classes linearly?
- ANOVA F-test: parametric univariate relevance
- Mutual information: nonparametric, captures nonlinear relationships

Each test is computed PER TASK (exercise, phase, fatigue, reps) since a feature
that's discriminative for exercise classification may be uninformative for
fatigue regression. Aggregated rank also produced for cases where a single
feature subset is desired across tasks.

CRITICAL — leakage warning:
  When using these scores for feature selection, the analysis MUST be run
  INSIDE each CV fold's training set, not on the full dataset. Selecting
  features on the full dataset and then doing CV reuses test-fold information
  in the selection step, inflating reported metrics (Saeb et al. 2017).
  See `select_features_within_fold()` for the safe pattern.

References:
- Fisher 1936 — original LDA
- Hastie, Tibshirani & Friedman 2009 — *The Elements of Statistical Learning*
- Saeb et al. 2017 — leakage-aware CV
- Guyon & Elisseeff 2003 — feature selection survey
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Per-task scoring functions.
# ---------------------------------------------------------------------------

def lda_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fisher LDA: |coefficient| magnitude per feature, summed across classes.

    Returns shape (n_features,) — higher = more discriminative.
    Standardizes X internally so coefficients are comparable across features.
    """
    valid = ~np.isnan(y) if y.dtype.kind == 'f' else (y >= 0)
    if valid.sum() < 5 or len(np.unique(y[valid])) < 2:
        return np.full(X.shape[1], np.nan)
    Xs = StandardScaler().fit_transform(X[valid])
    lda = LinearDiscriminantAnalysis()
    lda.fit(Xs, y[valid])
    # coef_ shape: (n_classes-1, n_features) for 2-class, (n_classes, n_features)
    # for multi-class. Take L2 norm across class axis to get per-feature score.
    return np.linalg.norm(lda.coef_, axis=0)


def anova_scores(X: np.ndarray, y: np.ndarray, task_kind: str) -> np.ndarray:
    """ANOVA F-statistic per feature.

    task_kind: 'classification' uses f_classif; 'regression' uses f_regression.
    Returns shape (n_features,) — higher F = more relevance.
    """
    valid = ~np.isnan(y) if y.dtype.kind == 'f' else (y >= 0)
    if valid.sum() < 5:
        return np.full(X.shape[1], np.nan)
    if task_kind == 'classification':
        f_vals, _ = f_classif(X[valid], y[valid])
    elif task_kind == 'regression':
        f_vals, _ = f_regression(X[valid], y[valid])
    else:
        raise ValueError(f"task_kind must be 'classification' or 'regression'")
    return np.nan_to_num(f_vals, nan=0.0)


def mutual_info_scores(X: np.ndarray, y: np.ndarray, task_kind: str,
                        random_state: int = 42) -> np.ndarray:
    """Mutual information per feature. Captures nonlinear relevance.

    task_kind: 'classification' uses mutual_info_classif;
               'regression' uses mutual_info_regression.
    Returns shape (n_features,) — higher MI = more relevance.
    """
    valid = ~np.isnan(y) if y.dtype.kind == 'f' else (y >= 0)
    if valid.sum() < 5:
        return np.full(X.shape[1], np.nan)
    if task_kind == 'classification':
        mi = mutual_info_classif(X[valid], y[valid], random_state=random_state)
    elif task_kind == 'regression':
        mi = mutual_info_regression(X[valid], y[valid], random_state=random_state)
    else:
        raise ValueError(f"task_kind must be 'classification' or 'regression'")
    return mi


# ---------------------------------------------------------------------------
# Per-task analysis: combine all three tests into a single DataFrame.
# ---------------------------------------------------------------------------

TASK_KIND = {
    'exercise': 'classification',
    'phase':    'classification',
    'fatigue':  'regression',
    'reps':     'regression',
}


def per_task_feature_table(
    X: np.ndarray,
    y_dict: Dict[str, np.ndarray],
    feature_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """For each task, returns a DataFrame with columns
       [feature, lda, anova, mutual_info, rank_lda, rank_anova, rank_mi, rank_avg]
    sorted by rank_avg ascending (1 = best).
    """
    out = {}
    for task, y in y_dict.items():
        kind = TASK_KIND[task]
        if kind == 'classification':
            lda_s = lda_scores(X, y)
        else:
            # LDA requires discrete classes; for regression we bin into
            # quartiles so LDA still says something useful as a comparator
            valid = ~np.isnan(y)
            if valid.sum() < 5:
                lda_s = np.full(X.shape[1], np.nan)
            else:
                y_binned = np.full_like(y, -1, dtype=np.int64)
                quantiles = np.nanpercentile(y[valid], [25, 50, 75])
                y_binned[valid] = np.digitize(y[valid], quantiles)
                lda_s = lda_scores(X, y_binned)

        anova_s = anova_scores(X, y, kind)
        mi_s = mutual_info_scores(X, y, kind)

        df = pd.DataFrame({
            'feature': feature_names,
            'lda': lda_s,
            'anova': anova_s,
            'mutual_info': mi_s,
        })
        # Higher is better for all three; rank descending so rank 1 = best.
        df['rank_lda'] = df['lda'].rank(ascending=False, method='min')
        df['rank_anova'] = df['anova'].rank(ascending=False, method='min')
        df['rank_mi'] = df['mutual_info'].rank(ascending=False, method='min')
        df['rank_avg'] = df[['rank_lda', 'rank_anova', 'rank_mi']].mean(axis=1)
        df = df.sort_values('rank_avg').reset_index(drop=True)
        out[task] = df
    return out


def aggregate_across_tasks(per_task: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Single combined ranking: feature scored by best rank across the 4 tasks.
    Useful when you want ONE feature subset for a multi-task model.
    """
    parts = []
    for task, df in per_task.items():
        s = df.set_index('feature')['rank_avg'].rename(f'rank_{task}')
        parts.append(s)
    combined = pd.concat(parts, axis=1)
    combined['rank_min'] = combined.min(axis=1)   # best across tasks
    combined['rank_mean'] = combined.mean(axis=1) # average across tasks
    return combined.sort_values('rank_mean').reset_index()


# ---------------------------------------------------------------------------
# Leakage-safe selection inside CV.
# ---------------------------------------------------------------------------

def select_features_within_fold(
    X_train: np.ndarray,
    y_dict_train: Dict[str, np.ndarray],
    feature_names: List[str],
    top_k: int = 30,
    aggregation: str = 'rank_min',  # or 'rank_mean'
) -> Tuple[List[str], np.ndarray]:
    """Compute feature relevance on the TRAINING fold only and return the
    top-k feature names + their indices in the original feature_names list.

    Use the returned indices to subset X_train AND X_test consistently.
    The test fold is NEVER used to compute relevance.
    """
    per_task = per_task_feature_table(X_train, y_dict_train, feature_names)
    combined = aggregate_across_tasks(per_task)
    selected_names = combined.head(top_k)['feature'].tolist()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    selected_idx = np.array([name_to_idx[n] for n in selected_names])
    return selected_names, selected_idx


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------

def write_report(
    per_task: Dict[str, pd.DataFrame],
    combined: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
):
    """Write a markdown + CSV report of per-task rankings."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-task CSVs
    for task, df in per_task.items():
        df.to_csv(out_dir / f'feature_ranking_{task}.csv', index=False)
    combined.to_csv(out_dir / 'feature_ranking_combined.csv', index=False)

    # Markdown summary
    lines = [
        "# Feature Relevance Analysis",
        "",
        "Three complementary tests run per task:",
        "- **LDA**: Fisher linear discriminant coefficient magnitude",
        "  (Fisher 1936; Hastie, Tibshirani & Friedman 2009)",
        "- **ANOVA**: F-statistic from one-way ANOVA (classification) or "
          "linear regression F-test (regression)",
        "- **Mutual information**: nonparametric, captures nonlinear "
          "relationships",
        "",
        "Higher score = more relevant. Rank 1 = best.",
        "",
        "## Top features per task",
        "",
    ]
    for task, df in per_task.items():
        lines += [f"### {task}", ""]
        head = df.head(top_n)[['feature', 'lda', 'anova', 'mutual_info',
                                  'rank_avg']]
        lines.append(head.to_markdown(index=False, floatfmt='.3f'))
        lines.append("")

    lines += [
        "## Combined ranking (across all tasks)",
        "",
        "Useful when picking a single feature subset for the multi-task model.",
        "",
        combined.head(top_n).to_markdown(index=False, floatfmt='.2f'),
        "",
        "## Methodological note",
        "",
        "These rankings were computed on the **full dataset** for exploratory "
        "analysis only. When using selected features in a model, re-run the "
        "analysis INSIDE each CV training fold to avoid leakage "
        "(Saeb et al. 2017). See `select_features_within_fold()` for the "
        "leakage-safe pattern.",
        "",
        "## References",
        "",
        "- Fisher, R. A. (1936). The use of multiple measurements in taxonomic "
          "problems. *Annals of Eugenics*, 7(2), 179–188.",
        "- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements "
          "of Statistical Learning* (2nd ed.). Springer.",
        "- Guyon, I., & Elisseeff, A. (2003). An introduction to variable "
          "and feature selection. *Journal of Machine Learning Research*, "
          "3, 1157–1182.",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, "
          "K. P. (2017). The need to approximate the use-case in clinical "
          "machine learning. *GigaScience*, 6(5), gix019.",
    ]

    (out_dir / 'feature_relevance_report.md').write_text('\n'.join(lines))
    print(f"[feature_analysis] Wrote report to {out_dir}/feature_relevance_report.md")


def write_top_k_list(combined: pd.DataFrame, out_path: Path, k: int = 30):
    """Write a JSON file with the top-k feature names — for use with
    --feature-cols flag in training scripts."""
    top = combined.head(k)['feature'].tolist()
    with open(out_path, 'w') as f:
        json.dump({'top_k': k, 'features': top}, f, indent=2)
    print(f"[feature_analysis] Top-{k} features written to {out_path}")
