"""Per-task evaluation metrics. References:
- Saeb et al. 2017 — subject-wise CV motivation
- Lundberg & Lee 2017 — SHAP for interpretation (used elsewhere)
"""

from __future__ import annotations
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, mean_absolute_error,
    confusion_matrix,
)
from scipy.stats import pearsonr


def compute_all_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    n_exercise: int,
    n_phase: int,
) -> Dict[str, Dict[str, float]]:
    """Compute per-task metrics from cat'd preds/targets/masks (CPU tensors)."""

    out: Dict[str, Dict[str, float]] = {}

    # --- exercise ---
    m = masks['exercise'].numpy().astype(bool)
    if m.sum() > 0:
        y_true = targets['exercise'].numpy()[m]
        y_pred = preds['exercise'].argmax(dim=-1).numpy()[m]
        out['exercise'] = {
            'f1_macro': float(f1_score(y_true, y_pred, average='macro',
                                         labels=list(range(n_exercise)),
                                         zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'n': int(m.sum()),
        }
    else:
        out['exercise'] = {'f1_macro': float('nan'),
                            'balanced_accuracy': float('nan'), 'n': 0}

    # --- phase ---
    m = masks['phase'].numpy().astype(bool)
    if m.sum() > 0:
        y_true = targets['phase'].numpy()[m]
        y_pred = preds['phase'].argmax(dim=-1).numpy()[m]
        out['phase'] = {
            'f1_macro': float(f1_score(y_true, y_pred, average='macro',
                                         labels=list(range(n_phase)),
                                         zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'n': int(m.sum()),
        }
    else:
        out['phase'] = {'f1_macro': float('nan'),
                         'balanced_accuracy': float('nan'), 'n': 0}

    # --- fatigue (regression on RPE) ---
    m = masks['fatigue'].numpy().astype(bool)
    if m.sum() > 1:
        y_true = targets['fatigue'].numpy()[m].astype(float)
        y_pred = preds['fatigue'].numpy()[m].astype(float)
        try:
            r, _ = pearsonr(y_true, y_pred)
        except Exception:
            r = float('nan')
        out['fatigue'] = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'pearson_r': float(r),
            'n': int(m.sum()),
        }
    else:
        out['fatigue'] = {'mae': float('nan'), 'pearson_r': float('nan'), 'n': 0}

    # --- reps (regression on rep_count_in_set as supervision) ---
    m = masks['reps'].numpy().astype(bool)
    if m.sum() > 0:
        y_true = targets['reps'].numpy()[m].astype(float)
        y_pred = preds['reps'].numpy()[m].astype(float)
        out['reps'] = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'n': int(m.sum()),
        }
    else:
        out['reps'] = {'mae': float('nan'), 'n': 0}

    return out


def per_subject_breakdown(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    subject_ids: np.ndarray,
    n_exercise: int,
    n_phase: int,
):
    """Returns DataFrame with per-subject metrics for catastrophic-failure detection."""
    import pandas as pd
    rows = []
    for s in np.unique(subject_ids):
        sub_mask = subject_ids == s
        sub_preds = {k: v[sub_mask] for k, v in preds.items()}
        sub_targets = {k: v[sub_mask] for k, v in targets.items()}
        sub_masks = {k: v[sub_mask] for k, v in masks.items()}
        m = compute_all_metrics(sub_preds, sub_targets, sub_masks,
                                  n_exercise=n_exercise, n_phase=n_phase)
        rows.append({
            'subject_id': s,
            'exercise_f1': m['exercise']['f1_macro'],
            'phase_f1': m['phase']['f1_macro'],
            'fatigue_mae': m['fatigue']['mae'],
            'fatigue_r': m['fatigue']['pearson_r'],
            'reps_mae': m['reps']['mae'],
        })
    return pd.DataFrame(rows)
