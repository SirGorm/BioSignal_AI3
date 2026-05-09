"""Per-task evaluation metrics. References:
- Saeb et al. 2017 — subject-wise CV motivation
- Lundberg & Lee 2017 — SHAP for interpretation (used elsewhere)
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, mean_absolute_error,
    confusion_matrix,
)
from scipy.stats import pearsonr


VALID_EXERCISE_AGGREGATIONS = ('per_window', 'per_set', 'both')


def per_set_exercise_metrics(
    window_logits: np.ndarray,
    valid_mask: np.ndarray,
    set_keys: np.ndarray,
    y_true_per_window: np.ndarray,
    n_exercise: int,
) -> Dict[str, float]:
    """Aggregate per-window exercise predictions to one prediction per set.

    Mirrors the RPE labeling pattern: each set has one canonical exercise
    label broadcast onto every window of the set. We aggregate the model's
    per-window class probabilities by mean-softmax over all valid windows in
    the set, then take argmax. Mean-softmax is used over majority-vote of
    per-window argmax because it is more stable when a few boundary windows
    pick the wrong class.

    Parameters
    ----------
    window_logits : (N, n_classes) float — raw exercise head logits.
    valid_mask    : (N,) bool — True where the per-window exercise label is
        valid (i.e. dataset.m_exercise). Rest-period windows are False and
        excluded from aggregation.
    set_keys      : (N,) object/str — per-window set identifier; windows
        with the same key are aggregated together. Use whatever encoding
        groups one set together (e.g. f"{recording_id}__{set_number}").
        Pass key="" or None for windows that should be skipped.
    y_true_per_window : (N,) int — exercise target indices (encoded). The
        per-set ground-truth is the first valid label encountered for that
        set (labels are broadcast within a set, so any valid window works).
    n_exercise    : number of classes — used for f1_score `labels=` arg so
        that classes absent from a fold still count toward macro F1.

    Returns
    -------
    dict with keys: f1_macro, balanced_accuracy, n_sets.
    NaN metrics + n_sets=0 if no sets can be evaluated.
    """
    if window_logits.ndim != 2:
        raise ValueError(
            f"window_logits must be 2D (N, n_classes); got shape {window_logits.shape}"
        )
    if not (len(window_logits) == len(valid_mask) == len(set_keys)
            == len(y_true_per_window)):
        raise ValueError(
            "Length mismatch: "
            f"logits={len(window_logits)} mask={len(valid_mask)} "
            f"keys={len(set_keys)} y_true={len(y_true_per_window)}"
        )

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.sum() == 0:
        return {'f1_macro': float('nan'),
                'balanced_accuracy': float('nan'),
                'n_sets': 0}

    # Stable softmax along class axis.
    logits = np.asarray(window_logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    keys_arr = np.asarray(set_keys, dtype=object)
    y_true = np.asarray(y_true_per_window)

    set_to_probs: Dict[object, list] = {}
    set_to_label: Dict[object, int] = {}
    for i in np.flatnonzero(valid_mask):
        k = keys_arr[i]
        if k is None or (isinstance(k, str) and k == ''):
            continue
        try:
            if isinstance(k, float) and np.isnan(k):
                continue
        except TypeError:
            pass
        set_to_probs.setdefault(k, []).append(probs[i])
        # Take the first valid label as the set's canonical label
        # (labels are broadcast within a set; if they ever disagree, prefer
        # the earlier one).
        if k not in set_to_label:
            lab = int(y_true[i])
            if lab >= 0:
                set_to_label[k] = lab

    pred_y, true_y = [], []
    for k, lst in set_to_probs.items():
        if k not in set_to_label:
            continue
        mean_p = np.mean(np.stack(lst, axis=0), axis=0)
        pred_y.append(int(np.argmax(mean_p)))
        true_y.append(set_to_label[k])

    if not pred_y:
        return {'f1_macro': float('nan'),
                'balanced_accuracy': float('nan'),
                'n_sets': 0}

    pred_y = np.asarray(pred_y)
    true_y = np.asarray(true_y)
    return {
        'f1_macro': float(f1_score(true_y, pred_y, average='macro',
                                     labels=list(range(n_exercise)),
                                     zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(true_y, pred_y)),
        'n_sets': len(pred_y),
    }


def compute_all_metrics(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    n_exercise: int,
    n_phase: int,
    set_keys: Optional[np.ndarray] = None,
    exercise_aggregation: str = 'per_window',
) -> Dict[str, Dict[str, float]]:
    """Compute per-task metrics from cat'd preds/targets/masks (CPU tensors).

    Exercise aggregation modes (controlled by ``exercise_aggregation``):
      - 'per_window' (default, legacy): one prediction per window. Backwards
        compatible — ``set_keys`` is ignored.
      - 'per_set'   : one prediction per (recording, set) — mean-softmax
        aggregation across all valid windows in the set. Replaces the
        ``out['exercise']`` block. Requires ``set_keys`` (length-N array
        aligned with ``preds['exercise']``).
      - 'both'      : keep per-window in ``out['exercise']`` and add per-set
        results under ``out['exercise_per_set']``. Useful for direct
        comparison in a thesis table without retraining.
    """
    if exercise_aggregation not in VALID_EXERCISE_AGGREGATIONS:
        raise ValueError(
            f"exercise_aggregation must be one of {VALID_EXERCISE_AGGREGATIONS}; "
            f"got {exercise_aggregation!r}"
        )
    if exercise_aggregation != 'per_window' and set_keys is None:
        raise ValueError(
            f"exercise_aggregation={exercise_aggregation!r} requires set_keys "
            "(per-window (recording, set) identifier). Pass set_keys=None to "
            "use the default 'per_window' mode."
        )

    out: Dict[str, Dict[str, float]] = {}

    # --- exercise ---
    m = masks['exercise'].numpy().astype(bool)
    if m.sum() > 0:
        y_true_full = targets['exercise'].numpy()
        logits_full = preds['exercise'].numpy()
        y_pred = logits_full.argmax(axis=-1)[m]
        per_window = {
            'f1_macro': float(f1_score(y_true_full[m], y_pred, average='macro',
                                         labels=list(range(n_exercise)),
                                         zero_division=0)),
            'balanced_accuracy': float(
                balanced_accuracy_score(y_true_full[m], y_pred)),
            'n': int(m.sum()),
        }
    else:
        y_true_full = targets['exercise'].numpy()
        logits_full = preds['exercise'].numpy()
        per_window = {'f1_macro': float('nan'),
                       'balanced_accuracy': float('nan'), 'n': 0}

    if exercise_aggregation == 'per_window':
        out['exercise'] = per_window
    else:
        per_set = per_set_exercise_metrics(
            window_logits=logits_full,
            valid_mask=m,
            set_keys=set_keys,
            y_true_per_window=y_true_full,
            n_exercise=n_exercise,
        )
        if exercise_aggregation == 'per_set':
            out['exercise'] = per_set
        else:  # 'both'
            out['exercise'] = per_window
            out['exercise_per_set'] = per_set

    # --- phase ---
    # Supports both hard (long-index target) and soft (probability-vector
    # target). For soft, take argmax of the target to recover discrete labels
    # so that F1 / balanced-accuracy stay comparable across hard/soft runs.
    m = masks['phase'].numpy().astype(bool)
    if m.sum() > 0:
        target_phase = targets['phase']
        if target_phase.dim() == 2:
            y_true = target_phase.argmax(dim=-1).numpy()[m]
        else:
            y_true = target_phase.numpy()[m]
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


def per_set_rep_count_metrics(
    window_preds: np.ndarray,
    set_ids: np.ndarray,
    true_rep_counts_per_set: Dict[object, int],
    hop_s: float = 0.1,
    window_s: float = 2.0,
) -> Dict[str, float]:
    """Per-set integer rep-count metrics for the soft_window training mode.

    Aggregates the model's per-window soft predictions into one integer per
    set (see src/eval/rep_aggregation.py) and compares to the ground-truth
    rep count derived from rep_markers / joint angles.

    Returns
    -------
    dict with:
        set_count_mae             — mean |pred - true| over evaluated sets
        set_count_off_by_one_rate — fraction of sets with |pred - true| <= 1
        n_sets                    — number of sets evaluated

    Returns NaN metrics if no sets can be matched.
    """
    from src.eval.rep_aggregation import soft_to_set_counts_grouped

    pred_counts = soft_to_set_counts_grouped(
        window_preds, set_ids, hop_s=hop_s, window_s=window_s,
    )
    pairs = [(pred_counts[sid], true_rep_counts_per_set[sid])
             for sid in pred_counts
             if sid in true_rep_counts_per_set]
    if not pairs:
        return {'set_count_mae': float('nan'),
                'set_count_off_by_one_rate': float('nan'),
                'n_sets': 0}
    diffs = np.abs(np.array([p - t for p, t in pairs], dtype=float))
    return {
        'set_count_mae': float(diffs.mean()),
        'set_count_off_by_one_rate': float((diffs <= 1).mean()),
        'n_sets': len(pairs),
    }


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
