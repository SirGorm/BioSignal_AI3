"""Aggregate per-window soft rep predictions into per-set integer counts.

Soft rep target per window = number of (fractional) reps captured in the
window's time interval (see src/data/raw_window_dataset.py and
diagrams/soft_targets_visualization.png).

Because windows overlap with stride hop_s < window_s, each rep contributes
roughly window_s / hop_s windows of mass. The unbiased estimator for total
reps in a set is therefore::

    total = sum(soft_pred_w for w in windows in set) * (hop_s / window_s)

For nn.yaml defaults (window_s=2.0, hop_s=0.1) the scale is 0.05.
"""

from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np


def soft_to_set_count(
    window_preds: Sequence[float],
    hop_s: float = 0.1,
    window_s: float = 2.0,
    clip_negative: bool = True,
) -> int:
    """Sum soft per-window predictions for one set, scale, round to integer.

    Parameters
    ----------
    window_preds : sequence of model outputs (in soft-target units = reps per
        window) for windows belonging to a single active set.
    hop_s, window_s : sliding-window geometry (must match the dataset that
        produced the predictions).
    clip_negative : zero out negative predictions before summing. Networks
        sometimes produce slightly negative regression outputs at boundaries;
        keeping them would bias the count low.

    Returns
    -------
    int — predicted integer rep count for the set (round-half-to-even).
    """
    if window_s <= 0:
        raise ValueError(f"window_s must be positive, got {window_s}")
    arr = np.asarray(window_preds, dtype=float)
    if clip_negative:
        arr = np.maximum(arr, 0.0)
    scale = hop_s / window_s
    total = float(arr.sum() * scale)
    return int(round(total))


def soft_to_set_counts_grouped(
    window_preds: Sequence[float],
    set_ids: Sequence,
    hop_s: float = 0.1,
    window_s: float = 2.0,
    clip_negative: bool = True,
) -> Dict[object, int]:
    """Group window predictions by set id and aggregate each group.

    `set_ids` is a per-window identifier (e.g. (recording_id, set_number));
    windows with set_id of None or NaN are ignored (rest periods).
    """
    arr = np.asarray(window_preds, dtype=float)
    if len(arr) != len(set_ids):
        raise ValueError(
            f"length mismatch: window_preds={len(arr)} set_ids={len(set_ids)}"
        )
    grouped: Dict[object, List[float]] = {}
    for pred, sid in zip(arr.tolist(), set_ids):
        if sid is None:
            continue
        try:
            if isinstance(sid, float) and np.isnan(sid):
                continue
        except TypeError:
            pass
        grouped.setdefault(sid, []).append(pred)
    return {
        sid: soft_to_set_count(preds, hop_s=hop_s, window_s=window_s,
                                 clip_negative=clip_negative)
        for sid, preds in grouped.items()
    }
