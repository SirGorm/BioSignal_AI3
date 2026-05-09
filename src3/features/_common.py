"""Shared helpers for src3/features/*."""

from __future__ import annotations

import numpy as np


def nanfill(x: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN. Required before IIR filtering."""
    out = np.asarray(x, dtype=float).copy()
    mask = np.isnan(out)
    if not mask.any():
        return out
    n = len(out)
    if n == 0 or mask.all():
        return out
    idx = np.arange(n)
    fp = np.maximum.accumulate(np.where(~mask, idx, 0))
    out[mask] = out[fp[mask]]
    mask2 = np.isnan(out)
    if mask2.any():
        rev = np.where(~mask2, idx, n - 1)[::-1]
        bp = np.minimum.accumulate(rev)[::-1]
        out[mask2] = out[bp[mask2]]
    return out


def windows(n: int, win: int, hop: int):
    """Yield (start, end) sample indices for sliding windows."""
    pos = 0
    while pos + win <= n:
        yield pos, pos + win
        pos += hop
