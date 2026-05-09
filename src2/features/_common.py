"""Shared windowing utility — pure NumPy, no dependencies."""

from __future__ import annotations

import numpy as np


def sliding_windows(
    signal: np.ndarray, window_samples: int, hop_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (windows, start_indices) where windows.shape = (N, window_samples)."""
    n = len(signal)
    if n < window_samples:
        return np.empty((0, window_samples), dtype=signal.dtype), np.empty(
            (0,), dtype=int
        )
    starts = np.arange(0, n - window_samples + 1, hop_samples, dtype=int)
    # numpy stride trick for zero-copy windows
    from numpy.lib.stride_tricks import sliding_window_view

    full = sliding_window_view(signal, window_samples)  # (n - W + 1, W)
    return full[starts], starts
