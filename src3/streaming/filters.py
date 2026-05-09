"""Causal filter primitives — scipy.signal.sosfilt with persisted state.

Project rule: filtfilt is forbidden in real-time code (check-no-filtfilt.sh).
Each filter holds its `zi` between calls.

References
----------
- Oppenheim & Schafer 2010 — Discrete-Time Signal Processing (3rd ed.)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, tf2sos


class _SosFilter:
    """Generic SOS filter with warm-started persisted state."""

    def __init__(self, sos: np.ndarray) -> None:
        self.sos = sos
        self._zi: np.ndarray | None = None

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        if self._zi is None:
            self._zi = sosfilt_zi(self.sos) * float(x[0])
        y, self._zi = sosfilt(self.sos, x, zi=self._zi)
        return y

    def reset(self) -> None:
        self._zi = None


def causal_bandpass(low_hz: float, high_hz: float, fs: int, order: int = 4) -> _SosFilter:
    return _SosFilter(butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos"))


def causal_lowpass(cutoff_hz: float, fs: int, order: int = 4) -> _SosFilter:
    return _SosFilter(butter(order, cutoff_hz, btype="low", fs=fs, output="sos"))


def causal_notch(notch_hz: float, fs: int, Q: float = 30.0) -> _SosFilter:
    b, a = iirnotch(notch_hz, Q, fs)
    return _SosFilter(tf2sos(b, a))


class FilterChain:
    """Apply a sequence of causal filters left-to-right."""

    def __init__(self, *filters: _SosFilter) -> None:
        self._filters = filters

    def step(self, x: np.ndarray) -> np.ndarray:
        for f in self._filters:
            x = f.step(x)
        return x

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
