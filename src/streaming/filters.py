"""
Causal (online) filter primitives with persisted state.

All filters use sosfilt with explicit zi (filter memory) that persists
between calls. This is the only correct approach for streaming DSP —
filtfilt is explicitly forbidden in src/streaming/ by the hook
check-no-filtfilt.sh.

References
----------
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson. [IIR causal filtering with persisted state]
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0:
  Fundamental algorithms for scientific computing in Python.
  Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch


class CausalBandpass:
    """Causal Butterworth bandpass filter with persisted state.

    Oppenheim & Schafer 2010 — IIR causal filtering with zi.

    Parameters
    ----------
    low_hz  : Lower cutoff in Hz.
    high_hz : Upper cutoff in Hz.
    fs      : Sample rate in Hz.
    order   : Filter order (default 4).
    """

    def __init__(self, low_hz: float, high_hz: float, fs: int, order: int = 4) -> None:
        self.sos = butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False

    def step(self, x: np.ndarray) -> np.ndarray:
        """Filter a chunk of samples, updating internal state.

        Parameters
        ----------
        x : 1D array of new samples (any length ≥ 1).

        Returns
        -------
        Filtered samples of the same length.
        """
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        if not self._initialized:
            # Warm-start: scale initial conditions by first sample to avoid
            # startup transient (Oppenheim & Schafer 2010)
            self._zi = sosfilt_zi(self.sos) * x[0]
            self._initialized = True
        y, self._zi = sosfilt(self.sos, x, zi=self._zi)
        return y

    def reset(self) -> None:
        """Reset filter state (call at session start)."""
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False


class CausalLowpass:
    """Causal Butterworth lowpass filter with persisted state."""

    def __init__(self, cutoff_hz: float, fs: int, order: int = 4) -> None:
        self.sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        if not self._initialized:
            self._zi = sosfilt_zi(self.sos) * x[0]
            self._initialized = True
        y, self._zi = sosfilt(self.sos, x, zi=self._zi)
        return y

    def reset(self) -> None:
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False


class CausalNotch:
    """Causal IIR notch filter (50 Hz power-line) with persisted state."""

    def __init__(self, notch_hz: float, fs: int, Q: float = 30.0) -> None:
        b, a = iirnotch(notch_hz, Q, fs)
        # Convert b/a to SOS for numerical stability
        from scipy.signal import tf2sos
        self.sos = tf2sos(b, a)
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False

    def step(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        if not self._initialized:
            self._zi = sosfilt_zi(self.sos) * x[0]
            self._initialized = True
        y, self._zi = sosfilt(self.sos, x, zi=self._zi)
        return y

    def reset(self) -> None:
        self._zi = sosfilt_zi(self.sos)
        self._initialized = False


class CausalFilterChain:
    """Apply a sequence of causal filters in order."""

    def __init__(self, filters: list) -> None:
        self._filters = filters

    def step(self, x: np.ndarray) -> np.ndarray:
        for f in self._filters:
            x = f.step(x)
        return x

    def reset(self) -> None:
        for f in self._filters:
            f.reset()
