"""
Causal sliding window buffer.

Maintains a fixed-length sample buffer and emits complete windows when the hop
condition is met. All samples are stored in chronological order with no lookahead.

References
----------
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SlidingWindowBuffer:
    """Causal sliding window with configurable size and hop.

    Parameters
    ----------
    size_samples : Window length in samples.
    hop_samples  : Hop between successive windows in samples.
    """

    def __init__(self, size_samples: int, hop_samples: int) -> None:
        if size_samples < 1:
            raise ValueError("size_samples must be >= 1")
        if hop_samples < 1:
            raise ValueError("hop_samples must be >= 1")
        self.size = size_samples
        self.hop = hop_samples
        self._buffer: deque = deque(maxlen=size_samples)
        self._samples_since_emit = 0

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        """Push new samples and return any complete windows ready for processing.

        Parameters
        ----------
        x : 1D array of new samples (any length).

        Returns
        -------
        List of np.ndarray windows (each of length size_samples).
        """
        out = []
        for v in np.atleast_1d(x):
            self._buffer.append(float(v))
            self._samples_since_emit += 1
            if (
                len(self._buffer) == self.size
                and self._samples_since_emit >= self.hop
            ):
                out.append(np.array(self._buffer, dtype=float))
                self._samples_since_emit = 0
        return out

    def reset(self) -> None:
        self._buffer.clear()
        self._samples_since_emit = 0
