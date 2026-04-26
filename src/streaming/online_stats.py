"""
Online (Welford) running statistics for streaming normalisation.

Welford's algorithm maintains running mean and variance using O(1) memory,
without storing the full signal. This is the causal alternative to
computing (x - mean) / std on the full session.

References
----------
- Welford, B. P. (1962). Note on a method for calculating corrected sums of
  squares and products. Technometrics, 4(3), 419-420.
"""

from __future__ import annotations

import numpy as np


class OnlineStats:
    """Welford running mean and variance (Welford 1962).

    Usage
    -----
    >>> stats = OnlineStats()
    >>> stats.update(x)   # x: scalar or 1D array
    >>> z = stats.z(new_x)   # z-score using running mean/std
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x) -> None:
        """Update running statistics with new observation(s)."""
        for v in np.atleast_1d(x):
            v = float(v)
            if np.isnan(v):
                continue
            self.n += 1
            delta = v - self.mean
            self.mean += delta / self.n
            delta2 = v - self.mean
            self._M2 += delta * delta2

    @property
    def var(self) -> float:
        """Running variance (biased estimator for n > 1)."""
        if self.n < 2:
            return 0.0
        return self._M2 / (self.n - 1)

    @property
    def std(self) -> float:
        """Running standard deviation."""
        return float(np.sqrt(self.var))

    def z(self, x: float) -> float:
        """Z-score of x given current running stats."""
        return (x - self.mean) / (self.std + 1e-8)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0
