"""Welford online mean/variance — used for streaming normalisation.

References
----------
- Welford 1962 — Note on a method for calculating corrected sums of squares
"""

from __future__ import annotations

import numpy as np


class OnlineStats:
    __slots__ = ("n", "mean", "_M2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0

    def update(self, x) -> None:
        for v in np.atleast_1d(x):
            v = float(v)
            if np.isnan(v):
                continue
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self._M2 += d * (v - self.mean)

    @property
    def var(self) -> float:
        return self._M2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def z(self, x: float) -> float:
        return (x - self.mean) / (self.std + 1e-8)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._M2 = 0.0
