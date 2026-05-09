"""Deterministic str -> int label encoder.

Thin wrapper to avoid pulling sklearn.preprocessing.LabelEncoder for two
labels — and to encode "unknown" as -1 explicitly so the masking layer
in the loss can detect it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class LabelEncoder:
    """Deterministic mapping. Unknown values → -1."""

    def __init__(self, classes: Sequence[str] | None = None):
        self.classes_: list[str] = list(classes) if classes else []
        self._idx = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, values: Iterable) -> "LabelEncoder":
        self.classes_ = sorted({
            str(v) for v in values
            if v is not None and not _is_nan(v)
        })
        self._idx = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, values: Iterable) -> np.ndarray:
        out = np.full(len(list(values)) if not hasattr(values, "__len__") else len(values),
                      -1, dtype=np.int64)
        for i, v in enumerate(values):
            if v is None or _is_nan(v):
                continue
            out[i] = self._idx.get(str(v), -1)
        return out

    @property
    def n_classes(self) -> int:
        return len(self.classes_)


def _is_nan(v) -> bool:
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False
