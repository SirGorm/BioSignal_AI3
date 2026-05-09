"""Shared multi-task heads — same as src/models/heads.py.

Hard parameter sharing (Caruana 1997): one encoder, four task-specific
linear projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    def __init__(self, repr_dim: int, n_exercise: int, n_phase: int,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.exercise = nn.Linear(repr_dim, n_exercise)
        self.phase = nn.Linear(repr_dim, n_phase)
        self.fatigue = nn.Linear(repr_dim, 1)
        self.reps = nn.Linear(repr_dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.dropout(h)
        return {
            "exercise": self.exercise(h),
            "phase":    self.phase(h),
            "fatigue":  self.fatigue(h).squeeze(-1),
            "reps":     self.reps(h).squeeze(-1),
        }
