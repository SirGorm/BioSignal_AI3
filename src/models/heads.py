"""Shared multi-task output heads — used by all 4 architectures.

Hard parameter sharing (Caruana 1997): one encoder produces a representation,
4 task-specific linear heads project to per-task outputs.
"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class MultiTaskHeads(nn.Module):
    """4 task heads sharing a common representation.

    All four tasks (exercise, phase, fatigue, reps) take input from the SAME
    encoder representation. The encoder differs per architecture (CNN-LSTM,
    LSTM, TCN, 1D-CNN); the heads are identical across architectures.
    """

    def __init__(self, repr_dim: int, n_exercise: int, n_phase: int,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head_exercise = nn.Linear(repr_dim, n_exercise)
        self.head_phase = nn.Linear(repr_dim, n_phase)
        self.head_fatigue = nn.Linear(repr_dim, 1)
        self.head_reps = nn.Linear(repr_dim, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.dropout(h)
        return {
            'exercise': self.head_exercise(h),
            'phase':    self.head_phase(h),
            'fatigue':  self.head_fatigue(h).squeeze(-1),
            'reps':     self.head_reps(h).squeeze(-1),
        }
