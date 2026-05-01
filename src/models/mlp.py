"""Plain MLP multi-task model over per-window engineered features.

Input: (batch, n_features) -> shared MLP encoder -> 4 task heads.
Used as a non-temporal baseline against the convolutional/recurrent archs.

References:
- Caruana 1997 — multi-task hard parameter sharing
- Goodfellow et al. 2016 — MLP regularization (Dropout, BatchNorm)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class MLPMultiTask(nn.Module):
    """Two hidden-layer MLP shared encoder + 4 task heads."""

    def __init__(
        self,
        n_features: int,
        n_exercise: int = 4,
        n_phase: int = 4,
        repr_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, repr_dim),
            nn.BatchNorm1d(repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
