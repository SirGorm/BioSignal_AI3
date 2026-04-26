"""1D-CNN multi-task model.

Treats per-window engineered features as a 1-channel sequence of length
n_features. The CNN learns local feature interactions through small kernels.

NOTE on input: per-window features arrive as (batch, n_features). We reshape
to (batch, 1, n_features) and apply 1D convolutions ALONG the feature
dimension. This is a meaningful operation — adjacent features computed from
the same modality are likely correlated, so a kernel sees that local context.

For optimal results, group related features adjacent in the column ordering
(all EMG features together, all HRV features together, etc.). The
biosignal-feature-extractor outputs in this order by default.

References:
- Yang et al. 2015 — 1D-CNN for multichannel sensor data
- Caruana 1997 — multitask hard parameter sharing
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class CNN1DMultiTask(nn.Module):
    """Multi-task 1D-CNN over feature sequences.

    Architecture:
      Input:    (B, n_features)   -> reshape to (B, 1, n_features)
      Conv1d:   1 -> 32, k=5
      Conv1d:   32 -> 64, k=3
      Conv1d:   64 -> 128, k=3
      AdaptiveAvgPool1d -> (B, 128)
      Linear:   128 -> repr_dim
      Heads:    4 task-specific linear projections
    """

    def __init__(
        self,
        n_features: int,
        n_exercise: int,
        n_phase: int,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        # Causal padding not needed here (offline training) — kernels see
        # both neighboring features symmetrically.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, repr_dim),
            nn.ReLU(),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                      dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
