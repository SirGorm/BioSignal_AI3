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
from typing import Sequence

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class CNN1DMultiTask(nn.Module):
    """Multi-task 1D-CNN over feature sequences.

    Architecture:
      Input:    (B, n_features)   -> reshape to (B, 1, n_features)
      Conv1d stack with channels (default (16, 32, 32))
      AdaptiveAvgPool1d -> (B, channels[-1])
      Linear:   channels[-1] -> repr_dim
      Heads:    4 task-specific linear projections
    """

    def __init__(
        self,
        n_features: int,
        n_exercise: int,
        n_phase: int,
        channels: Sequence[int] = (16, 32, 32),
        kernel_sizes: Sequence[int] = (5, 3, 3),
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must be same length")
        self.n_features = n_features

        layers = []
        in_ch = 1
        for out_ch, k in zip(channels, kernel_sizes):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], repr_dim),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                      dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, 1, n_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
