"""Causal 1D-CNN multi-task model for raw multimodal biosignals.

Input: (B, C=6, T=200) — 6 biosignal channels, 200 time-steps (2 s at 100 Hz).

Causal padding: for each Conv1d with kernel k, we left-pad by (k-1) and truncate
the right end by (k-1) positions after convolution. This ensures output at time t
depends only on input at times <= t, making it suitable for real-time deployment.

Architecture:
  Conv1d(6 -> 32, k=7) + causal pad + BN + ReLU + Dropout
  Conv1d(32 -> 64, k=5) + causal pad + BN + ReLU + Dropout
  Conv1d(64 -> 128, k=3) + causal pad + BN + ReLU + Dropout
  AdaptiveAvgPool1d(1) -> Flatten -> Linear(128, repr_dim) -> ReLU
  MultiTaskHeads (4 task-specific linear heads)

Deployment: CAUSAL — suitable for real-time streaming pipeline.

References:
- Yang et al. 2015 — Deep convolutional neural networks on multichannel time
  series for human activity recognition (HAR with wearables; baseline CNN design)
- Caruana 1997 — Multitask learning; hard parameter sharing rationale
- Loshchilov & Hutter 2019 — Decoupled weight decay regularization (AdamW)
- Goodfellow et al. 2016 — Deep Learning; dropout regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class _CausalConvBlock(nn.Module):
    """Causal Conv1d block: left-pad + conv + BN + ReLU + Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dropout: float = 0.3):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad (causal): prepend zeros on the left of time dimension
        x = torch.nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return self.drop(x)


class CNN1DRawMultiTask(nn.Module):
    """Causal 1D-CNN over raw multimodal biosignals.

    Deployment candidate: CAUSAL — safe for real-time streaming.
    """

    def __init__(
        self,
        n_channels: int = 5,
        n_timesteps: int = 200,
        n_exercise: int = 4,
        n_phase: int = 3,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        self.encoder = nn.Sequential(
            _CausalConvBlock(n_channels, 32, kernel_size=7, dropout=dropout),
            _CausalConvBlock(32, 64, kernel_size=5, dropout=dropout),
            _CausalConvBlock(64, 128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, repr_dim),
            nn.ReLU(),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                     dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> representation (B, repr_dim)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
