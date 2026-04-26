"""TCN (Temporal Convolutional Network) multi-task model.

Dilated causal convolutions with residual connections. Same temporal modeling
power as LSTM but parallelizable. Strictly causal — making it the natural
deployment candidate for the real-time pipeline.

In Phase 1 (feature input), causality applies to the feature dimension rather
than time. In Phase 2 (raw signals), the same architecture applies causally
to actual time and is deployable as-is.

References:
- Bai et al. 2018 — TCN: An empirical evaluation of generic convolutional and
  recurrent networks for sequence modeling
- Caruana 1997 — multitask hard parameter sharing
"""

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class _TCNBlock(nn.Module):
    """Causal dilated conv block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=self.pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=self.pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _trim(self, x):
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = self.relu(self.bn1(self._trim(self.conv1(x))))
        out = self.drop(out)
        out = self.relu(self.bn2(self._trim(self.conv2(out))))
        out = self.drop(out)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class TCNMultiTask(nn.Module):
    """Multi-task TCN over feature sequences.

    Architecture:
      Input:    (B, n_features) -> reshape to (B, 1, n_features)
      TCN blocks with exponentially increasing dilation (1, 2, 4, 8)
      Take final timestep (causal) -> (B, last_channels)
      Linear:   last_channels -> repr_dim
      Heads:    4 task-specific linear projections
    """

    def __init__(
        self,
        n_features: int,
        n_exercise: int,
        n_phase: int,
        channels: Sequence[int] = (32, 64, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        repr_dim: int = 128,
    ):
        super().__init__()
        self.n_features = n_features

        blocks = []
        ch_in = 1
        for i, ch_out in enumerate(channels):
            blocks.append(_TCNBlock(ch_in, ch_out, kernel_size,
                                      dilation=2 ** i, dropout=dropout))
            ch_in = ch_out
        self.tcn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                      dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)             # (B, 1, n_features)
        h = self.tcn(x)                     # (B, C_last, n_features)
        h = h[..., -1]                      # last timestep — causal
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
