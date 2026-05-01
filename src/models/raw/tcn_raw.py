"""TCN (Temporal Convolutional Network) multi-task model for raw biosignals.

Input: (B, C=6, T=200) — 6 channels, 200 time-steps (2 s at 100 Hz).

Dilated causal convolutions with residual connections, following Bai et al. 2018.
4 TCN blocks with exponentially increasing dilation: [1, 2, 4, 8].
Channels: [32, 64, 64, 128]. Kernel size: 5. Dropout: 0.2.

Each TCN block is strictly causal: output at time t depends only on
input at times <= t. This is verified by test_raw_tcn_causal.py.

Architecture:
  Input: (B, 6, 200)
  TCNBlock(6->32, k=5, dil=1) — causal: receptive field = 5
  TCNBlock(32->64, k=5, dil=2) — receptive field = 9
  TCNBlock(64->64, k=5, dil=4) — receptive field = 17
  TCNBlock(64->128, k=5, dil=8) — receptive field = 33
  Take final time-step (causal): (B, 128)
  Linear(128, repr_dim) -> ReLU -> Dropout
  MultiTaskHeads (4 task-specific linear projections)

Total receptive field: 1 + 2*(5-1)*(1+2+4+8) = 1 + 2*4*15 = 121 time-steps
(1.21 s at 100 Hz). Covers most relevant temporal patterns in a 2 s window.

Deployment: CAUSAL — suitable for real-time streaming pipeline.

References:
- Bai et al. 2018 — An empirical evaluation of generic convolutional and
  recurrent networks for sequence modeling (TCN)
- Caruana 1997 — Multitask learning; hard parameter sharing
- Yang et al. 2015 — 1D-CNN for multichannel sensor data
- Loshchilov & Hutter 2019 — Decoupled weight decay regularization (AdamW)
- Goodfellow et al. 2016 — Deep Learning; dropout regularization
"""

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class _TCNRawBlock(nn.Module):
    """Dilated causal Conv1d block with residual connection.

    Causality: left-pad by (kernel_size - 1) * dilation, then trim right.
    The padding on the right is zero (not applied), and we slice [:, :, :-pad]
    to remove the non-causal lookahead introduced by symmetric padding.

    This is the standard Bai et al. 2018 causal TCN block.
    """

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
        # Residual projection if channel dimensions differ
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        """Remove right-side padding added by symmetric padding in Conv1d."""
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self._trim(self.conv1(x))))
        out = self.drop(out)
        out = self.relu(self.bn2(self._trim(self.conv2(out))))
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNRawMultiTask(nn.Module):
    """Multi-task TCN over raw multimodal biosignals.

    Deployment candidate: CAUSAL — safe for real-time streaming pipeline.

    Receptive field: 1 + 2*(kernel_size-1)*(sum of dilations)
    With k=5, dilations=[1,2,4,8]: RF = 1 + 2*4*15 = 121 samples (1.21 s at 100 Hz).
    """

    def __init__(
        self,
        n_channels: int = 5,
        n_timesteps: int = 200,
        n_exercise: int = 4,
        n_phase: int = 3,
        channels: Sequence[int] = (32, 64, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        repr_dim: int = 128,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        blocks = []
        ch_in = n_channels
        for i, ch_out in enumerate(channels):
            blocks.append(_TCNRawBlock(ch_in, ch_out, kernel_size,
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
        """x: (B, C, T) -> representation (B, repr_dim)."""
        h = self.tcn(x)                    # (B, channels[-1], T)
        h = h[..., -1]                     # last timestep — causal
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
