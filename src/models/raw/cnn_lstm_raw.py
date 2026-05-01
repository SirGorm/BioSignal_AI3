"""CNN-LSTM (DeepConvLSTM-style) multi-task model for raw multimodal biosignals.

Input: (B, C=6, T=200) — 6 channels, 200 time-steps.

Architecture follows Ordóñez & Roggen 2016 (DeepConvLSTM): convolutional front-end
extracts local temporal features, then a BiLSTM models longer-range dependencies in
the resulting feature sequence.

Architecture:
  Conv1d(6 -> 32, k=5, causal pad) + BN + ReLU + Dropout
  Conv1d(32 -> 64, k=3, causal pad) + BN + ReLU + Dropout
  Transpose (B, 64, T) -> (B, T, 64)
  BiLSTM: input_size=64, hidden=64, num_layers=1, bidirectional=True
  Mean-pool over T
  Linear(128, repr_dim) -> ReLU -> Dropout
  MultiTaskHeads

DEPLOYMENT NOTE: This model uses a BIDIRECTIONAL LSTM component. Non-causal —
research_only. For deployment, use TCN_raw or CNN1D_raw (both fully causal).

References:
- Ordóñez & Roggen 2016 — Deep convolutional and LSTM recurrent neural networks
  for multimodal wearable activity recognition (DeepConvLSTM)
- Hochreiter & Schmidhuber 1997 — LSTM
- Yang et al. 2015 — 1D-CNN for multichannel sensor data
- Caruana 1997 — Multitask learning; hard parameter sharing
- Loshchilov & Hutter 2019 — AdamW
- Goodfellow et al. 2016 — regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import MultiTaskHeads


class CNNLSTMRawMultiTask(nn.Module):
    """Multi-task CNN-LSTM (DeepConvLSTM) over raw multimodal biosignals.

    research_only=True: BiLSTM component is non-causal.
    """

    research_only: bool = True

    def __init__(
        self,
        n_channels: int = 5,
        n_timesteps: int = 200,
        n_exercise: int = 4,
        n_phase: int = 3,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        n_lstm_layers: int = 1,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps

        # Causal conv front-end
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, conv_channels, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                     dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> representation (B, repr_dim)."""
        # Causal pad: pad=k-1 on left only
        x = F.pad(x, (4, 0))             # pad 4 for k=5
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = F.pad(x, (2, 0))             # pad 2 for k=3
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        # x: (B, conv_channels, T)
        x = x.transpose(1, 2)            # (B, T, conv_channels)
        out, _ = self.lstm(x)             # (B, T, 2*lstm_hidden)
        h = out.mean(dim=1)               # mean-pool over T
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
