"""LSTM multi-task model.

Treats per-window engineered features as a sequence of length n_features
(each "timestep" is one feature value, single channel). The LSTM learns
sequential dependencies in feature ordering.

This is a more aggressive interpretation than the 1D-CNN — LSTMs assume real
temporal structure that doesn't exist between, say, "EMG MNF" and "ECG HR".
Bidirectional helps because feature ordering is arbitrary; it lets the model
see context in both directions.

For Phase 2 with raw signals, this same architecture will see real time series.

References:
- Hochreiter & Schmidhuber 1997 — LSTM
- Schuster & Paliwal 1997 — bidirectional RNN
- Caruana 1997 — multitask hard parameter sharing
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class LSTMMultiTask(nn.Module):
    """Multi-task BiLSTM over feature sequences.

    Architecture:
      Input:        (B, n_features) -> reshape to (B, n_features, 1)
      BiLSTM:       hidden=64, num_layers=2, bidirectional
      Mean-pool:    over the feature dimension
      Linear:       128 -> repr_dim
      Heads:        4 task-specific linear projections
    """

    def __init__(
        self,
        n_features: int,
        n_exercise: int,
        n_phase: int,
        hidden: int = 64,
        n_layers: int = 2,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                      dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)             # (B, n_features, 2*hidden)
        h = out.mean(dim=1)                # mean-pool over features
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
