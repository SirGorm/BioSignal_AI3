"""CNN-LSTM multi-task model (DeepConvLSTM-style).

Conv front-end extracts local feature interactions, LSTM models the resulting
"sequence" of conv activations. For per-window features, this means: 1D convs
mix neighboring features, then BiLSTM processes the conv-pooled sequence.

This architecture was the strongest in Ordóñez & Roggen 2016 for multimodal
wearable activity recognition — directly relevant prior work.

References:
- Ordóñez & Roggen 2016 — DeepConvLSTM, the canonical reference
- Karpathy et al. 2014 — origin of CNN+LSTM hybrid
- Hochreiter & Schmidhuber 1997 — LSTM
- Caruana 1997 — multitask hard parameter sharing
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads


class CNNLSTMMultiTask(nn.Module):
    """Multi-task CNN-LSTM over feature sequences.

    Architecture:
      Input:    (B, n_features) -> reshape to (B, 1, n_features)
      Conv1d:   1 -> 32, k=5
      Conv1d:   32 -> 64, k=3
      -> (B, 64, T') with T' = n_features
      Transpose to (B, T', 64)
      BiLSTM:   hidden=64, num_layers=1
      Mean-pool over T'
      Linear:   128 -> repr_dim
      Heads:    4 task-specific linear projections
    """

    def __init__(
        self,
        n_features: int,
        n_exercise: int,
        n_phase: int,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        n_lstm_layers: int = 1,
        repr_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.Dropout(dropout),
        )
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
        if x.dim() == 2:
            x = x.unsqueeze(1)             # (B, 1, n_features)
        f = self.conv(x)                    # (B, conv_channels, n_features)
        f = f.transpose(1, 2)               # (B, n_features, conv_channels)
        out, _ = self.lstm(f)               # (B, n_features, 2*hidden)
        h = out.mean(dim=1)
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
