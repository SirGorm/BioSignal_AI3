"""BiLSTM multi-task model for raw multimodal biosignals.

Input: (B, C=6, T=200) — 6 channels, 200 time-steps.
Internally transposed to (B, T=200, C=6) for LSTM processing.

Architecture:
  Transpose (B, C, T) -> (B, T, C)
  BiLSTM: input_size=6, hidden=128, num_layers=2, bidirectional=True
  Final hidden state (last timestep, concat forward+backward) -> (B, 2*hidden)
  Linear(2*hidden, repr_dim) -> ReLU -> Dropout
  MultiTaskHeads

DEPLOYMENT NOTE: This model uses a BIDIRECTIONAL LSTM. BiLSTM is NON-CAUSAL —
it requires the full sequence in both directions, making it unsuitable for
real-time streaming. Marked as research_only. For deployment, use TCN_raw
or CNN1D_raw (causal).

For a deployment-ready LSTM, set bidirectional=False and use the final hidden
state of the FORWARD direction only.

References:
- Hochreiter & Schmidhuber 1997 — Long short-term memory (LSTM)
- Schuster & Paliwal 1997 — Bidirectional recurrent neural networks
- Caruana 1997 — Multitask learning; hard parameter sharing
- Loshchilov & Hutter 2019 — Decoupled weight decay regularization (AdamW)
- Goodfellow et al. 2016 — Deep Learning; regularization
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import MultiTaskHeads

RESEARCH_ONLY = True  # BiLSTM cannot deploy in real-time pipeline


class LSTMRawMultiTask(nn.Module):
    """Multi-task BiLSTM over raw multimodal biosignals.

    research_only=True: BiLSTM is non-causal; cannot stream in real-time.
    Use TCN_raw or CNN1D_raw for deployment.
    """

    research_only: bool = True

    def __init__(
        self,
        n_channels: int = 5,
        n_timesteps: int = 200,
        n_exercise: int = 4,
        n_phase: int = 3,
        hidden: int = 128,
        n_layers: int = 2,
        repr_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_size = hidden * 2 if bidirectional else hidden
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_size, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = MultiTaskHeads(repr_dim, n_exercise, n_phase,
                                     dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> representation (B, repr_dim)."""
        # Transpose to (B, T, C) for LSTM
        x = x.transpose(1, 2)            # (B, T, C)
        out, _ = self.lstm(x)             # (B, T, lstm_out_size)
        # Use final timestep (for BiLSTM this combines both directions' final hidden)
        h = out[:, -1, :]                 # (B, lstm_out_size)
        return self.proj(h)

    def forward(self, x: torch.Tensor):
        return self.heads(self.encode(x))
