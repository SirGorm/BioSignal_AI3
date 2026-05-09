"""Per-architecture encoders. Input (B, C, T) → (B, repr_dim).

The TCN encoder uses **pytorch-tcn** (`pip install pytorch-tcn`) so we don't
hand-roll the causal-trim residual block (which `src/models/raw/tcn_raw.py`
does). The other three encoders are thin enough that torch.nn already wins —
they're kept here for symmetry.
"""

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1D-CNN -----------------------------------------------------------


class _CausalConv1dBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dropout: float):
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(c_in, c_out, k)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.drop(F.relu(self.bn(self.conv(x))))


class CNN1DEncoder(nn.Module):
    """Stacked causal Conv1d blocks → adaptive avg pool → linear → repr."""

    def __init__(
        self,
        n_channels: int,
        channels: Sequence[int] = (32, 16, 8),
        kernel_sizes: Sequence[int] = (5, 5, 3),
        repr_dim: int = 96,
        dropout: float = 0.3,
        **_unused,
    ):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must have same length")
        blocks = []
        c_in = n_channels
        for c_out, k in zip(channels, kernel_sizes):
            blocks.append(_CausalConv1dBlock(c_in, c_out, k, dropout))
            c_in = c_out
        self.body = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], repr_dim),
            nn.ReLU(),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.body(x)


# ---------- LSTM -------------------------------------------------------------


class LSTMEncoder(nn.Module):
    """Unidirectional LSTM over (B, T, C) — final hidden state → repr."""

    def __init__(
        self,
        n_channels: int,
        hidden: int = 24,
        n_layers: int = 1,
        repr_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = False,
        **_unused,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, repr_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        out, _ = self.lstm(x.transpose(1, 2))
        return self.proj(out[:, -1, :])


# ---------- CNN-LSTM (DeepConvLSTM-style) -----------------------------------


class CNNLSTMEncoder(nn.Module):
    """Causal Conv1d front-end → LSTM → mean-pool → repr."""

    def __init__(
        self,
        n_channels: int,
        conv_first_channels: int = 8,
        conv_channels: int = 16,
        lstm_hidden: int = 24,
        n_lstm_layers: int = 1,
        repr_dim: int = 32,
        dropout: float = 0.3,
        bidirectional: bool = False,
        **_unused,
    ):
        super().__init__()
        self.conv1 = _CausalConv1dBlock(n_channels, conv_first_channels, 5, dropout)
        self.conv2 = _CausalConv1dBlock(conv_first_channels, conv_channels, 3, dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, repr_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(self.conv1(x))         # (B, C', T)
        h, _ = self.lstm(h.transpose(1, 2))   # (B, T, H)
        return self.proj(h.mean(dim=1))


# ---------- TCN via pytorch-tcn ---------------------------------------------


class TCNEncoder(nn.Module):
    """Causal TCN encoder built on `pytorch_tcn.TCN`.

    pytorch-tcn handles the dilated-causal residual block + receptive-field
    bookkeeping (Bai et al. 2018) and exposes streaming-friendly inference,
    so this is a clean drop-in for `src/models/raw/tcn_raw.py`'s _TCNRawBlock.
    """

    def __init__(
        self,
        n_channels: int,
        channels: Sequence[int] = (32, 64, 64, 128),
        kernel_size: int = 5,
        dropout: float = 0.2,
        repr_dim: int = 64,
        **_unused,
    ):
        super().__init__()
        from pytorch_tcn import TCN

        self.tcn = TCN(
            num_inputs=n_channels,
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        h = self.tcn(x)            # (B, channels[-1], T)
        return self.proj(h[..., -1])


# ---------- MLP (feature-input only) ----------------------------------------


class MLPEncoder(nn.Module):
    """Two-layer MLP over (B, n_features). Non-temporal baseline (Goodfellow 2016)."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 80,
        repr_dim: int = 80,
        dropout: float = 0.3,
        **_unused,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, repr_dim),
            nn.BatchNorm1d(repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, n_features)
        return self.body(x)


# ---------- Registry --------------------------------------------------------

ENCODERS = {
    "cnn1d": CNN1DEncoder,
    "lstm": LSTMEncoder,
    "cnn_lstm": CNNLSTMEncoder,
    "tcn": TCNEncoder,
    "mlp": MLPEncoder,
}
