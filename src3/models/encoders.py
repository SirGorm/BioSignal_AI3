"""Sequence encoders for raw multimodal biosignals.

Input: (B, C, T). Output: (B, repr_dim).

CNN1D and CNN-LSTM are causal (deployable in real-time).
LSTM has a `bidirectional` switch — set False for deployment.
TCN uses pytorch-tcn (causal=True) when installed; falls back to a manual
trim-based causal block otherwise (Bai et al. 2018).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- CNN1D (causal) ---------------------------------------------------------


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.drop(F.relu(self.bn(self.conv(x))))


class CNN1DEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        channels: Sequence[int] = (32, 16, 8),
        kernel_sizes: Sequence[int] = (5, 5, 3),
        repr_dim: int = 96,
        dropout: float = 0.3,
    ):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError("channels and kernel_sizes must be same length")
        blocks: list[nn.Module] = []
        ch_in = n_channels
        for ch_out, k in zip(channels, kernel_sizes):
            blocks.append(_CausalConvBlock(ch_in, ch_out, k, dropout))
            ch_in = ch_out
        self.body = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], repr_dim),
            nn.ReLU(),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# --- LSTM -------------------------------------------------------------------


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        hidden: int = 24,
        n_layers: int = 1,
        repr_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_size = hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_size, repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.proj(out[:, -1, :])


# --- CNN-LSTM (DeepConvLSTM) ------------------------------------------------


class CNNLSTMEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        conv_first_channels: int = 8,
        conv_channels: int = 16,
        lstm_hidden: int = 24,
        n_lstm_layers: int = 1,
        repr_dim: int = 32,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_first_channels, 5, padding=0)
        self.bn1 = nn.BatchNorm1d(conv_first_channels)
        self.conv2 = nn.Conv1d(conv_first_channels, conv_channels, 3, padding=0)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        out_size = lstm_hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_size, repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (4, 0))
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = F.pad(x, (2, 0))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.proj(out.mean(dim=1))


# --- TCN (causal) -----------------------------------------------------------


class _TrimCausalBlock(nn.Module):
    """Bai et al. 2018 — pad+trim causal block. Used as fallback when
    pytorch-tcn is not installed."""

    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(in_ch, out_ch, k, padding=self.pad, dilation=dilation)
        self.b1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, k, padding=self.pad, dilation=dilation)
        self.b2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _trim(self, x):
        return x[..., :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        out = F.relu(self.b1(self._trim(self.c1(x))))
        out = self.drop(out)
        out = F.relu(self.b2(self._trim(self.c2(out))))
        out = self.drop(out)
        res = x if self.down is None else self.down(x)
        return F.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        channels: Sequence[int] = (4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.2,
        repr_dim: int = 64,
    ):
        super().__init__()
        try:
            from pytorch_tcn import TCN
            self._lib_tcn = TCN(
                num_inputs=n_channels,
                num_channels=list(channels),
                kernel_size=kernel_size,
                dropout=dropout,
                causal=True,
                use_norm="batch_norm",
            )
            self._using_lib = True
        except Exception:
            blocks = []
            ch_in = n_channels
            for i, ch_out in enumerate(channels):
                blocks.append(_TrimCausalBlock(
                    ch_in, ch_out, kernel_size, dilation=2 ** i, dropout=dropout,
                ))
                ch_in = ch_out
            self._fb_tcn = nn.Sequential(*blocks)
            self._using_lib = False

        self.proj = nn.Sequential(
            nn.Linear(channels[-1], repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._lib_tcn(x) if self._using_lib else self._fb_tcn(x)
        return self.proj(h[..., -1])


# --- MLP (features-only path) -----------------------------------------------


class FeatureLSTMEncoder(nn.Module):
    """v17-style LSTM over engineered features.

    Treats the (n_features,) vector as a sequence of length n_features with
    1 scalar per "timestep" — same as src/models/lstm.py:LSTMMultiTask. There
    is no real temporal axis here; the LSTM learns dependencies across
    feature ordering.

    Unidirectional only — required for the project's real-time deployment rule.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 40,
        n_layers: int = 1,
        repr_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        out_size = hidden * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(out_size, repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)              # (B, n_features, hidden)
        h = out.mean(dim=1)                 # mean-pool over the feature axis
        return self.proj(h)


class MLPEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 80,
        repr_dim: int = 80,
        dropout: float = 0.3,
        n_hidden_layers: int = 1,
    ):
        super().__init__()
        # Architecture: input -> [hidden_dim -> ReLU -> Dropout] × n_hidden_layers
        #               -> repr_dim -> ReLU -> Dropout
        layers: list[nn.Module] = []
        ch_in = n_features
        for _ in range(max(0, int(n_hidden_layers))):
            layers += [nn.Linear(ch_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            ch_in = hidden_dim
        layers += [nn.Linear(ch_in, repr_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*layers)
        self.repr_dim = repr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
