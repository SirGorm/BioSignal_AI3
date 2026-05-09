"""Encoder + MultiTaskHeads composition.

Single class that wraps any encoder. The Lightning module just sees a
dict-returning nn.Module and a `repr_dim` attribute.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from src3.models.encoders import (
    CNN1DEncoder, CNNLSTMEncoder, LSTMEncoder, TCNEncoder, MLPEncoder,
    FeatureLSTMEncoder,
)
from src3.models.heads import MultiTaskHeads


_RAW_REGISTRY = {
    "cnn1d":    CNN1DEncoder,
    "lstm":     LSTMEncoder,
    "cnn_lstm": CNNLSTMEncoder,
    "tcn":      TCNEncoder,
}

_FEATURE_REGISTRY = {
    "mlp":  MLPEncoder,
    "lstm": FeatureLSTMEncoder,
}


def build_raw_encoder(arch: str, n_channels: int, kwargs: Mapping) -> nn.Module:
    if arch not in _RAW_REGISTRY:
        raise ValueError(f"Unknown raw arch {arch!r}; valid: {list(_RAW_REGISTRY)}")
    return _RAW_REGISTRY[arch](n_channels=n_channels, **dict(kwargs))


def build_feature_encoder(arch: str, n_features: int, kwargs: Mapping) -> nn.Module:
    if arch not in _FEATURE_REGISTRY:
        raise ValueError(f"Feature-input encoder {arch!r} not implemented; "
                         f"valid: {list(_FEATURE_REGISTRY)}")
    return _FEATURE_REGISTRY[arch](n_features=n_features, **dict(kwargs))


class MultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, n_exercise: int, n_phase: int,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.heads = MultiTaskHeads(
            repr_dim=encoder.repr_dim,
            n_exercise=n_exercise,
            n_phase=n_phase,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.encoder(x))
