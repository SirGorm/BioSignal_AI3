"""Encoder + MultiTaskHeads composition. Pure factory."""

from __future__ import annotations

import torch
import torch.nn as nn

from src2.models.encoders import ENCODERS
from src2.models.heads import MultiTaskHeads


class MultiTaskModel(nn.Module):
    """Encoder → 4-head multi-task model. Returns dict[str, Tensor]."""

    def __init__(
        self,
        arch: str,
        n_channels: int,
        n_exercise: int,
        n_phase: int,
        repr_dim: int = 64,
        encoder_kwargs: dict | None = None,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        if arch not in ENCODERS:
            raise ValueError(f"Unknown arch={arch!r}; choices: {list(ENCODERS)}")
        ekwargs = dict(encoder_kwargs or {})
        ekwargs["repr_dim"] = repr_dim
        # MLP takes n_features; everyone else takes n_channels (the input shape
        # for MLP is (B, n_features), for others (B, C, T) where C = n_channels).
        input_kw = "n_features" if arch == "mlp" else "n_channels"
        self.encoder = ENCODERS[arch](**{input_kw: n_channels}, **ekwargs)
        self.heads = MultiTaskHeads(
            self.encoder.repr_dim, n_exercise, n_phase, dropout=head_dropout
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.encoder(x))
