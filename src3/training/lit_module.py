"""LightningModule wrapping MultiTaskModel + MultiTaskLoss + MultiTaskMetrics.

Replaces src/training/loop.py:run_cv. Lightning handles AMP, grad clip,
optimizer step, scheduler, checkpoint, early stopping, TB logging.
"""

from __future__ import annotations

from typing import Mapping

import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src3.eval.metrics import MultiTaskMetrics
from src3.training.losses import MultiTaskLoss


class LitMultiTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_exercise: int,
        n_phase: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        loss_kwargs: Mapping | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = MultiTaskLoss(**(loss_kwargs or {}))
        self.train_metrics = MultiTaskMetrics(n_exercise, n_phase, prefix="train")
        self.val_metrics   = MultiTaskMetrics(n_exercise, n_phase, prefix="val")
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        # Don't pickle the model into hparams.
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, metrics: MultiTaskMetrics, stage: str):
        preds = self(batch["x"])
        total, parts = self.loss_fn(preds, batch["targets"], batch["masks"])
        self.log(f"{stage}/loss", total, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in parts.items():
            self.log(f"{stage}/loss_{k}", v, on_step=False, on_epoch=True)
        # Move to CPU for torchmetrics so masking ops on bool tensors stay safe
        # under AMP — torchmetrics handles dtype promotion internally.
        metrics.update(preds, batch["targets"], batch["masks"])
        return total

    def training_step(self, batch, _):
        return self._step(batch, self.train_metrics, "train")

    def validation_step(self, batch, _):
        return self._step(batch, self.val_metrics, "val")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
