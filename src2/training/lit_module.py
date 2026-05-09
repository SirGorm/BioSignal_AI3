"""LightningModule that ties model + multi-task loss + torchmetrics together.

Replaces the entire content of `src/training/loop.py`:
- `_train_one_epoch` / `_evaluate`     → `training_step` / `validation_step`
- AMP autocast / GradScaler            → handled by `Trainer(precision='16-mixed')`
- early stopping / checkpointing       → Lightning callbacks
- TensorBoard logging                  → `self.log_dict(...)` is auto-routed
- multi-seed × multi-fold orchestration→ src2.pipeline.train loops over folds.
"""

from __future__ import annotations

import lightning as L
import torch
from torchmetrics import MetricCollection

from src2.eval.metrics import make_metrics, update_metrics
from src2.models.multitask import MultiTaskModel
from src2.training.losses import MultiTaskLoss


class LitMultiTask(L.LightningModule):
    """Single-fold multi-task LightningModule. Stateless across folds."""

    def __init__(
        self,
        arch: str,
        n_channels: int,
        n_exercise: int,
        n_phase: int,
        repr_dim: int = 64,
        encoder_kwargs: dict | None = None,
        head_dropout: float = 0.3,
        loss_weights: dict[str, float] | None = None,
        target_modes: dict[str, str] | None = None,
        enabled_tasks: list[str] | None = None,
        uncertainty: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        epochs: int = 50,
    ):
        super().__init__()
        # Lightning best-practice: persist all hparams so checkpoints reload cleanly.
        self.save_hyperparameters()

        self.model = MultiTaskModel(
            arch=arch,
            n_channels=n_channels,
            n_exercise=n_exercise,
            n_phase=n_phase,
            repr_dim=repr_dim,
            encoder_kwargs=encoder_kwargs,
            head_dropout=head_dropout,
        )
        self.loss_fn = MultiTaskLoss(
            weights=loss_weights,
            target_modes=target_modes,
            enabled_tasks=enabled_tasks,
            uncertainty=uncertainty,
        )

        # MetricCollections — only for tasks that actually receive gradient.
        # Tasks not in enabled_tasks have a randomly-initialised head and would
        # spam torchmetrics "compute() before update()" warnings each epoch.
        self._enabled = set(
            enabled_tasks or ["exercise", "phase", "fatigue", "reps"]
        )
        train_m = make_metrics(n_exercise=n_exercise, n_phase=n_phase)
        val_m = make_metrics(n_exercise=n_exercise, n_phase=n_phase)
        self.train_metrics: dict[str, MetricCollection] = torch.nn.ModuleDict(
            {f"train_{k}": v for k, v in train_m.items() if k in self._enabled}
        )  # type: ignore[assignment]
        self.val_metrics: dict[str, MetricCollection] = torch.nn.ModuleDict(
            {f"val_{k}": v for k, v in val_m.items() if k in self._enabled}
        )  # type: ignore[assignment]

    # ---- forward + step -----------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def _shared_step(
        self, batch: dict, phase: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        preds = self.model(batch["x"])
        total, parts = self.loss_fn(preds, batch["targets"], batch["masks"])
        # log losses
        self.log(f"{phase}/loss/total", total, prog_bar=(phase == "val"))
        for k, v in parts.items():
            self.log(f"{phase}/loss/{k}", v)
        # update metrics
        coll = (
            {k.replace(f"{phase}_", ""): m for k, m in self.train_metrics.items()}
            if phase == "train"
            else {k.replace(f"{phase}_", ""): m for k, m in self.val_metrics.items()}
        )
        update_metrics(coll, preds, batch["targets"], batch["masks"])
        return total, parts

    def training_step(self, batch: dict, batch_idx: int):
        # Skip batches with no valid label for any enabled task — their loss
        # is a detached zero (no grad_fn), which would crash backward.
        # Returning None tells Lightning to skip this optimizer step.
        if not any(bool(batch["masks"][k].any()) for k in self._enabled):
            return None
        total, _ = self._shared_step(batch, "train")
        return total

    def validation_step(self, batch: dict, batch_idx: int):
        if not any(bool(batch["masks"][k].any()) for k in self._enabled):
            return None
        total, _ = self._shared_step(batch, "val")
        return total

    @staticmethod
    def _coll_has_updates(coll: MetricCollection) -> bool:
        """torchmetrics fires compute()-before-update() warnings when no batch
        in an epoch had valid mask. Sanity check + masked tasks make this easy
        to hit. Detect via the per-metric _update_called flag."""
        return any(getattr(m, "_update_called", False) for m in coll.values())

    def on_train_epoch_end(self) -> None:
        for name, coll in self.train_metrics.items():
            if self._coll_has_updates(coll):
                self.log_dict(coll.compute(), prog_bar=False)
            coll.reset()

    def on_validation_epoch_end(self) -> None:
        for name, coll in self.val_metrics.items():
            if self._coll_has_updates(coll):
                self.log_dict(coll.compute(), prog_bar=True)
            coll.reset()

    # ---- optim --------------------------------------------------------------

    def configure_optimizers(self):
        # Kendall log_var must NOT be weight-decayed (it's a noise scale, not a weight).
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "log_var" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        param_groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(param_groups, lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs
        )
        return {"optimizer": opt, "lr_scheduler": sched}
