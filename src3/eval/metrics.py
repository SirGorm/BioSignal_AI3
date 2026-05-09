"""torchmetrics MetricCollection per task — masked.

Replaces hand-rolled sklearn wrappers in src/eval/metrics.py. Reset at
each epoch by Lightning automatically.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from torchmetrics import (
    MetricCollection, F1Score, Accuracy, MeanAbsoluteError, PearsonCorrCoef,
)


class MultiTaskMetrics(nn.Module):
    """Per-task masked metrics. Call .update(preds, targets, masks) per batch,
    .compute() once per epoch, .reset() between epochs (Lightning handles).
    """

    def __init__(self, n_exercise: int, n_phase: int, prefix: str = "val"):
        super().__init__()
        self.prefix = prefix
        self.exercise = MetricCollection({
            "f1_macro": F1Score(task="multiclass", num_classes=n_exercise, average="macro"),
            "balanced_accuracy": Accuracy(
                task="multiclass", num_classes=n_exercise, average="macro",
            ),
        }, prefix=f"{prefix}/exercise/")
        self.phase = MetricCollection({
            "f1_macro": F1Score(task="multiclass", num_classes=n_phase, average="macro"),
            "balanced_accuracy": Accuracy(
                task="multiclass", num_classes=n_phase, average="macro",
            ),
        }, prefix=f"{prefix}/phase/")
        self.fatigue = MetricCollection({
            "mae": MeanAbsoluteError(),
            "pearson_r": PearsonCorrCoef(),
        }, prefix=f"{prefix}/fatigue/")
        self.reps = MetricCollection({
            "mae": MeanAbsoluteError(),
        }, prefix=f"{prefix}/reps/")

    def update(self, preds: Mapping[str, torch.Tensor],
               targets: Mapping[str, torch.Tensor],
               masks: Mapping[str, torch.Tensor]) -> None:
        m = masks["exercise"]
        if m.any():
            self.exercise.update(preds["exercise"][m].argmax(-1), targets["exercise"][m])

        m = masks["phase"]
        if m.any():
            t = targets["phase"]
            t = t.argmax(-1) if t.dim() == 2 else t
            self.phase.update(preds["phase"][m].argmax(-1), t[m])

        m = masks["fatigue"]
        if m.any():
            self.fatigue.update(preds["fatigue"][m].float(), targets["fatigue"][m].float())

        m = masks["reps"]
        if m.any():
            self.reps.update(preds["reps"][m].float(), targets["reps"][m].float())

    def compute(self) -> dict[str, torch.Tensor]:
        return {**self.exercise.compute(), **self.phase.compute(),
                **self.fatigue.compute(), **self.reps.compute()}

    def reset(self) -> None:
        for m in (self.exercise, self.phase, self.fatigue, self.reps):
            m.reset()
