"""torchmetrics MetricCollection per task — replaces sklearn/scipy wrappers.

Live on GPU, support `.update(preds, target)` per batch + `.compute()` at
epoch end. Lightning auto-resets between train/val/test.
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef


def make_metrics(n_exercise: int, n_phase: int) -> dict[str, MetricCollection]:
    """One MetricCollection per task. Use a separate instance per train/val/test
    by Lightning convention (`self.train_metrics = make_metrics(...)['exercise']`)."""
    return {
        "exercise": MetricCollection(
            {
                "f1_macro": MulticlassF1Score(
                    num_classes=n_exercise, average="macro"
                ),
                "balanced_accuracy": MulticlassAccuracy(
                    num_classes=n_exercise, average="macro"
                ),
            },
            prefix="exercise/",
        ),
        "phase": MetricCollection(
            {
                "f1_macro": MulticlassF1Score(num_classes=n_phase, average="macro"),
                "balanced_accuracy": MulticlassAccuracy(
                    num_classes=n_phase, average="macro"
                ),
            },
            prefix="phase/",
        ),
        "fatigue": MetricCollection(
            {"mae": MeanAbsoluteError(), "pearson_r": PearsonCorrCoef()},
            prefix="fatigue/",
        ),
        "reps": MetricCollection({"mae": MeanAbsoluteError()}, prefix="reps/"),
    }


def update_metrics(
    metrics: dict[str, MetricCollection],
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> None:
    """Update each task's MetricCollection from a batch's preds+targets+masks."""
    # Tasks may be missing from `metrics` when only a subset is enabled
    # (e.g. --tasks exercise) — skip those without erroring.
    if "exercise" in metrics and masks["exercise"].any():
        m = masks["exercise"]
        metrics["exercise"].update(
            preds["exercise"][m].argmax(dim=-1), targets["exercise"][m]
        )
    if "phase" in metrics and masks["phase"].any():
        m = masks["phase"]
        # phase target may be soft (B, K) — collapse to argmax for hard metrics.
        t = targets["phase"][m]
        if t.dim() == 2:
            t = t.argmax(dim=-1)
        metrics["phase"].update(preds["phase"][m].argmax(dim=-1), t)
    if "fatigue" in metrics and masks["fatigue"].any():
        m = masks["fatigue"]
        # PearsonCorrCoef wants ≥ 2 samples; torchmetrics handles smaller batches
        # at compute() time but updating with 1 sample raises — guard explicitly.
        if int(m.sum().item()) >= 2:
            metrics["fatigue"].update(preds["fatigue"][m], targets["fatigue"][m])
    if "reps" in metrics and masks["reps"].any():
        m = masks["reps"]
        metrics["reps"].update(preds["reps"][m], targets["reps"][m])
