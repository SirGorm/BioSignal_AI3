"""Multi-task loss with masking + optional Kendall uncertainty weighting.

Same semantics as src/training/losses.py — no behavioural change.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        w_exercise: float = 1.0,
        w_phase: float = 1.0,
        w_fatigue: float = 1.0,
        w_reps: float = 0.5,
        use_uncertainty_weighting: bool = False,
        target_modes: Mapping[str, str] | None = None,
        enabled_tasks: list[str] | None = None,
    ):
        super().__init__()
        self.weights = {
            "exercise": w_exercise, "phase": w_phase,
            "fatigue":  w_fatigue,  "reps":  w_reps,
        }
        self.enabled = (set(enabled_tasks) if enabled_tasks
                        else {"exercise", "phase", "fatigue", "reps"})
        self.modes = {"reps": "hard", "phase": "soft", **(target_modes or {})}
        self.use_uw = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self._uw_keys = [k for k in ("exercise", "phase", "fatigue", "reps")
                             if k in self.enabled]
            self.log_var = nn.Parameter(torch.zeros(len(self._uw_keys)))

    def forward(self, preds, targets, masks):
        device = preds["exercise"].device
        zero = torch.zeros((), device=device)
        losses, has = {}, {}

        m = masks["exercise"]
        has["exercise"] = bool(m.any())
        losses["exercise"] = (
            F.cross_entropy(preds["exercise"][m], targets["exercise"][m])
            if has["exercise"] else zero.clone()
        )

        m = masks["phase"]
        has["phase"] = bool(m.any())
        if has["phase"]:
            if self.modes["phase"] == "soft":
                log_p = F.log_softmax(preds["phase"][m], dim=-1)
                losses["phase"] = F.kl_div(
                    log_p, targets["phase"][m].float(), reduction="batchmean",
                )
            else:
                losses["phase"] = F.cross_entropy(preds["phase"][m], targets["phase"][m])
        else:
            losses["phase"] = zero.clone()

        m = masks["fatigue"]
        has["fatigue"] = bool(m.any())
        losses["fatigue"] = (
            F.l1_loss(preds["fatigue"][m], targets["fatigue"][m].float())
            if has["fatigue"] else zero.clone()
        )

        m = masks["reps"]
        has["reps"] = bool(m.any())
        losses["reps"] = (
            F.smooth_l1_loss(preds["reps"][m], targets["reps"][m].float())
            if has["reps"] else zero.clone()
        )

        if self.use_uw:
            terms = [
                0.5 * torch.exp(-self.log_var[i]) * losses[k] + 0.5 * self.log_var[i]
                for i, k in enumerate(self._uw_keys) if has[k]
            ]
            total = sum(terms) if terms else zero.clone()
        else:
            total = sum(self.weights[k] * losses[k]
                        for k in losses if k in self.enabled)
        return total, losses
