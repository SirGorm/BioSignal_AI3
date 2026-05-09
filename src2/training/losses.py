"""Multi-task loss with optional Kendall uncertainty weighting.

torchmetrics + torch.nn.functional already provide CE / KL-div / SmoothL1, so
this module is just composition. Same masking semantics as
`src/training/losses.py`.

References:
- Kendall et al. 2018 — multi-task uncertainty weighting
- Hinton et al. 2015 — soft targets / KL distillation (phase)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Per-task masked loss + scalar combination.

    Per-task losses:
      exercise : cross_entropy
      phase    : cross_entropy (hard target) or kl_div (soft target)
      fatigue  : l1_loss
      reps     : smooth_l1_loss

    Combination:
      uncertainty=True  -> learnable log_var per ENABLED task (Kendall 2018)
      uncertainty=False -> fixed weighted sum from `weights`
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        target_modes: dict[str, str] | None = None,
        enabled_tasks: list[str] | None = None,
        uncertainty: bool = True,
    ):
        super().__init__()
        self.weights = {
            "exercise": 1.0, "phase": 1.0, "fatigue": 1.0, "reps": 0.5,
        }
        if weights:
            self.weights.update(weights)
        self.target_modes = {"phase": "soft", "reps": "soft_window"}
        if target_modes:
            self.target_modes.update(target_modes)
        self.enabled = set(
            enabled_tasks or ["exercise", "phase", "fatigue", "reps"]
        )
        self.uncertainty = uncertainty
        if uncertainty:
            self._uw_keys = [
                k for k in ("exercise", "phase", "fatigue", "reps") if k in self.enabled
            ]
            self.log_var = nn.Parameter(torch.zeros(len(self._uw_keys)))

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = preds["exercise"].device
        zero = torch.zeros((), device=device)
        parts: dict[str, torch.Tensor] = {}
        signal: dict[str, bool] = {}

        # exercise (hard CE)
        m = masks["exercise"]
        signal["exercise"] = bool(m.any())
        parts["exercise"] = (
            F.cross_entropy(preds["exercise"][m], targets["exercise"][m])
            if signal["exercise"] else zero.clone()
        )

        # phase (KL or CE) — auto-detect from target dim so a dataset that
        # returns hard labels (long, 1D after masking) works regardless of
        # the configured target_mode. Soft mode is only used when the dataset
        # actually returns a (B, K) probability vector.
        m = masks["phase"]
        signal["phase"] = bool(m.any())
        if signal["phase"]:
            tgt = targets["phase"][m]
            if self.target_modes["phase"] == "soft" and tgt.dim() == 2:
                logp = F.log_softmax(preds["phase"][m], dim=-1)
                parts["phase"] = F.kl_div(
                    logp, tgt.float(), reduction="batchmean"
                )
            else:
                parts["phase"] = F.cross_entropy(preds["phase"][m], tgt.long())
        else:
            parts["phase"] = zero.clone()

        # fatigue (L1)
        m = masks["fatigue"]
        signal["fatigue"] = bool(m.any())
        parts["fatigue"] = (
            F.l1_loss(preds["fatigue"][m], targets["fatigue"][m].float())
            if signal["fatigue"] else zero.clone()
        )

        # reps (smooth L1)
        m = masks["reps"]
        signal["reps"] = bool(m.any())
        parts["reps"] = (
            F.smooth_l1_loss(preds["reps"][m], targets["reps"][m].float())
            if signal["reps"] else zero.clone()
        )

        if self.uncertainty:
            terms = [
                0.5 * torch.exp(-self.log_var[i]) * parts[k]
                + 0.5 * self.log_var[i]
                for i, k in enumerate(self._uw_keys)
                if signal[k]
            ]
            total = sum(terms) if terms else zero.clone()
        else:
            total = sum(self.weights[k] * parts[k] for k in parts if k in self.enabled)
        return total, parts
