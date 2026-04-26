"""Multi-task loss with masking for partial labels.

References:
- Kendall et al. 2018 — uncertainty-weighted multi-task loss (optional)
"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combines 4 task losses, masked where targets are invalid.

    By default uses a fixed weighted sum. Set use_uncertainty_weighting=True
    to learn task weights per Kendall et al. 2018.
    """

    def __init__(
        self,
        w_exercise: float = 1.0,
        w_phase: float = 1.0,
        w_fatigue: float = 1.0,
        w_reps: float = 0.5,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.weights = {
            'exercise': w_exercise, 'phase': w_phase,
            'fatigue':  w_fatigue,  'reps': w_reps,
        }
        self.use_uncertainty = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Learnable log-variance per task
            self.log_var = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ):
        device = preds['exercise'].device
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        losses: Dict[str, torch.Tensor] = {}

        # Classification (cross-entropy)
        for k in ('exercise', 'phase'):
            m = masks[k]
            if m.any():
                losses[k] = F.cross_entropy(preds[k][m], targets[k][m])
            else:
                losses[k] = zero.clone()

        # Regression (L1 / smooth L1)
        m = masks['fatigue']
        if m.any():
            losses['fatigue'] = F.l1_loss(preds['fatigue'][m],
                                            targets['fatigue'][m].float())
        else:
            losses['fatigue'] = zero.clone()

        m = masks['reps']
        if m.any():
            losses['reps'] = F.smooth_l1_loss(preds['reps'][m],
                                                targets['reps'][m].float())
        else:
            losses['reps'] = zero.clone()

        if self.use_uncertainty:
            keys = ['exercise', 'phase', 'fatigue', 'reps']
            total = sum(
                0.5 * torch.exp(-self.log_var[i]) * losses[k] + 0.5 * self.log_var[i]
                for i, k in enumerate(keys)
            )
        else:
            total = sum(self.weights[k] * losses[k] for k in losses)

        return total, losses
