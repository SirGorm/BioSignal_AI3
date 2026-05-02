"""Multi-task loss with masking for partial labels.

References:
- Kendall et al. 2018 — uncertainty-weighted multi-task loss (optional)
- Hinton et al. 2015 — soft targets / KL distillation (phase soft labels)
"""

from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Combines 4 task losses, masked where targets are invalid.

    By default uses a fixed weighted sum. Set use_uncertainty_weighting=True
    to learn task weights per Kendall et al. 2018.

    target_modes selects per-task target representation:
      reps:  'hard'        -> smooth_l1 on cumulative integer count
             'soft_window' -> smooth_l1 on fractional in-window count (same loss
                              kernel; only the dataset target differs)
      phase: 'hard'        -> cross_entropy on long index target
             'soft'        -> KL divergence on (B, K) probability target;
                              degenerates to cross_entropy when target is
                              one-hot, so this mode is a strict generalization.
    """

    def __init__(
        self,
        w_exercise: float = 1.0,
        w_phase: float = 1.0,
        w_fatigue: float = 1.0,
        w_reps: float = 0.5,
        use_uncertainty_weighting: bool = False,
        target_modes: Optional[Dict[str, str]] = None,
        enabled_tasks: Optional[list[str]] = None,
    ):
        super().__init__()
        self.weights = {
            'exercise': w_exercise, 'phase': w_phase,
            'fatigue':  w_fatigue,  'reps': w_reps,
        }
        # Optional task selection — when set, only these tasks contribute to
        # the total loss. Other heads are still computed for evaluation but
        # their gradients do not update encoder weights via this loss.
        # Use case: --tasks fatigue → train a fatigue-specialised model
        # without exercise/phase/reps interference.
        self.enabled = (set(enabled_tasks) if enabled_tasks else
                        {'exercise', 'phase', 'fatigue', 'reps'})
        self.use_uncertainty = use_uncertainty_weighting
        if use_uncertainty_weighting:
            # Learnable log-variance per ENABLED task; matches ordering below.
            self._uw_keys = [k for k in ('exercise', 'phase', 'fatigue', 'reps')
                              if k in self.enabled]
            self.log_var = nn.Parameter(torch.zeros(len(self._uw_keys)))
        modes = {'reps': 'hard', 'phase': 'hard'}
        if target_modes:
            modes.update(target_modes)
        self.target_modes = modes

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ):
        device = preds['exercise'].device
        zero = torch.tensor(0.0, device=device, requires_grad=False)
        losses: Dict[str, torch.Tensor] = {}

        # exercise: hard cross-entropy
        m = masks['exercise']
        if m.any():
            losses['exercise'] = F.cross_entropy(preds['exercise'][m],
                                                   targets['exercise'][m])
        else:
            losses['exercise'] = zero.clone()

        # phase: hard CE or soft KL-div
        m = masks['phase']
        if m.any():
            if self.target_modes['phase'] == 'soft':
                # targets['phase'] is (B, K) probability vector
                log_p = F.log_softmax(preds['phase'][m], dim=-1)
                losses['phase'] = F.kl_div(log_p,
                                            targets['phase'][m].float(),
                                            reduction='batchmean')
            else:
                losses['phase'] = F.cross_entropy(preds['phase'][m],
                                                    targets['phase'][m])
        else:
            losses['phase'] = zero.clone()

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
            total = sum(
                0.5 * torch.exp(-self.log_var[i]) * losses[k] + 0.5 * self.log_var[i]
                for i, k in enumerate(self._uw_keys)
            )
        else:
            total = sum(self.weights[k] * losses[k]
                        for k in losses if k in self.enabled)

        return total, losses
