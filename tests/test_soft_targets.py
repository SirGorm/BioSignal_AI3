"""Tests for the soft-target plumbing introduced for soft phase + soft reps.

Covers:
- _fill_rep_density_hz produces densities that integrate to the rep count
- soft rep target over a 2 s window equals (fractional) reps in window
- soft phase one-hot KL-div equals hard cross-entropy (degeneracy check)
- soft_to_set_count rounds aggregated soft predictions to ground-truth count
- MultiTaskLoss branches correctly on phase target mode
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.labeling.align import _fill_rep_density_hz
from src.eval.rep_aggregation import soft_to_set_count, soft_to_set_counts_grouped
from src.training.losses import MultiTaskLoss


# ---------------------------------------------------------------------------
# rep_density_hz construction (Step 1)
# ---------------------------------------------------------------------------

def test_rep_density_integral_equals_n_reps():
    """∫ rep_density_hz dt over a set must equal the number of reps."""
    set_start = 0.0
    n_reps = 5
    rep_dur = 2.0
    boundaries = [set_start + (k + 1) * rep_dur for k in range(n_reps)]  # ends
    set_end = boundaries[-1]

    fs = 100
    grid_t = np.linspace(set_start, set_end, int((set_end - set_start) * fs) + 1)
    indices = np.arange(len(grid_t))
    out = np.zeros(len(grid_t), dtype=float)

    _fill_rep_density_hz(grid_t, indices, set_start, boundaries, out)

    dt = 1.0 / fs
    integral = float(out.sum() * dt)
    # Expect ≈ n_reps; tiny error from boundary inclusion at endpoints.
    assert abs(integral - n_reps) < 0.05, (
        f"integral {integral} vs n_reps {n_reps}"
    )


def test_rep_density_one_window_equals_one_rep():
    """Mean density over a 2 s window inside the set × 2 s = 1.0 rep."""
    set_start = 0.0
    boundaries = [2.0, 4.0, 6.0, 8.0, 10.0]  # 5 reps of 2 s each

    fs = 100
    grid_t = np.linspace(set_start, 10.0, int(10.0 * fs) + 1)
    out = np.zeros(len(grid_t), dtype=float)
    _fill_rep_density_hz(grid_t, np.arange(len(grid_t)),
                          set_start, boundaries, out)

    # 2 s window starting at t=4 (covers rep 3 fully).
    win_start = int(4.0 * fs)
    win_end = win_start + int(2.0 * fs)
    soft_target = float(out[win_start:win_end].mean() * 2.0)
    assert abs(soft_target - 1.0) < 0.02


def test_rep_density_boundary_window_partial():
    """Window straddling the set end should give a fractional target."""
    set_start = 0.0
    boundaries = [2.0, 4.0, 6.0, 8.0, 10.0]
    fs = 100

    # Build a longer grid so we can place a 2 s window at t=[9, 11].
    grid_t = np.linspace(set_start, 12.0, int(12.0 * fs) + 1)
    in_set_idx = np.where(grid_t <= 10.0)[0]
    out = np.zeros(len(grid_t), dtype=float)
    _fill_rep_density_hz(grid_t, in_set_idx, set_start, boundaries, out)

    win_start = int(9.0 * fs)
    win_end = win_start + int(2.0 * fs)
    soft_target = float(out[win_start:win_end].mean() * 2.0)
    # Expect ≈ 0.5: only half the window overlaps a rep at density 0.5 Hz.
    assert 0.40 < soft_target < 0.55, soft_target


# ---------------------------------------------------------------------------
# rep aggregation (Step 6)
# ---------------------------------------------------------------------------

def test_soft_to_set_count_rounds_correctly():
    """100 windows of soft pred 1.0 with hop=0.1, win=2.0 → 5 reps."""
    preds = [1.0] * 100
    assert soft_to_set_count(preds, hop_s=0.1, window_s=2.0) == 5


def test_soft_to_set_count_clips_negatives():
    preds = [1.0, -0.2, 1.0, -0.5, 1.0]
    # without clipping: sum=2.3, with clip: 3.0
    out_clip = soft_to_set_count(preds, hop_s=1.0, window_s=1.0,
                                   clip_negative=True)
    out_noclip = soft_to_set_count(preds, hop_s=1.0, window_s=1.0,
                                     clip_negative=False)
    assert out_clip == 3
    assert out_noclip == 2


def test_soft_to_set_counts_grouped_skips_nan():
    preds = [1.0, 1.0, 1.0, 1.0]
    sids = [('rec', 1), ('rec', 1), float('nan'), ('rec', 2)]
    out = soft_to_set_counts_grouped(preds, sids, hop_s=1.0, window_s=1.0)
    assert ('rec', 1) in out
    assert ('rec', 2) in out
    assert len(out) == 2


# ---------------------------------------------------------------------------
# soft phase loss degeneracy (Step 5)
# ---------------------------------------------------------------------------

def test_soft_phase_kl_equals_ce_on_one_hot():
    """KL(softmax(logits), one_hot(label)) == CE(logits, label)."""
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    hard = torch.tensor([0, 2, 1, 0])
    soft = F.one_hot(hard, num_classes=3).float()

    log_p = F.log_softmax(logits, dim=-1)
    loss_kl = F.kl_div(log_p, soft, reduction='batchmean')
    loss_ce = F.cross_entropy(logits, hard)
    assert torch.allclose(loss_kl, loss_ce, atol=1e-5)


def test_multitask_loss_phase_soft_branch():
    """MultiTaskLoss with phase='soft' accepts (B, K) targets without crash."""
    loss_fn = MultiTaskLoss(target_modes={'phase': 'soft'})
    B, K, n_ex = 4, 3, 2
    preds = {
        'exercise': torch.randn(B, n_ex),
        'phase':    torch.randn(B, K),
        'fatigue':  torch.randn(B),
        'reps':     torch.randn(B),
    }
    targets = {
        'exercise': torch.tensor([0, 1, 0, 1]),
        'phase':    torch.softmax(torch.randn(B, K), dim=-1),
        'fatigue':  torch.rand(B) * 10,
        'reps':     torch.rand(B) * 2,
    }
    masks = {k: torch.ones(B, dtype=torch.bool) for k in targets}
    total, parts = loss_fn(preds, targets, masks)
    assert torch.isfinite(total)
    assert 'phase' in parts and torch.isfinite(parts['phase'])


def test_multitask_loss_phase_hard_unchanged():
    """phase='hard' (default) still uses cross_entropy."""
    loss_fn = MultiTaskLoss()
    B, K, n_ex = 4, 3, 2
    preds = {
        'exercise': torch.randn(B, n_ex),
        'phase':    torch.randn(B, K),
        'fatigue':  torch.randn(B),
        'reps':     torch.randn(B),
    }
    targets = {
        'exercise': torch.tensor([0, 1, 0, 1]),
        'phase':    torch.tensor([0, 2, 1, 0]),
        'fatigue':  torch.rand(B) * 10,
        'reps':     torch.rand(B) * 2,
    }
    masks = {k: torch.ones(B, dtype=torch.bool) for k in targets}
    total, parts = loss_fn(preds, targets, masks)
    assert torch.isfinite(total)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
