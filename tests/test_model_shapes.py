"""Verify that all 4 architectures produce correct multi-task output shapes
on per-window feature input.
"""

import torch

from src.models.cnn1d import CNN1DMultiTask
from src.models.lstm import LSTMMultiTask
from src.models.cnn_lstm import CNNLSTMMultiTask
from src.models.tcn import TCNMultiTask


N_FEATURES = 47
N_EXERCISE = 5
N_PHASE = 4
BATCH = 8


def _check_outputs(out):
    assert set(out.keys()) == {'exercise', 'phase', 'fatigue', 'reps'}
    assert out['exercise'].shape == (BATCH, N_EXERCISE)
    assert out['phase'].shape == (BATCH, N_PHASE)
    assert out['fatigue'].shape == (BATCH,)
    assert out['reps'].shape == (BATCH,)


def _x():
    return torch.randn(BATCH, N_FEATURES)


def test_cnn1d_shape():
    m = CNN1DMultiTask(n_features=N_FEATURES,
                        n_exercise=N_EXERCISE, n_phase=N_PHASE)
    _check_outputs(m(_x()))


def test_lstm_shape():
    m = LSTMMultiTask(n_features=N_FEATURES,
                       n_exercise=N_EXERCISE, n_phase=N_PHASE)
    _check_outputs(m(_x()))


def test_cnn_lstm_shape():
    m = CNNLSTMMultiTask(n_features=N_FEATURES,
                          n_exercise=N_EXERCISE, n_phase=N_PHASE)
    _check_outputs(m(_x()))


def test_tcn_shape():
    m = TCNMultiTask(n_features=N_FEATURES,
                      n_exercise=N_EXERCISE, n_phase=N_PHASE)
    _check_outputs(m(_x()))


def test_tcn_is_causal():
    """The TCN must be strictly causal — output at position t doesn't depend
    on input after t. We verify by computing the encoder output on a full
    sequence and on a truncated sequence; the last position of the truncated
    output should match the corresponding position of the full output."""
    m = TCNMultiTask(n_features=N_FEATURES,
                      n_exercise=N_EXERCISE, n_phase=N_PHASE)
    m.eval()
    x = _x().unsqueeze(1)  # (B, 1, N_FEATURES) — match what encode() does

    with torch.no_grad():
        z_full = m.tcn(x)                      # (B, C, N_FEATURES)
        truncate_to = N_FEATURES - 5
        z_trunc = m.tcn(x[..., :truncate_to])  # (B, C, truncate_to)

    # Last column of truncated output should match corresponding column
    # of full output (within numerical tolerance — BatchNorm running stats
    # are equal because we're in eval mode).
    assert torch.allclose(
        z_trunc[..., -1], z_full[..., truncate_to - 1],
        atol=1e-4, rtol=1e-4,
    ), "TCN is not causal — output at position t depends on input after t"


def test_loss_module():
    """Sanity-check the multi-task loss with masking."""
    from src.training.losses import MultiTaskLoss
    loss_fn = MultiTaskLoss()

    preds = {
        'exercise': torch.randn(BATCH, N_EXERCISE, requires_grad=True),
        'phase':    torch.randn(BATCH, N_PHASE,    requires_grad=True),
        'fatigue':  torch.randn(BATCH,             requires_grad=True),
        'reps':     torch.randn(BATCH,             requires_grad=True),
    }
    targets = {
        'exercise': torch.randint(0, N_EXERCISE, (BATCH,)),
        'phase':    torch.randint(0, N_PHASE, (BATCH,)),
        'fatigue':  torch.rand(BATCH) * 9 + 1,
        'reps':     torch.rand(BATCH) * 10,
    }
    masks = {k: torch.ones(BATCH, dtype=torch.bool) for k in preds}

    total, parts = loss_fn(preds, targets, masks)
    assert total.requires_grad
    assert all(k in parts for k in preds.keys())
    assert all(p.item() >= 0 for p in parts.values())
