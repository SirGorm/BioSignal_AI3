"""Shape and causality tests for raw-signal multi-task architectures.

All 4 architectures accept (B, C=4, T=200) input and produce 4-task output dict.
ECG dropped (signal quality insufficient on this dataset).
EDA dropped (sensor floor on all recordings; Greco et al. 2016).
See src/data/raw_window_dataset.py:RAW_CHANNELS.
TCN causality is verified by comparing full vs. truncated sequence outputs.
"""

import torch
import pytest

from src.models.raw.cnn1d_raw import CNN1DRawMultiTask
from src.models.raw.lstm_raw import LSTMRawMultiTask
from src.models.raw.cnn_lstm_raw import CNNLSTMRawMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask

N_CHANNELS = 4
N_TIMESTEPS = 200
N_EXERCISE = 4
N_PHASE = 3
BATCH = 8


def _check_outputs(out):
    """Verify 4-task output shapes."""
    assert set(out.keys()) == {'exercise', 'phase', 'fatigue', 'reps'}, \
        f"Expected 4-task keys, got {set(out.keys())}"
    assert out['exercise'].shape == (BATCH, N_EXERCISE), \
        f"exercise shape: {out['exercise'].shape}"
    assert out['phase'].shape == (BATCH, N_PHASE), \
        f"phase shape: {out['phase'].shape}"
    assert out['fatigue'].shape == (BATCH,), \
        f"fatigue shape: {out['fatigue'].shape}"
    assert out['reps'].shape == (BATCH,), \
        f"reps shape: {out['reps'].shape}"


def _x():
    """Standard input tensor (B, C, T)."""
    return torch.randn(BATCH, N_CHANNELS, N_TIMESTEPS)


def test_cnn1d_raw_shape():
    m = CNN1DRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    _check_outputs(m(_x()))


def test_lstm_raw_shape():
    m = LSTMRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    _check_outputs(m(_x()))


def test_cnn_lstm_raw_shape():
    m = CNNLSTMRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    _check_outputs(m(_x()))


def test_tcn_raw_shape():
    m = TCNRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    _check_outputs(m(_x()))


def test_raw_tcn_causal():
    """TCNRawMultiTask must be strictly causal.

    Method: run the TCN stack on a full sequence and on a shorter truncated
    sequence. The output at the last step of the truncated output must match
    the corresponding step of the full output.

    BatchNorm is set to eval() mode so running statistics are fixed.
    """
    m = TCNRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    m.eval()

    x = torch.randn(BATCH, N_CHANNELS, N_TIMESTEPS)
    truncate_to = N_TIMESTEPS - 10  # shorter sequence

    with torch.no_grad():
        z_full = m.tcn(x)                          # (B, C_last, T)
        z_trunc = m.tcn(x[..., :truncate_to])      # (B, C_last, truncate_to)

    # Last column of truncated output should match the same position in full output
    assert torch.allclose(
        z_trunc[..., -1], z_full[..., truncate_to - 1],
        atol=1e-4, rtol=1e-4,
    ), (
        f"TCNRawMultiTask is NOT causal — output at position {truncate_to - 1} "
        f"differs between full and truncated sequences. "
        f"Max abs diff: {(z_trunc[..., -1] - z_full[..., truncate_to - 1]).abs().max():.6f}"
    )


def test_cnn1d_raw_causal():
    """CNN1DRawMultiTask should be causal (left-pad only).

    The encode() method uses only causal padding blocks. Test that the
    encoder output at position t is independent of input after t.
    We verify via a single causal conv block: output position t-1 in truncated
    sequence should match position t-1 of full sequence.
    """
    m = CNN1DRawMultiTask(
        n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
        n_exercise=N_EXERCISE, n_phase=N_PHASE,
    )
    m.eval()

    # Test the first conv block directly
    block = m.encoder[0]  # _CausalConvBlock
    x = torch.randn(BATCH, N_CHANNELS, N_TIMESTEPS)
    truncate_to = N_TIMESTEPS - 5

    with torch.no_grad():
        out_full = block(x)
        out_trunc = block(x[..., :truncate_to])

    assert torch.allclose(
        out_trunc[..., -1], out_full[..., truncate_to - 1],
        atol=1e-4, rtol=1e-4,
    ), "CNN1D causal padding test failed — not strictly causal"


def test_lstm_raw_research_only_flag():
    """BiLSTM_raw must have research_only=True (not a deployment candidate)."""
    m = LSTMRawMultiTask(n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                          n_exercise=N_EXERCISE, n_phase=N_PHASE)
    assert m.research_only is True, \
        "LSTMRawMultiTask must be marked research_only (BiLSTM is non-causal)"


def test_cnn_lstm_raw_research_only_flag():
    """CNN-LSTM_raw must have research_only=True."""
    m = CNNLSTMRawMultiTask(n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS,
                              n_exercise=N_EXERCISE, n_phase=N_PHASE)
    assert m.research_only is True, \
        "CNNLSTMRawMultiTask must be marked research_only (BiLSTM is non-causal)"
