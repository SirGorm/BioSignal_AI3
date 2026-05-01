"""Unit tests for the phase-head (recording, set) whitelist."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phase_whitelist import load_phase_whitelist, whitelist_mask


def test_load_phase_whitelist_round_trip(tmp_path: Path):
    csv = tmp_path / "wl.csv"
    csv.write_text(
        "# header comment\n"
        "recording_id,set_number\n"
        "recording_006,1\n"
        "recording_006,3\n"
        "recording_014,12\n"
    )
    wl = load_phase_whitelist(csv)
    assert wl == {("recording_006", 1), ("recording_006", 3), ("recording_014", 12)}


def test_load_phase_whitelist_none():
    assert load_phase_whitelist(None) is None


def test_load_phase_whitelist_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_phase_whitelist(tmp_path / "does_not_exist.csv")


def test_load_phase_whitelist_missing_columns(tmp_path: Path):
    csv = tmp_path / "bad.csv"
    csv.write_text("recording_id,foo\nrecording_006,1\n")
    with pytest.raises(ValueError):
        load_phase_whitelist(csv)


def test_load_phase_whitelist_float_set_number(tmp_path: Path):
    csv = tmp_path / "wl.csv"
    csv.write_text("recording_id,set_number\nrecording_006,1.0\nrecording_006,2.0\n")
    wl = load_phase_whitelist(csv)
    assert wl == {("recording_006", 1), ("recording_006", 2)}


def test_whitelist_mask_basic():
    rids = ["recording_006"] * 4 + ["recording_007"] * 4
    sets = [1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 2.0, np.nan]
    wl = {("recording_006", 1), ("recording_006", 3), ("recording_007", 2)}
    m = whitelist_mask(rids, sets, wl)
    expected = np.array([True, True, False, True, False, True, True, False])
    assert m.dtype == bool
    np.testing.assert_array_equal(m, expected)


def test_whitelist_mask_none_passes_through():
    rids = ["recording_006", "recording_007"]
    sets = [1.0, 2.0]
    m = whitelist_mask(rids, sets, None)
    np.testing.assert_array_equal(m, np.array([True, True]))


def test_window_feature_dataset_phase_whitelist(tmp_path: Path):
    """End-to-end: WindowFeatureDataset masks out phase for non-whitelisted sets."""
    from src.data.datasets import WindowFeatureDataset

    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        'recording_id': ['recording_006'] * 25 + ['recording_007'] * 25,
        'subject_id':   ['Alice'] * 25 + ['Bob'] * 25,
        'set_number':   [1.0] * 12 + [2.0] * 13 + [1.0] * 25,
        'in_active_set': [True] * n,
        'exercise':     ['squat'] * n,
        'phase_label':  ['concentric'] * n,
        'rep_count_in_set': [5.0] * n,
        'rpe_for_this_set': [6.0] * n,
        'feat_a':       rng.standard_normal(n).astype(np.float32),
        'feat_b':       rng.standard_normal(n).astype(np.float32),
    })
    parquet = tmp_path / "window_features.parquet"
    df.to_parquet(parquet)

    # No whitelist: every active window has m_phase=True
    ds_full = WindowFeatureDataset([parquet], active_only=True, verbose=False)
    assert ds_full.m_phase.all().item()

    # Whitelist only (recording_006, set 1): only first 12 windows keep phase mask
    ds_wl = WindowFeatureDataset(
        [parquet], active_only=True,
        phase_whitelist={("recording_006", 1)},
        verbose=False,
    )
    assert ds_wl.m_phase.sum().item() == 12
    # The other heads must be unaffected
    assert ds_wl.m_exercise.sum().item() == n
    assert ds_wl.m_fatigue.sum().item() == n
    assert ds_wl.m_reps.sum().item() == n
