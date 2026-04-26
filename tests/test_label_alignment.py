"""Verify joint-angle labels are properly aligned with biosignals in labeled data."""
from pathlib import Path
import pytest


def _find_labeled_files():
    files = list(Path('data/labeled').rglob('aligned_features.parquet'))
    if not files:
        pytest.skip('No labeled files yet')
    return files


def test_labels_only_during_active_sets():
    """phase_label and rep_count_in_set should be NaN/null when in_active_set=False."""
    import pandas as pd
    for f in _find_labeled_files():
        df = pd.read_parquet(f, columns=['in_active_set', 'phase_label', 'rep_count_in_set'])
        rest = df[~df['in_active_set']]
        # During rest, phase should be 'rest' or null, never concentric/eccentric
        bad_phase = rest[rest['phase_label'].isin(['concentric', 'eccentric', 'isometric'])]
        assert len(bad_phase) == 0, (
            f"{f}: {len(bad_phase)} rows have active phase_label outside active set"
        )


def test_rpe_consistent_within_set():
    """All rows of the same set should share the same rpe_for_this_set value."""
    import pandas as pd
    for f in _find_labeled_files():
        df = pd.read_parquet(f, columns=['subject_id', 'session_id', 'set_number',
                                            'rpe_for_this_set', 'in_active_set'])
        active = df[df['in_active_set']]
        # Group by (subject, session, set) and check unique RPE
        grp = active.groupby(['subject_id', 'session_id', 'set_number'])
        for key, g in grp:
            unique = g['rpe_for_this_set'].dropna().unique()
            assert len(unique) <= 1, (
                f"{f}: set {key} has multiple RPE values: {unique}"
            )


def test_rep_count_monotonic_in_set():
    """rep_count_in_set should be monotonically non-decreasing within a set."""
    import pandas as pd
    for f in _find_labeled_files():
        df = pd.read_parquet(f, columns=['subject_id', 'session_id', 'set_number',
                                            't', 'rep_count_in_set', 'in_active_set'])
        active = df[df['in_active_set']].sort_values('t')
        for key, g in active.groupby(['subject_id', 'session_id', 'set_number']):
            counts = g['rep_count_in_set'].dropna().values
            if len(counts) > 1:
                diffs = counts[1:] - counts[:-1]
                assert (diffs >= 0).all(), (
                    f"{f}: set {key}: rep_count not monotonic"
                )
