"""Verify subject_id never appears in both train and test splits."""
from pathlib import Path
import pytest


def _load_splits():
    candidates = [
        Path('configs/splits.csv'),
        Path('runs/current/splits.csv'),
        Path('data/splits.csv'),
    ]
    for c in candidates:
        if c.exists():
            import pandas as pd
            return pd.read_csv(c)
    pytest.skip('No splits.csv found yet')


def test_no_subject_in_both_splits():
    splits = _load_splits()
    train = set(splits.query("split == 'train'")['subject_id'])
    test = set(splits.query("split == 'test'")['subject_id'])
    overlap = train & test
    assert not overlap, f"LEAKAGE: subjects in both splits: {overlap}"


def test_no_session_split_within_subject():
    splits = _load_splits()
    by_subject = splits.groupby('subject_id')['split'].nunique()
    bad = by_subject[by_subject > 1].index.tolist()
    assert not bad, f"Subjects across multiple splits: {bad}"
