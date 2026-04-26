"""Subject-wise CV. Reuses configs/splits.csv from the LightGBM baseline run."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold


def load_or_generate_splits(
    subject_ids: np.ndarray,
    splits_path: Path = Path('configs/splits.csv'),
    n_splits_if_new: int = 5,
):
    """Load splits.csv from disk if present (REQUIRED for fair NN-vs-LGBM comparison).
    Otherwise generate fresh subject-wise GroupKFold and save.
    """
    if splits_path.exists():
        print(f"[cv] Reusing baseline splits from {splits_path}")
        df = pd.read_csv(splits_path)
        # Map fold assignments per unique subject_id
        sub_to_fold = dict(zip(df['subject_id'].astype(str), df['fold']))
        unique_subjects = np.unique(subject_ids)
        missing = [s for s in unique_subjects if s not in sub_to_fold]
        if missing:
            raise ValueError(
                f"splits.csv missing fold assignment for subjects: {missing[:10]}... "
                f"(first 10 of {len(missing)}). Cannot do fair comparison "
                f"without complete split coverage. Either regenerate splits "
                f"or rerun the LightGBM baseline on this subject set."
            )
        fold_array = np.array([sub_to_fold[str(s)] for s in subject_ids])
    else:
        print(f"[cv] No baseline splits at {splits_path} — generating new "
              f"GroupKFold({n_splits_if_new}). This is OK if you have NOT yet "
              f"run /train (LightGBM baseline). Otherwise this BREAKS "
              f"comparison — abort and use the baseline's splits.")
        gkf = GroupKFold(n_splits=n_splits_if_new)
        fold_array = np.full(len(subject_ids), -1, dtype=int)
        for fold, (_, test_idx) in enumerate(
            gkf.split(np.zeros(len(subject_ids)), groups=subject_ids)
        ):
            fold_array[test_idx] = fold
        # Persist
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'subject_id': subject_ids,
            'fold': fold_array,
            'split': ['cv'] * len(subject_ids),  # filled later by LGBM pipeline
        }).drop_duplicates('subject_id').to_csv(splits_path, index=False)

    # Build outer-fold list
    folds: List[Dict] = []
    for f in sorted(np.unique(fold_array)):
        if f < 0:
            continue
        test_mask = fold_array == f
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        test_subjects = sorted(np.unique(np.array(subject_ids)[test_idx]).tolist())
        folds.append({
            'fold': int(f),
            'train_idx': train_idx,
            'test_idx': test_idx,
            'test_subjects': test_subjects,
        })

    print(f"[cv] {len(folds)} folds prepared. "
          f"Subjects per test fold: {[len(f['test_subjects']) for f in folds]}")
    return folds


def loso_splits(subject_ids: np.ndarray) -> List[Dict]:
    """Strict Leave-One-Subject-Out (use only when N <= ~24 for compute reasons)."""
    logo = LeaveOneGroupOut()
    folds = []
    for fold, (train_idx, test_idx) in enumerate(
        logo.split(np.zeros(len(subject_ids)), groups=subject_ids)
    ):
        folds.append({
            'fold': fold,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'test_subjects': sorted(np.unique(
                np.array(subject_ids)[test_idx]).tolist()),
        })
    return folds
