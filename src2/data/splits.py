"""Subject-wise CV splits — sklearn directly, no wrapper class.

Reads `configs/splits.csv` when present so NN runs reuse the LightGBM
baseline's folds (required for fair comparison per CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


def cv_iter(
    subject_ids: np.ndarray,
    splits_csv: Path | str = Path("configs/splits.csv"),
    fallback_n_splits: int = 5,
    scheme: str = "auto",
) -> Iterator[tuple[int, np.ndarray, np.ndarray, list]]:
    """Yield (fold, train_idx, test_idx, test_subjects) tuples.

    1. If splits_csv exists, reuse its fold column (LightGBM baseline parity).
    2. Else if scheme == 'loso' or (scheme == 'auto' and n_subjects ≤ 24),
       use LeaveOneGroupOut.
    3. Else GroupKFold(fallback_n_splits).
    """
    splits_csv = Path(splits_csv)
    if splits_csv.exists():
        df = pd.read_csv(splits_csv)
        sub_to_fold = dict(zip(df["subject_id"].astype(str), df["fold"]))
        unique_subs = np.unique(subject_ids)
        missing = [s for s in unique_subs if str(s) not in sub_to_fold]
        if missing:
            raise ValueError(
                f"splits.csv missing fold for subjects {missing[:10]!r} "
                f"(of {len(missing)}). Re-run /train (LightGBM baseline) first."
            )
        fold_arr = np.array(
            [sub_to_fold[str(s)] for s in subject_ids], dtype=int
        )
        for f in sorted(np.unique(fold_arr)):
            test_idx = np.where(fold_arr == f)[0]
            train_idx = np.where(fold_arr != f)[0]
            test_subs = sorted(np.unique(np.array(subject_ids)[test_idx]).tolist())
            yield int(f), train_idx, test_idx, test_subs
        return

    n_subs = len(np.unique(subject_ids))
    use_loso = scheme == "loso" or (scheme == "auto" and n_subs <= 24)
    splitter = LeaveOneGroupOut() if use_loso else GroupKFold(fallback_n_splits)
    X = np.zeros(len(subject_ids))
    for f, (train_idx, test_idx) in enumerate(
        splitter.split(X, groups=subject_ids)
    ):
        test_subs = sorted(np.unique(np.array(subject_ids)[test_idx]).tolist())
        yield f, train_idx, test_idx, test_subs
