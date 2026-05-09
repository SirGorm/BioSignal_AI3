"""Subject-wise CV splits.

Reuses the LightGBM baseline's configs/splits.csv when present so that NN
results stay comparable. Falls back to sklearn.GroupKFold otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_subjects: tuple[str, ...]


def _from_baseline_csv(
    splits_csv: Path,
    subject_ids: Sequence[str],
) -> list[Fold]:
    df = pd.read_csv(splits_csv)
    sub_to_fold = dict(zip(df["subject_id"].astype(str), df["fold"].astype(int)))
    sids = np.asarray([str(s) for s in subject_ids])
    missing = sorted({s for s in sids if s not in sub_to_fold})
    if missing:
        raise ValueError(
            f"{splits_csv} missing fold assignment for {len(missing)} subjects: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    fold_arr = np.array([sub_to_fold[s] for s in sids])
    folds: list[Fold] = []
    for f in sorted(np.unique(fold_arr)):
        val_mask = fold_arr == f
        val_subs = tuple(sorted(np.unique(sids[val_mask]).tolist()))
        folds.append(Fold(
            fold=int(f),
            train_idx=np.where(~val_mask)[0],
            val_idx=np.where(val_mask)[0],
            val_subjects=val_subs,
        ))
    return folds


def cv_iter(
    subject_ids: Sequence[str],
    splits_csv: Path | None = None,
    scheme: str = "groupkfold",
    n_splits: int = 5,
) -> Iterator[Fold]:
    """Yield Fold objects, one per outer CV fold.

    If splits_csv exists, it is used verbatim (subject -> fold mapping).
    Otherwise scheme drives the split: 'loso' uses LeaveOneGroupOut,
    'groupkfold' uses GroupKFold(n_splits).
    """
    sids = np.asarray([str(s) for s in subject_ids])

    if splits_csv is not None and Path(splits_csv).exists():
        yield from _from_baseline_csv(Path(splits_csv), sids)
        return

    if scheme == "loso":
        splitter = LeaveOneGroupOut()
    elif scheme == "groupkfold":
        splitter = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"unknown cv scheme: {scheme!r}")

    for f, (tr, va) in enumerate(splitter.split(np.zeros(len(sids)), groups=sids)):
        yield Fold(
            fold=f,
            train_idx=tr,
            val_idx=va,
            val_subjects=tuple(sorted(np.unique(sids[va]).tolist())),
        )
