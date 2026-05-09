"""LightningDataModule wrapping a pre-built dataset for one CV fold.

Accepts any dataset with the {x, targets, masks} item contract — works for
both `AlignedWindowDataset` (raw signals) and `WindowFeatureDataset`
(per-window engineered features).
"""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class FoldDataModule(L.LightningDataModule):
    """One LightningDataModule per CV fold over a shared dataset instance."""

    def __init__(
        self,
        dataset: Dataset,
        train_window_idx: np.ndarray,
        val_window_idx: np.ndarray,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self._dataset = dataset
        self.train_window_idx = np.asarray(train_window_idx)
        self.val_window_idx = np.asarray(val_window_idx)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        return None  # dataset already built

    @property
    def dataset(self) -> Any:
        return self._dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self._dataset, self.train_window_idx.tolist()),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self._dataset, self.val_window_idx.tolist()),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
