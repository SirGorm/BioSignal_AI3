"""LightningDataModule that wires (Dataset, Fold) → train/val DataLoaders.

Plain DataLoader with pin_memory + persistent_workers. Lightning handles
device transfer, shuffling, and worker lifecycle. No custom GPU iterator.
"""

from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader, Subset

from src3.data.splits import Fold


class FoldDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        fold: Fold,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        self._train = Subset(self.dataset, self.fold.train_idx.tolist())
        self._val = Subset(self.dataset, self.fold.val_idx.tolist())

    def _loader(self, subset, shuffle: bool) -> DataLoader:
        # When the underlying dataset has been moved to GPU (its tensors
        # already live on cuda), pin_memory and num_workers > 0 are both
        # wrong: pin_memory tries to pin GPU tensors → error; workers fork
        # the dataset, duplicating GPU memory per worker → OOM.
        on_gpu = (
            getattr(self.dataset, "_device", None) is not None
            and self.dataset._device.type == "cuda"
        )
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0 if on_gpu else self.num_workers,
            pin_memory=False if on_gpu else self.pin_memory,
            persistent_workers=(self.num_workers > 0) and not on_gpu,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)
