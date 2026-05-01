"""Benchmark DataLoader throughput for various num_workers settings.

For each (variant, num_workers) combination, time one epoch worth of batches
through the dataset + a tiny model forward pass. Reports samples/sec.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import WindowFeatureDataset
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.models.mlp import MLPMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask


def find_files(pattern: str):
    return sorted(Path('data/labeled').rglob(pattern))


def bench_features(workers_list, n_batches=200, batch_size=64):
    print(f"\n=== FEATURES (WindowFeatureDataset, MLP) ===")
    files = find_files('window_features.parquet')
    ds = WindowFeatureDataset(window_parquets=files, active_only=True, verbose=False)
    print(f"Dataset: {len(ds)} windows, {ds.n_features} features")
    indices = list(range(min(n_batches * batch_size, len(ds))))
    sub = Subset(ds, indices)
    model = MLPMultiTask(ds.n_features, ds.n_exercise, ds.n_phase).cuda()

    for nw in workers_list:
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=(nw > 0))
        # warmup
        it = iter(loader); next(it); del it
        t0 = time.time()
        n_samples = 0
        for batch in loader:
            x = batch['x'].cuda(non_blocking=True)
            with torch.amp.autocast('cuda'):
                _ = model(x)
            n_samples += x.size(0)
        elapsed = time.time() - t0
        rate = n_samples / elapsed
        print(f"  num_workers={nw:>2}: {n_samples} samples in {elapsed:.1f}s = {rate:.0f} samples/s")


def bench_raw(workers_list, n_batches=200, batch_size=64):
    print(f"\n=== RAW (RawMultimodalWindowDataset, TCN-raw) ===")
    files = find_files('aligned_features.parquet')
    ds = RawMultimodalWindowDataset(parquet_paths=files, active_only=True, verbose=False)
    print(f"Dataset: {len(ds)} windows, {ds.n_channels}x{ds.n_timesteps}")
    indices = list(range(min(n_batches * batch_size, len(ds))))
    sub = Subset(ds, indices)
    model = TCNRawMultiTask(n_channels=ds.n_channels, n_timesteps=ds.n_timesteps,
                              n_exercise=ds.n_exercise, n_phase=ds.n_phase).cuda()

    for nw in workers_list:
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True,
                             persistent_workers=(nw > 0))
        it = iter(loader); next(it); del it
        t0 = time.time()
        n_samples = 0
        for batch in loader:
            x = batch['x'].cuda(non_blocking=True)
            with torch.amp.autocast('cuda'):
                _ = model(x)
            n_samples += x.size(0)
        elapsed = time.time() - t0
        rate = n_samples / elapsed
        print(f"  num_workers={nw:>2}: {n_samples} samples in {elapsed:.1f}s = {rate:.0f} samples/s")


if __name__ == '__main__':
    # Test 0, 2, 4, 8, 12
    workers = [0, 2, 4, 8, 12]
    bench_features(workers)
    bench_raw(workers)
