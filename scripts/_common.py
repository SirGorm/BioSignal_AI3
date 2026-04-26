"""Shared training entry point for the 4 architecture scripts.

Each architecture script (train_cnn1d, train_lstm, train_cnn_lstm, train_tcn)
is just a thin wrapper that calls run_training() with its arch_name. This keeps
model definitions cleanly separated from the training loop:

  src/models/<arch>.py  — model definition
  src/training/loop.py  — training loop (architecture-agnostic)
  scripts/_common.py    — wires data + model + loop + CV together
  scripts/train_<arch>.py — selects which architecture to train
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import argparse
import json
import sys

# Make repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from src.data.datasets import WindowFeatureDataset
from src.models.cnn1d import CNN1DMultiTask
from src.models.lstm import LSTMMultiTask
from src.models.cnn_lstm import CNNLSTMMultiTask
from src.models.tcn import TCNMultiTask
from src.training.cv import load_or_generate_splits
from src.training.loop import TrainConfig, run_cv


# Architecture registry — maps name to model class.
# All take the same constructor signature: (n_features, n_exercise, n_phase, **kwargs)
ARCH_REGISTRY = {
    'cnn1d':    CNN1DMultiTask,
    'lstm':     LSTMMultiTask,
    'cnn_lstm': CNNLSTMMultiTask,
    'tcn':      TCNMultiTask,
}


def find_window_feature_files(labeled_root: Path) -> List[Path]:
    files = sorted(labeled_root.rglob('window_features.parquet'))
    if not files:
        raise FileNotFoundError(
            f"No window_features.parquet under {labeled_root}. "
            f"Run /label and the biosignal-feature-extractor first."
        )
    return files


def make_model_factory(arch_name: str, n_features: int,
                        n_exercise: int, n_phase: int,
                        repr_dim: int = 128, dropout: float = 0.3
                        ) -> Callable[[], nn.Module]:
    """Returns a 0-arg factory that constructs the chosen architecture.
    Re-instantiated per fold/seed to ensure independent initialization."""
    if arch_name not in ARCH_REGISTRY:
        raise ValueError(
            f"Unknown arch '{arch_name}'. Choose from: {list(ARCH_REGISTRY)}"
        )
    cls = ARCH_REGISTRY[arch_name]

    def factory():
        return cls(
            n_features=n_features,
            n_exercise=n_exercise,
            n_phase=n_phase,
            repr_dim=repr_dim,
            dropout=dropout,
        )
    return factory


def run_training(
    arch_name: str,
    run_slug: Optional[str] = None,
    labeled_root: Path = Path('data/labeled'),
    runs_root: Path = Path('runs'),
    splits_path: Path = Path('configs/splits.csv'),
    seeds: List[int] = (42, 1337, 7),
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    smoke_test: bool = False,
):
    if smoke_test:
        seeds = (42,)
        epochs = 3
        print("[smoke_test] 1 seed × 1 fold × 3 epochs to verify pipeline.")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slug = run_slug or f"nn_features_{arch_name}"
    run_dir = runs_root / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] Output directory: {run_dir}")

    win_paths = find_window_feature_files(labeled_root)
    print(f"[run] Loading {len(win_paths)} window_features.parquet file(s)")
    dataset = WindowFeatureDataset(window_parquets=win_paths, active_only=True)
    print(f"[run] Dataset: {len(dataset)} windows, {dataset.n_features} features, "
          f"{dataset.n_exercise} exercise classes, {dataset.n_phase} phase classes.")

    subject_ids = np.array(dataset.subject_ids)
    folds = load_or_generate_splits(subject_ids, splits_path=splits_path)
    if smoke_test:
        folds = folds[:1]

    encoders = {
        'exercise_classes': dataset.exercise_encoder.classes_,
        'phase_classes':    dataset.phase_encoder.classes_,
        'feature_cols':     dataset.feature_cols,
        'n_features':       dataset.n_features,
        'arch':             arch_name,
    }
    with open(run_dir / 'dataset_meta.json', 'w') as f:
        json.dump(encoders, f, indent=2)

    factory = make_model_factory(
        arch_name=arch_name,
        n_features=dataset.n_features,
        n_exercise=dataset.n_exercise,
        n_phase=dataset.n_phase,
    )

    # Sanity-check: instantiate once and report parameter count
    sample_model = factory()
    n_params = sum(p.numel() for p in sample_model.parameters())
    print(f"[run] Architecture: {arch_name}, parameters: {n_params:,}")
    del sample_model

    cfg = TrainConfig(
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=1e-4, grad_clip=1.0, patience=8,
        mixed_precision=True, num_workers=2,
    )
    with open(run_dir / 'train_config.json', 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"[run] Starting CV: arch={arch_name} seeds={list(seeds)} "
          f"epochs={epochs} folds={len(folds)}")
    summary, all_results = run_cv(
        dataset=dataset,
        model_factory=factory,
        arch_name=arch_name,
        cfg=cfg,
        splits=folds,
        out_root=run_dir,
        seeds=seeds,
    )

    print("\n[run] Final summary:")
    print(json.dumps(summary, indent=2))
    return run_dir, summary


def parse_common_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-slug', type=str, default=None)
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 7])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--smoke-test', action='store_true',
                   help='1 seed × 1 fold × 3 epochs (verify pipeline only)')
    return p
