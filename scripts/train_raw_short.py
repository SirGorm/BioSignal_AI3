"""Short raw-signal training for all 4 architectures.

Trains cnn1d_raw, lstm_raw, cnn_lstm_raw, tcn_raw on RawMultimodalWindowDataset
with 1 seed, 1 fold, 5 epochs each. Saves test_preds.pt per fold (via run_cv)
so confusion matrices and calibration plots can be generated afterwards.

Run:
    python scripts/train_raw_short.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.models.raw.cnn1d_raw import CNN1DRawMultiTask
from src.models.raw.lstm_raw import LSTMRawMultiTask
from src.models.raw.cnn_lstm_raw import CNNLSTMRawMultiTask
from src.models.raw.tcn_raw import TCNRawMultiTask
from src.training.cv import load_or_generate_splits
from src.training.loop import TrainConfig, run_cv


RAW_REGISTRY = {
    'cnn1d_raw':    CNN1DRawMultiTask,
    'lstm_raw':     LSTMRawMultiTask,
    'cnn_lstm_raw': CNNLSTMRawMultiTask,
    'tcn_raw':      TCNRawMultiTask,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archs', nargs='+', default=list(RAW_REGISTRY.keys()))
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n-folds', type=int, default=1,
                   help='Number of folds to run (≤5). 1 = quickest verification.')
    p.add_argument('--labeled-root', type=Path, default=Path('data/labeled'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    p.add_argument('--splits', type=Path, default=Path('configs/splits.csv'))
    p.add_argument('--uncertainty-weighting', action='store_true')
    p.add_argument('--phase-whitelist', type=Path, default=None)
    p.add_argument('--exclude-recordings', nargs='*', default=[])
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--run-slug-prefix', type=str, default='short')
    args = p.parse_args()

    parquets = sorted(args.labeled_root.rglob('aligned_features.parquet'))
    if not parquets:
        raise FileNotFoundError(f"No aligned_features.parquet under {args.labeled_root}")
    if args.exclude_recordings:
        before = len(parquets)
        parquets = [p for p in parquets
                     if not any(ex in str(p) for ex in args.exclude_recordings)]
        print(f"[raw_short] Excluded {before - len(parquets)} parquet(s)")

    from src.data.phase_whitelist import load_phase_whitelist
    phase_wl = load_phase_whitelist(args.phase_whitelist)
    if phase_wl is not None:
        print(f"[raw_short] Phase whitelist: {len(phase_wl)} (recording, set) pairs")

    print(f"[raw_short] Loading {len(parquets)} aligned_features.parquet files")
    dataset = RawMultimodalWindowDataset(parquets, active_only=True,
                                           phase_whitelist=phase_wl)
    print(f"[raw_short] {len(dataset)} windows, "
          f"{dataset.n_channels} channels x {dataset.n_timesteps} timesteps")

    subject_ids = np.array(dataset.subject_ids)
    folds_full = load_or_generate_splits(subject_ids, splits_path=args.splits)
    folds = folds_full[:args.n_folds]
    print(f"[raw_short] Running {len(folds)}/{len(folds_full)} folds")

    for arch_name in args.archs:
        cls = RAW_REGISTRY[arch_name]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = args.runs_root / f"{timestamp}_{args.run_slug_prefix}-{arch_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[raw_short] === {arch_name} -> {run_dir} ===")

        meta = {
            'arch': arch_name,
            'input_variant': 'raw',
            'n_channels': dataset.n_channels,
            'n_timesteps': dataset.n_timesteps,
            'exercise_classes': list(dataset.exercise_encoder.classes_),
            'phase_classes':    list(dataset.phase_encoder.classes_),
            'epochs': args.epochs,
            'seeds': list(args.seeds),
            'n_folds': len(folds),
        }
        (run_dir / 'dataset_meta.json').write_text(json.dumps(meta, indent=2))

        def factory(_cls=cls, _ds=dataset):
            return _cls(
                n_channels=_ds.n_channels,
                n_timesteps=_ds.n_timesteps,
                n_exercise=_ds.n_exercise,
                n_phase=_ds.n_phase,
            )

        sample = factory()
        n_params = sum(p.numel() for p in sample.parameters())
        print(f"[raw_short] {arch_name} parameters: {n_params:,}")
        del sample

        cfg = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=1e-4, grad_clip=1.0, patience=8,
            mixed_precision=True, num_workers=args.num_workers,
            use_uncertainty_weighting=args.uncertainty_weighting,
        )
        (run_dir / 'train_config.json').write_text(json.dumps(cfg.__dict__, indent=2))

        summary, _ = run_cv(
            dataset=dataset,
            model_factory=factory,
            arch_name=arch_name,
            cfg=cfg,
            splits=folds,
            out_root=run_dir,
            seeds=tuple(args.seeds),
        )
        print(f"[raw_short] {arch_name} summary:")
        print(json.dumps(summary, indent=2)[:600])


if __name__ == '__main__':
    main()
