"""Train multi-task 1D-CNN on per-window engineered features.

The same model encodes ALL FOUR tasks (exercise, phase, fatigue, reps) via
hard parameter sharing — one shared CNN encoder + 4 task heads.

Run:
    python scripts/train_cnn1d.py --run-slug cnn1d-baseline
    python scripts/train_cnn1d.py --smoke-test    # 1 fold × 1 seed × 3 epochs

References:
- Yang et al. 2015 — 1D-CNN for multichannel sensor data
- Caruana 1997 — multi-task hard parameter sharing
- Saeb et al. 2017 — subject-wise CV
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import run_training, parse_common_args


def main():
    args = parse_common_args().parse_args()
    run_training(
        arch_name='cnn1d',
        run_slug=args.run_slug,
        labeled_root=args.labeled_root,
        runs_root=args.runs_root,
        splits_path=args.splits,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
    )


if __name__ == '__main__':
    main()
