"""Train multi-task CNN-LSTM (DeepConvLSTM-style) on per-window features.

The same model encodes ALL FOUR tasks (exercise, phase, fatigue, reps) via
hard parameter sharing — one shared CNN-LSTM encoder + 4 task heads.

Run:
    python scripts/train_cnn_lstm.py --run-slug cnn-lstm-baseline
    python scripts/train_cnn_lstm.py --smoke-test

References:
- Ordóñez & Roggen 2016 — DeepConvLSTM, the canonical reference for multimodal
  wearable activity recognition (closest analogue to this project's setup)
- Karpathy et al. 2014 — CNN+LSTM origin
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
        arch_name='cnn_lstm',
        run_slug=args.run_slug,
        labeled_root=args.labeled_root,
        runs_root=args.runs_root,
        splits_path=args.splits,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke_test=args.smoke_test,
        use_uncertainty_weighting=args.uncertainty_weighting,
        phase_whitelist_path=args.phase_whitelist,
        exercise_aggregation=args.exercise_aggregation,
    )


if __name__ == '__main__':
    main()
