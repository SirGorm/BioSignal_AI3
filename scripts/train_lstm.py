"""Train multi-task BiLSTM on per-window engineered features.

The same model encodes ALL FOUR tasks (exercise, phase, fatigue, reps) via
hard parameter sharing — one shared BiLSTM encoder + 4 task heads.

NOTE: Bidirectional LSTM is non-causal — the deployed real-time model would
use a unidirectional variant. For Phase 1 offline comparison, BiLSTM is fine
and is the standard reference (Schuster & Paliwal 1997).

Run:
    python scripts/train_lstm.py --run-slug lstm-baseline
    python scripts/train_lstm.py --smoke-test

References:
- Hochreiter & Schmidhuber 1997 — LSTM
- Schuster & Paliwal 1997 — bidirectional RNN
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
        arch_name='lstm',
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
