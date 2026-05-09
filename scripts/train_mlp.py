"""Train multi-task MLP on per-window engineered features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import run_training, parse_common_args


def main():
    args = parse_common_args().parse_args()
    run_training(
        arch_name='mlp',
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
