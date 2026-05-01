"""Train multi-task TCN on per-window engineered features.

The same model encodes ALL FOUR tasks (exercise, phase, fatigue, reps) via
hard parameter sharing — one shared TCN encoder + 4 task heads.

The TCN is strictly causal by design (dilated causal convolutions). When
Phase 2 trains the same architecture on raw signals, this same model class
is deployable in src/streaming/realtime.py without modification.

Run:
    python scripts/train_tcn.py --run-slug tcn-baseline
    python scripts/train_tcn.py --smoke-test

References:
- Bai et al. 2018 — TCN
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
        arch_name='tcn',
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
    )


if __name__ == '__main__':
    main()
