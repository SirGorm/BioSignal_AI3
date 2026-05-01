"""CLI: generate training curves + confusion matrices + fatigue calibration
for one or more run directories.

Run:
    python scripts/generate_plots.py --runs runs/<ts>_nn-full-tcn ...
    python scripts/generate_plots.py --runs runs/<ts>_*  # glob expansion
"""

from __future__ import annotations
from pathlib import Path

import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.plotting import plot_everything_for_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True,
                    help='Run directories to plot. Each should contain '
                         'history.json files under <arch>/seed_*/fold_*/')
    args = p.parse_args()
    for run_dir in args.runs:
        if not run_dir.exists():
            print(f"[plot] SKIP missing: {run_dir}")
            continue
        plot_everything_for_run(run_dir)
    print("[plot] Done.")


if __name__ == '__main__':
    main()
