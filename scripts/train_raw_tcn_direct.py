"""Train multi-raw-TCN directly (no Optuna HP search).

Uses sensible default HPs (lifted from a previous v12 best_hps where
available, otherwise reasonable defaults). Runs the same 7-fold LOSO ×
N seeds CV that Phase 2 of train_optuna.py would run, and writes a
cv_summary.json so the standard plotting / aggregation tools work.

Goal: fast feedback loop to A/B-test the new norm_mode options without
waiting 2+ hours per HP search. ~10-30 min depending on epochs/seeds.

Usage:
    python scripts/train_raw_tcn_direct.py --tag tcn_pct_norm \
        --norm-mode percentile --window-s 2.0 --epochs 100 --seeds 42

    # Compare modes side-by-side:
    for mode in baseline robust percentile; do
        python scripts/train_raw_tcn_direct.py --tag tcn_${mode}_w2s \
            --norm-mode $mode --window-s 2.0 --epochs 100 --seeds 42
    done
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.models.raw.tcn_raw import TCNRawMultiTask
from src.training.loop import TrainConfig, run_cv

EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

# Default HPs — taken from the v12 best_hps for w2s-multi-raw-tcn (a
# reasonable starting point; adjust via CLI if needed).
DEFAULT_HPS = {
    "lr": 1.25e-3,
    "weight_decay": 5e-5,
    "batch_size": 64,
    "dropout": 0.15,
    "repr_dim": 64,
    "tcn_kernel": 7,
}


def parse_splits(splits_path: Path):
    import csv
    from collections import defaultdict
    folds = defaultdict(lambda: {"train": [], "test": []})
    with splits_path.open() as f:
        for row in csv.DictReader(f):
            fold = int(row["fold"])
            folds[fold]["test"].append(row["subject_id"])
    return folds


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--norm-mode",
                    choices=["baseline", "robust", "percentile"],
                    default="baseline")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--lr", type=float, default=DEFAULT_HPS["lr"])
    ap.add_argument("--weight-decay", type=float,
                    default=DEFAULT_HPS["weight_decay"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_HPS["batch_size"])
    ap.add_argument("--dropout", type=float, default=DEFAULT_HPS["dropout"])
    ap.add_argument("--repr-dim", type=int, default=DEFAULT_HPS["repr_dim"])
    ap.add_argument("--tcn-kernel", type=int, default=DEFAULT_HPS["tcn_kernel"])
    ap.add_argument("--labeled-root", type=Path,
                    default=ROOT / "data" / "labeled_clean")
    ap.add_argument("--splits", type=Path,
                    default=ROOT / "configs" / "splits_clean_loso.csv")
    args = ap.parse_args()

    out_root = ROOT / "runs" / f"direct_tcn-{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(args.labeled_root.rglob("aligned_features.parquet"))
    files = [p for p in files if not any(x in str(p) for x in EXCLUDE)]
    print(f"[direct] {len(files)} recordings after exclusion")

    dataset = RawMultimodalWindowDataset(
        parquet_paths=files, active_only=True,
        target_modes={"phase": "soft", "reps": "soft_overlap"},
        window_s=args.window_s, norm_mode=args.norm_mode,
    )
    print(f"[direct] {len(dataset)} active windows  norm_mode={args.norm_mode}")

    # Build LOSO folds matching v12 — group by subject from splits CSV
    fold_subjects = parse_splits(args.splits)
    n_folds = len(fold_subjects)
    print(f"[direct] {n_folds} folds")

    import numpy as np
    subj_per_window = np.array(dataset.subject_ids)
    splits = []
    for fold_id in sorted(fold_subjects):
        test_subs = set(fold_subjects[fold_id]["test"])
        test_idx = np.where(np.isin(subj_per_window, list(test_subs)))[0]
        train_idx = np.where(~np.isin(subj_per_window, list(test_subs)))[0]
        if len(test_idx) == 0:
            print(f"  [warn] fold {fold_id} has 0 test windows — skipping")
            continue
        splits.append({
            "fold": fold_id, "train_idx": train_idx, "test_idx": test_idx,
            "test_subjects": list(test_subs),
        })

    # Materialize on GPU for speed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dataset.materialize_to_device(device)
        print(f"[direct] dataset materialized on {device}")

    # Model factory
    n_t = dataset.n_timesteps
    n_c = dataset.n_channels

    def factory():
        return TCNRawMultiTask(
            n_channels=n_c, n_timesteps=n_t,
            n_exercise=dataset.n_exercise, n_phase=dataset.n_phase,
            kernel_size=args.tcn_kernel,
            repr_dim=args.repr_dim, dropout=args.dropout,
        )

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=1.0, patience=args.patience,
        mixed_precision=True, num_workers=0,
        use_uncertainty_weighting=True,
        target_modes={"phase": "soft", "reps": "soft_overlap"},
        enabled_tasks=["exercise", "phase", "fatigue", "reps"],
    )

    summary, _ = run_cv(
        dataset=dataset, model_factory=factory, arch_name="tcn_raw_direct",
        cfg=cfg, splits=splits, out_root=out_root, seeds=tuple(args.seeds),
    )

    print("\n=== SUMMARY ===")
    print(f"norm_mode={args.norm_mode}  window_s={args.window_s}  "
          f"epochs={args.epochs}  seeds={args.seeds}")
    for task in ["exercise", "phase", "fatigue", "reps"]:
        block = summary.get(task, {})
        if block.get("untrained"):
            continue
        for metric, stats in block.items():
            if isinstance(stats, dict) and "mean" in stats:
                print(f"  {task:9} {metric:18} = {stats['mean']:.4f} "
                      f"± {stats['std']:.4f} (n={stats['n']})")


if __name__ == "__main__":
    main()
