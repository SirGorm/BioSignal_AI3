"""End-to-end training entrypoint for src2.

Replaces `src/pipeline/train_nn.py` + `src/training/loop.py:run_cv` with
Lightning's `Trainer` driving a per-fold loop. Supports two input variants:

    --variant raw        → AlignedWindowDataset, encoders cnn1d/lstm/cnn_lstm/tcn
    --variant features   → WindowFeatureDataset, any encoder including mlp

Phase 1 (--n-trials > 0) does an Optuna search on a single fold; Phase 2
trains with the best hyperparameters across all CV folds × multiple seeds.
With --n-trials 0 (default) Phase 1 is skipped and Phase 2 runs with
config-default HPs.

Usage
-----
    # Full default run (raw, all archs, all tasks):
    python -m src2.pipeline.train --variant raw --arch tcn

    # MLP on features, exercise head only, 1 Optuna trial + full Phase 2:
    python -m src2.pipeline.train \\
        --variant features --arch mlp --tasks exercise \\
        --n-trials 1 --phase2-seeds 42 1337 7

    # OmegaConf dotted overrides:
    python -m src2.pipeline.train --arch tcn training.batch_size=128
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
import numpy as np
import optuna
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset

from src2.config import load_config, resolve_paths
from src2.data.parquet_dataset import AlignedWindowDataset
from src2.data.feature_dataset import WindowFeatureDataset
from src2.data.splits import cv_iter
from src2.training.data_module import FoldDataModule
from src2.training.lit_module import LitMultiTask

ARCH_CHOICES = ("cnn1d", "lstm", "cnn_lstm", "tcn", "mlp", "all")
VARIANT_CHOICES = ("raw", "features")
TASK_CHOICES = ("exercise", "phase", "fatigue", "reps")


def _parse() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="Lightning multi-task training (src2)")
    p.add_argument("--arch", choices=ARCH_CHOICES, default="tcn")
    p.add_argument("--variant", choices=VARIANT_CHOICES, default="raw")
    p.add_argument(
        "--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASK_CHOICES),
        help="Which task heads contribute to the loss. Other heads are still "
             "computed but their gradients do not update the encoder.",
    )
    p.add_argument("--features-parquet", type=Path, default=None,
                   help="Path to window_features.parquet for --variant features.")
    p.add_argument("--n-trials", type=int, default=0,
                   help="Optuna Phase-1 trials. 0 = skip Phase 1.")
    p.add_argument("--phase1-epochs", type=int, default=15)
    p.add_argument("--phase2-epochs", type=int, default=None,
                   help="Phase-2 epochs. Defaults to cfg.training.epochs.")
    p.add_argument("--phase2-seeds", type=int, nargs="+", default=None,
                   help="Phase-2 seeds. Defaults to cfg.training.seeds.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out", default=None)
    return p.parse_known_args()


# ---- dataset factory --------------------------------------------------------


def _build_dataset(cfg, variant: str, features_parquet: Path | None) -> Dataset:
    if variant == "raw":
        labeled_root = Path(cfg.paths.labeled_dir)
        parquets = sorted(labeled_root.glob("recording_*/aligned_features.parquet"))
        if not parquets:
            raise FileNotFoundError(
                f"No aligned_features.parquet under {labeled_root}. "
                "Run /label first."
            )
        return AlignedWindowDataset(
            parquet_paths=parquets,
            channels=tuple(cfg.modalities.raw_channels),
            window_s=cfg.windows.dataset.window_s,
            hop_s=cfg.windows.dataset.hop_s,
            active_only=False,
            target_modes=dict(cfg.soft_targets.default_modes),
        )

    # variant == features
    if features_parquet is None:
        # Auto-discover: pick the most recent runs/*/window_features.parquet.
        candidates = sorted(
            Path("runs").glob("**/window_features.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No window_features.parquet under runs/. "
                "Run /train first or pass --features-parquet."
            )
        features_parquet = candidates[0]
        print(f"[src2] auto-selected features parquet: {features_parquet}")
    return WindowFeatureDataset(parquet_paths=[features_parquet], active_only=False)


# ---- encoder kwargs ---------------------------------------------------------


def _encoder_kwargs_default(cfg, arch: str, variant: str) -> dict:
    if variant == "raw":
        section = cfg.models_nn.raw
    else:
        section = cfg.models_nn.features
    arch_cfg = dict(section.get(arch, {}))
    arch_cfg.pop("repr_dim", None)
    return arch_cfg


# ---- one-fold training ------------------------------------------------------


def _train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_subjects: list,
    arch: str,
    variant: str,
    seed: int,
    cfg,
    dataset: Dataset,
    encoder_kwargs: dict,
    repr_dim: int,
    head_dropout: float,
    enabled_tasks: list[str],
    epochs: int,
    out_dir: Path,
    is_phase2: bool,
) -> dict:
    L.seed_everything(seed, workers=True)

    if variant == "raw":
        n_input = len(dataset.channels)  # type: ignore[attr-defined]
    else:
        n_input = dataset.n_features  # type: ignore[attr-defined]

    dm = FoldDataModule(
        dataset=dataset,
        train_window_idx=train_idx,
        val_window_idx=val_idx,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )

    model = LitMultiTask(
        arch=arch,
        n_channels=n_input,  # MultiTaskModel routes this to the right encoder kw
        n_exercise=dataset.n_exercise,  # type: ignore[attr-defined]
        n_phase=dataset.n_phase,  # type: ignore[attr-defined]
        repr_dim=repr_dim,
        encoder_kwargs=encoder_kwargs,
        head_dropout=head_dropout,
        loss_weights=dict(cfg.losses.weights),
        target_modes=dict(cfg.soft_targets.default_modes),
        enabled_tasks=list(enabled_tasks),
        uncertainty=cfg.training.uncertainty_weighting,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        epochs=epochs,
    )

    callbacks = [
        EarlyStopping(
            monitor="val/loss/total", mode="min", patience=cfg.training.patience
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if is_phase2:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=out_dir / "ckpt",
                monitor="val/loss/total",
                mode="min",
                save_top_k=1,
                filename="best",
            ),
        )

    logger = TensorBoardLogger(save_dir=str(out_dir / "tb"), name="", version="")

    trainer = L.Trainer(
        max_epochs=epochs,
        precision="16-mixed" if cfg.training.mixed_precision else "32-true",
        gradient_clip_val=cfg.training.grad_clip,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        log_every_n_steps=20,
        deterministic=False,
        # Sanity-check val batches are sequential from the start of the val
        # set, which on this dataset is the baseline rest period — no valid
        # exercise/phase labels. Skip it so torchmetrics doesn't warn before
        # the first real epoch updates the metrics.
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
    val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]

    return {
        "fold": int(fold),
        "seed": int(seed),
        "test_subjects": list(test_subjects),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
    }


# ---- Optuna phase 1 ---------------------------------------------------------


def _suggest_hp(trial: optuna.Trial, arch: str) -> dict:
    """One small search space — sane defaults; widen if you need more variance."""
    hp = {
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "repr_dim": trial.suggest_categorical("repr_dim", [32, 64, 128]),
    }
    if arch == "mlp":
        hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [40, 80, 160])
    elif arch == "tcn":
        hp["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
    elif arch == "lstm":
        hp["hidden"] = trial.suggest_categorical("hidden", [24, 48, 96])
    return hp


def _phase1_optuna(
    arch: str,
    variant: str,
    cfg,
    dataset: Dataset,
    folds: list,
    n_trials: int,
    epochs: int,
    enabled_tasks: list[str],
    out_dir: Path,
) -> dict:
    """Run Optuna n_trials on the first fold. Returns the best HPs."""
    fold = folds[0]
    fold_id, train_idx, val_idx, test_subs = (
        fold["fold"], fold["train_idx"], fold["val_idx"], fold["test_subs"],
    )

    base_kwargs = _encoder_kwargs_default(cfg, arch, variant)

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_hp(trial, arch)
        cfg.training.lr = hp["lr"]
        encoder_kwargs = dict(base_kwargs)
        for k, v in hp.items():
            if k in ("lr",):
                continue
            encoder_kwargs[k] = v if k != "repr_dim" else encoder_kwargs.get("repr_dim")
        repr_dim = hp["repr_dim"]
        head_d = hp["dropout"]
        trial_dir = out_dir / "phase1" / f"trial_{trial.number:03d}"
        try:
            res = _train_one_fold(
                fold=fold_id,
                train_idx=train_idx,
                val_idx=val_idx,
                test_subjects=test_subs,
                arch=arch,
                variant=variant,
                seed=42,
                cfg=cfg,
                dataset=dataset,
                encoder_kwargs=encoder_kwargs,
                repr_dim=repr_dim,
                head_dropout=head_d,
                enabled_tasks=enabled_tasks,
                epochs=epochs,
                out_dir=trial_dir,
                is_phase2=False,
            )
        except Exception as e:
            print(f"[phase1 trial {trial.number}] failed: {e}")
            return float("inf")
        # Single-task → use that task's val loss. Multi-task → use total.
        m = res["val_metrics"]
        if len(enabled_tasks) == 1:
            return float(m.get(f"val/loss/{enabled_tasks[0]}", m.get("val/loss/total", float("inf"))))
        return float(m.get("val/loss/total", float("inf")))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[phase1] best params: {study.best_params}  best_value={study.best_value:.4f}")
    return study.best_params


# ---- main -------------------------------------------------------------------


def _materialise_folds(subjects, splits_csv, scheme, n_splits) -> list[dict]:
    """One-shot: collect folds from cv_iter into a list (so we can re-iterate)."""
    out = []
    for fold, train_idx, val_idx, test_subs in cv_iter(
        subjects, splits_csv=splits_csv, fallback_n_splits=n_splits, scheme=scheme,
    ):
        out.append({
            "fold": fold,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_subs": test_subs,
        })
    return out


def main() -> None:
    args, dotlist = _parse()
    cfg = resolve_paths(load_config(args.config, cli_overrides=dotlist))

    phase2_epochs = args.phase2_epochs if args.phase2_epochs is not None else cfg.training.epochs
    seeds = args.phase2_seeds if args.phase2_seeds else list(cfg.training.seeds)
    archs = (
        ["cnn1d", "lstm", "cnn_lstm", "tcn"] if args.arch == "all" else [args.arch]
    )
    if args.variant == "raw" and "mlp" in archs:
        raise SystemExit("--arch mlp requires --variant features")

    out_dir = Path(
        args.out
        or f"runs/{datetime.now():%Y%m%d_%H%M%S}_src2_{args.variant}_{'-'.join(archs)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[src2] output: {out_dir}")
    print(f"[src2] tasks: {args.tasks}  variant: {args.variant}  archs: {archs}")
    print(f"[src2] phase1 trials: {args.n_trials}  phase2 seeds: {seeds}")

    # Build the dataset once; reuse across folds × seeds × archs.
    dataset = _build_dataset(cfg, args.variant, args.features_parquet)
    folds = _materialise_folds(
        dataset.subject_ids,  # type: ignore[attr-defined]
        splits_csv=cfg.paths.splits_csv,
        scheme=cfg.cv.scheme,
        n_splits=cfg.cv.n_splits,
    )
    print(f"[src2] dataset: {len(dataset)} samples  folds: {len(folds)}")

    all_results: list[dict] = []
    for arch in archs:
        base_kwargs = _encoder_kwargs_default(cfg, arch, args.variant)
        # Phase 1
        if args.n_trials > 0:
            best_hp = _phase1_optuna(
                arch=arch, variant=args.variant, cfg=cfg, dataset=dataset,
                folds=folds, n_trials=args.n_trials, epochs=args.phase1_epochs,
                enabled_tasks=args.tasks, out_dir=out_dir / arch,
            )
        else:
            best_hp = {}
        with open(out_dir / arch_safe_name(arch, "best_hp.json"), "w") as f:
            json.dump({"arch": arch, "best_hp": best_hp}, f, indent=2)

        # Phase 2: full CV × seeds with best HPs.
        encoder_kwargs = dict(base_kwargs)
        for k, v in best_hp.items():
            if k in ("lr",):
                cfg.training.lr = v
                continue
            if k == "repr_dim":
                continue  # passed separately below
            encoder_kwargs[k] = v
        repr_dim = best_hp.get("repr_dim", base_kwargs.get("repr_dim", 64))
        head_dropout = best_hp.get("dropout", base_kwargs.get("dropout", 0.3))

        for seed in seeds:
            for fold in folds:
                fold_dir = out_dir / arch / "phase2" / f"seed_{seed}" / f"fold_{fold['fold']}"
                print(
                    f"[src2 phase2] arch={arch} seed={seed} fold={fold['fold']} "
                    f"train={len(fold['train_idx'])} val={len(fold['val_idx'])}"
                )
                res = _train_one_fold(
                    fold=fold["fold"],
                    train_idx=fold["train_idx"],
                    val_idx=fold["val_idx"],
                    test_subjects=fold["test_subs"],
                    arch=arch,
                    variant=args.variant,
                    seed=seed,
                    cfg=cfg,
                    dataset=dataset,
                    encoder_kwargs=encoder_kwargs,
                    repr_dim=repr_dim,
                    head_dropout=head_dropout,
                    enabled_tasks=args.tasks,
                    epochs=phase2_epochs,
                    out_dir=fold_dir,
                    is_phase2=True,
                )
                res["arch"] = arch
                res["variant"] = args.variant
                res["tasks"] = list(args.tasks)
                all_results.append(res)
                with open(out_dir / "results.json", "w") as f:
                    json.dump(all_results, f, indent=2)

    print(f"[src2] done. results -> {out_dir / 'results.json'}")


def arch_safe_name(arch: str, fname: str) -> str:
    """Used to write per-arch metadata files at the run root."""
    Path("runs").mkdir(exist_ok=True)
    return f"{arch}_{fname}"


if __name__ == "__main__":
    main()
