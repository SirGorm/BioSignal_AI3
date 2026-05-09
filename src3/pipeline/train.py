"""Lightning-driven multi-task training entrypoint for src3.

Replaces src/training/loop.py + src/pipeline/train_nn.py. One CLI driver
runs all 4 NN architectures on either raw windows or engineered features,
across all CV folds × seeds, with Lightning handling AMP, grad clip,
checkpoints, early stopping and TB logging.

Two-phase strategy (matches src/pipeline/train_nn.py):
    Phase 1 (--n-trials N) : Optuna search on the FIRST CV fold for N trials.
    Phase 2                : full CV × all seeds with the best HPs.

Usage
-----
    # Default: TCN on raw windows, full CV (uses configs/splits.csv)
    python -m src3.pipeline.train

    # MLP on features with 5 Optuna trials, then full Phase 2 (3 seeds × all folds)
    python -m src3.pipeline.train --variant features --arch mlp --n-trials 5

    # All 4 raw architectures, 30 epochs each
    python -m src3.pipeline.train --arch all --variant raw --epochs 30

    # OmegaConf dotted overrides
    python -m src3.pipeline.train training.batch_size=128 training.lr=2e-3
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src3.config import load_config, resolve_paths, PROJECT_ROOT
from src3.data.feature_dataset import WindowFeatureDataset
from src3.data.raw_window_dataset import AlignedWindowDataset, WindowSpec
from src3.data.splits import cv_iter
from src3.models.multitask import (
    MultiTaskModel, build_feature_encoder, build_raw_encoder,
)
from src3.training.data_module import FoldDataModule
from src3.training.lit_module import LitMultiTask
from src3.utils.device import torch_device


RAW_ARCHS = ("cnn1d", "lstm", "cnn_lstm", "tcn")
FEATURE_ARCHS = ("mlp",)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description="src3 Lightning multi-task trainer")
    # Alle defaults er None — faktiske defaults plukkes fra configs/src3.yaml
    # (`run.*` og `optuna.*` seksjonene) hvis CLI ikke setter dem.
    p.add_argument("--arch", default=None,
                   help=f"One of {RAW_ARCHS + FEATURE_ARCHS} or 'all'.")
    p.add_argument("--variant", choices=("raw", "features"), default=None)
    p.add_argument("--tasks", nargs="+",
                   choices=("exercise", "phase", "fatigue", "reps"),
                   default=None)
    p.add_argument("--epochs", type=int, default=None,
                   help="Phase-2 epochs (default: cfg.training.epochs).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Phase-2 seeds (default: cfg.training.seeds).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--features-parquet", type=Path, default=None,
                   help="Override cfg.paths.features_parquet.")
    p.add_argument("--labeled-root", type=Path, default=None,
                   help="Folder with recording_*/window_features.parquet files. "
                        "Used INSTEAD of --features-parquet when given.")
    p.add_argument("--exclude-recordings", nargs="+", default=None,
                   help="Recording folder names to exclude when using --labeled-root.")
    p.add_argument("--window-s", type=float, default=None,
                   help="Stride window for feature dataset (decimation + soft target column).")
    p.add_argument("--config", default="configs/src3.yaml")
    p.add_argument("--out", type=Path, default=None,
                   help="Run output directory. Default: <cfg.run.out_dir>/<ts>_src3_<arch>.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override cfg.optuna.n_trials.")
    p.add_argument("--phase1-epochs", type=int, default=None,
                   help="Override cfg.optuna.phase1_epochs.")
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap number of CV folds (debug/quick test).")
    return p.parse_known_args()


def _resolve_cli_defaults(args: argparse.Namespace, cfg) -> argparse.Namespace:
    """Fill in None-valued args from cfg.run / cfg.optuna sections."""
    if args.arch is None:
        args.arch = str(cfg.run.arch)
    if args.variant is None:
        args.variant = str(cfg.run.variant)
    if args.tasks is None:
        args.tasks = list(cfg.run.tasks)
    if args.n_trials is None:
        args.n_trials = int(cfg.optuna.n_trials)
    if args.phase1_epochs is None:
        args.phase1_epochs = int(cfg.optuna.phase1_epochs)
    return args


# ---- dataset / model ------------------------------------------------------


def _build_dataset(cfg, variant, features_parquet, labeled_root=None,
                   exclude_recordings=None, window_s=None):
    if variant == "raw":
        labeled = Path(cfg.paths.labeled_dir)
        parquets = sorted(labeled.glob("recording_*/aligned_features.parquet"))
        if not parquets:
            raise FileNotFoundError(
                f"No aligned_features.parquet under {labeled}. Run /label first."
            )
        spec = WindowSpec(
            window_s=float(cfg.windows.dataset.window_s),
            hop_s=float(cfg.windows.dataset.hop_s),
            norm_mode=str(cfg.windows.norm.mode),
            active_only=False,
        )
        return AlignedWindowDataset(
            parquet_paths=parquets,
            channels=tuple(cfg.modalities.raw_channels),
            spec=spec,
            target_modes=dict(cfg.soft_targets.default_modes),
        )
    # Variant=features supports two input modes:
    #   --labeled-root <dir>   : multiple recording_*/window_features.parquet
    #                            (matches v17's per-recording layout)
    #   --features-parquet <p> : a single concatenated parquet (legacy)
    if labeled_root is not None:
        parquets = sorted(Path(labeled_root).glob("recording_*/window_features.parquet"))
        if exclude_recordings:
            parquets = [p for p in parquets
                        if not any(ex in str(p) for ex in exclude_recordings)]
        if not parquets:
            raise FileNotFoundError(f"No recording_*/window_features.parquet under {labeled_root}")
        print(f"[src3] using labeled-root {labeled_root}: {len(parquets)} recordings")
    elif features_parquet is not None:
        parquets = [features_parquet]
    else:
        cfg_path = cfg.paths.get("features_parquet") if hasattr(cfg.paths, "get") else None
        if cfg_path and Path(cfg_path).exists():
            parquets = [Path(cfg_path)]
        else:
            cands = sorted(
                PROJECT_ROOT.glob("runs/**/window_features.parquet"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not cands:
                raise FileNotFoundError(
                    "Set cfg.paths.features_parquet, pass --features-parquet/--labeled-root."
                )
            parquets = [cands[0]]
            print(f"[src3] auto-selected features parquet: {cands[0]}")
    return WindowFeatureDataset(
        parquets=parquets,
        target_modes=dict(cfg.soft_targets.default_modes),
        active_only=False,
        window_s=float(window_s) if window_s is not None else 2.0,
    )


def _build_model(arch: str, variant: str, dataset, cfg, hp_overrides: dict | None = None,
                 ) -> tuple[MultiTaskModel, dict]:
    """Build a fresh MultiTaskModel. Returns (model, kwargs_used)."""
    if variant == "raw":
        kwargs = dict(cfg.models_nn.raw[arch])
    else:
        kwargs = dict(cfg.models_nn.features[arch])
    if hp_overrides:
        kwargs.update(hp_overrides)
    if variant == "raw":
        encoder = build_raw_encoder(arch, n_channels=dataset.n_channels, kwargs=kwargs)
    else:
        encoder = build_feature_encoder(arch, n_features=dataset.n_features, kwargs=kwargs)
    model = MultiTaskModel(
        encoder=encoder,
        n_exercise=dataset.n_exercise,
        n_phase=dataset.n_phase,
        dropout=float(kwargs.get("dropout", 0.3)),
    )
    return model, kwargs


# ---- one-fold training ----------------------------------------------------


def _train_one(arch, variant, dataset, fold, seed, cfg, args, out_dir: Path,
               *, hp_overrides: dict | None = None, epochs: int | None = None,
               save_ckpt: bool = True, lr_override: float | None = None,
               ) -> dict:
    L.seed_everything(seed, workers=True)
    model, _ = _build_model(arch, variant, dataset, cfg, hp_overrides=hp_overrides)
    epochs = int(epochs or args.epochs or cfg.training.epochs)
    bs = int(args.batch_size or cfg.training.batch_size)
    lr = float(lr_override if lr_override is not None else cfg.training.lr)

    lit = LitMultiTask(
        model=model,
        n_exercise=dataset.n_exercise,
        n_phase=dataset.n_phase,
        lr=lr,
        weight_decay=float(cfg.training.weight_decay),
        epochs=epochs,
        loss_kwargs=dict(
            w_exercise=float(cfg.losses.weights.exercise),
            w_phase=float(cfg.losses.weights.phase),
            w_fatigue=float(cfg.losses.weights.fatigue),
            w_reps=float(cfg.losses.weights.reps),
            use_uncertainty_weighting=bool(cfg.training.uncertainty_weighting),
            target_modes=dict(cfg.soft_targets.default_modes),
            enabled_tasks=list(args.tasks),
        ),
    )
    dm = FoldDataModule(
        dataset=dataset, fold=fold, batch_size=bs,
        num_workers=int(cfg.training.num_workers),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "gpu" if torch_device() == "cuda" else "cpu"
    cbs: list = [
        EarlyStopping(monitor="val/loss", patience=int(cfg.training.patience), mode="min"),
    ]
    if save_ckpt:
        cbs.insert(0, ModelCheckpoint(
            dirpath=out_dir, monitor="val/loss", mode="min",
            save_top_k=1, filename="best",
        ))
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else 32,
        gradient_clip_val=float(cfg.training.grad_clip),
        callbacks=cbs,
        logger=TensorBoardLogger(save_dir=str(out_dir), name="tb"),
        enable_progress_bar=False,
        log_every_n_steps=20,
        deterministic=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, datamodule=dm)
    metrics = {k: float(v.item()) if hasattr(v, "item") else float(v)
               for k, v in trainer.callback_metrics.items()}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---- Optuna phase 1 -------------------------------------------------------


_RAW_ARCH_KEYS = set(RAW_ARCHS) | set(FEATURE_ARCHS)


def _suggest_one(trial, name: str, spec):
    """Map one search-space entry → an optuna suggestion call.

    Spec shape:
      list                                  -> categorical
      dict {low, high}                      -> suggest_float
      dict {low, high, log: true}           -> suggest_float (log scale)
      dict {low, high, int: true}           -> suggest_int
      dict {low, high, step: ..., int}      -> suggest_int with step
    """
    # Resolve OmegaConf wrappers to plain Python so isinstance/dict-keys work.
    from omegaconf import OmegaConf as _OC
    if hasattr(spec, "_content") or hasattr(spec, "_get_node"):
        spec = _OC.to_container(spec, resolve=True)

    if isinstance(spec, list):
        return trial.suggest_categorical(name, list(spec))
    if isinstance(spec, dict):
        low = spec["low"]
        high = spec["high"]
        is_int = bool(spec.get("int", False))
        log = bool(spec.get("log", False))
        step = spec.get("step")
        if is_int:
            return trial.suggest_int(name, int(low), int(high),
                                       step=int(step) if step else 1)
        return trial.suggest_float(name, float(low), float(high), log=log)
    raise ValueError(f"Bad search_space entry for {name!r}: {spec!r}")


def _suggest_hp(trial, arch: str, cfg) -> dict:
    """Build per-trial HPs from cfg.optuna.search_space.

    Top-level keys (lr, dropout, repr_dim, ...) apply to all archs.
    Arch-specific keys live under cfg.optuna.search_space.<arch>.<param>.
    """
    space = cfg.optuna.search_space
    hp: dict = {}
    for key, spec in space.items():
        if key in _RAW_ARCH_KEYS:
            continue  # arch-specific, handled below
        hp[key] = _suggest_one(trial, key, spec)

    arch_space = space.get(arch) if hasattr(space, "get") else None
    if arch_space:
        for key, spec in arch_space.items():
            hp[key] = _suggest_one(trial, key, spec)
    return hp


def _phase1(arch, variant, dataset, folds, n_trials, cfg, args, out_dir: Path) -> dict:
    """Run Optuna n_trials on the first fold. Returns best HPs."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold = folds[0]

    def objective(trial) -> float:
        hp = _suggest_hp(trial, arch, cfg)
        encoder_overrides = {k: v for k, v in hp.items() if k != "lr"}
        trial_dir = out_dir / "phase1" / f"trial_{trial.number:03d}"
        try:
            metrics = _train_one(
                arch, variant, dataset, fold, seed=42,
                cfg=cfg, args=args, out_dir=trial_dir,
                hp_overrides=encoder_overrides,
                epochs=int(args.phase1_epochs),
                save_ckpt=False,
                lr_override=float(hp["lr"]),
            )
        except Exception as e:
            print(f"[phase1 trial {trial.number}] FAILED: {e}")
            return float("inf")
        # Single-task → use that task's loss; multi-task → total val loss.
        if len(args.tasks) == 1:
            key = f"val/loss_{args.tasks[0]}"
            v = metrics.get(key, metrics.get("val/loss", float("inf")))
        else:
            v = metrics.get("val/loss", float("inf"))
        print(f"[phase1 trial {trial.number}] val_obj={v:.4f}  hp={hp}")
        return float(v)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[phase1] best params: {study.best_params}  best_value={study.best_value:.4f}")
    return study.best_params


# ---- main -----------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
    # Tensor Cores: trade tiny precision for ~2× matmul throughput on RTX 30+/40/50.
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    args, extras = _parse_args()
    cfg = resolve_paths(load_config(args.config, overrides=extras))
    args = _resolve_cli_defaults(args, cfg)

    if args.arch == "all":
        archs = RAW_ARCHS if args.variant == "raw" else FEATURE_ARCHS
    else:
        archs = (args.arch,)
    if args.variant == "raw" and "mlp" in archs:
        raise SystemExit("--arch mlp requires --variant features")

    dataset = _build_dataset(
        cfg, args.variant, args.features_parquet,
        labeled_root=args.labeled_root,
        exclude_recordings=args.exclude_recordings,
        window_s=args.window_s,
    )
    # Pin features dataset to GPU once — eliminates per-batch CPU→GPU copies
    # for the small MLP path. The raw dataset uses lazy pandas access so we
    # leave it on host memory.
    if args.variant == "features" and torch_device() == "cuda" and hasattr(dataset, "to"):
        dataset.to("cuda")
        print(f"[src3.train] dataset pinned to cuda — "
              f"{dataset._x.element_size() * dataset._x.numel() / 1e6:.1f} MB")
    seeds = args.seeds or list(cfg.training.seeds)
    folds = list(cv_iter(
        dataset.subject_ids,
        splits_csv=Path(cfg.paths.splits_csv),
        scheme=str(cfg.cv.scheme),
        n_splits=int(cfg.cv.n_splits),
    ))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "-".join(archs)
    out_parent = (PROJECT_ROOT / cfg.run.out_dir
                  if hasattr(cfg, "run") and cfg.run.get("out_dir")
                  else PROJECT_ROOT / "runs")
    out_dir = args.out or out_parent / f"{ts}_src3_{args.variant}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[src3.train] dataset_len={len(dataset)} folds={len(folds)} "
          f"seeds={seeds} archs={list(archs)} variant={args.variant}")
    print(f"[src3.train] out={out_dir}")
    print(f"[src3.train] tasks={args.tasks}  n_trials={args.n_trials}")

    summary: dict = {"args": vars(args), "results": {}}
    for arch in archs:
        # Phase 1
        if args.n_trials > 0:
            print(f"\n=== Phase 1 ({arch}): Optuna {args.n_trials} trials on fold 0 ===")
            best_hp = _phase1(arch, args.variant, dataset, folds, args.n_trials,
                              cfg, args, out_dir / arch)
        else:
            best_hp = {}
        (out_dir / f"{arch}_best_hp.json").write_text(
            json.dumps({"arch": arch, "best_hp": best_hp}, indent=2),
        )

        # Phase 2: full CV × seeds
        encoder_overrides = {k: v for k, v in best_hp.items() if k != "lr"}
        lr_override = best_hp.get("lr")
        print(f"\n=== Phase 2 ({arch}): {len(folds)} folds × {len(seeds)} seeds ===")
        for fold in folds:
            for seed in seeds:
                tag = f"{arch}/fold{fold.fold}_seed{seed}"
                fold_dir = out_dir / arch / "phase2" / f"fold{fold.fold}_seed{seed}"
                print(f"[phase2] {tag} — train={len(fold.train_idx)} "
                      f"val={len(fold.val_idx)} val_subjects={fold.val_subjects}")
                try:
                    summary["results"][tag] = _train_one(
                        arch, args.variant, dataset, fold, seed,
                        cfg=cfg, args=args, out_dir=fold_dir,
                        hp_overrides=encoder_overrides,
                        lr_override=lr_override,
                    )
                except Exception as e:
                    print(f"[phase2] {tag} FAILED: {e}")
                    summary["results"][tag] = {"error": str(e)}
                # Persist after every fold so a crash leaves partial results.
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[src3.train] done — {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
