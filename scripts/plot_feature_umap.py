"""UMAP projection of per-window features, colored by exercise.

Projects the engineered feature vector to 2D and scatters each active-set
window. If the 4 exercises form distinct clusters, the feature set carries
strong exercise-discriminating information; if they overlap heavily, the
features (or modalities) are not separating the classes.

Two figures are produced per run:
  1. all_modalities — one panel using all features at once
  2. per_modality   — 2x2 grid: separate UMAP per modality (EMG / Acc / PPG / Temp)
                       lets you see which modality drives the separation

ECG and EDA features are excluded (not used as model inputs).

Usage:
    python scripts/plot_feature_umap.py
    python scripts/plot_feature_umap.py --features-path runs/<slug>/features/window_features.parquet
    python scripts/plot_feature_umap.py --subsample 30000
    python scripts/plot_feature_umap.py --rel
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.plot_style import apply_style, despine

apply_style()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_FEATURES_PATH = ROOT / "runs/20260427_110653_default/features/window_features.parquet"

EXCLUDED_PREFIXES = ("ecg_", "eda_")

EXERCISE_ORDER = ["squat", "deadlift", "benchpress", "pullup"]
EXERCISE_COLORS = {
    "squat":      "#4C78A8",
    "deadlift":   "#F58518",
    "benchpress": "#54A24B",
    "pullup":     "#E45756",
}

MODALITIES = {
    "EMG":  "emg_",
    "Acc":  "acc_",
    "PPG":  "ppg_",
    "Temp": "temp_",
}


def _select_features(df: pd.DataFrame, use_rel: bool, prefix: str | None = None) -> list[str]:
    """Pick numeric model-input features, optionally restricted to one modality."""
    cols = [
        c for c in df.columns
        if not c.startswith(EXCLUDED_PREFIXES)
        and df[c].dtype.kind in "fi"
        and any(c.startswith(p) for p in MODALITIES.values())
    ]
    if prefix is not None:
        cols = [c for c in cols if c.startswith(prefix)]
    if use_rel:
        rel_cols = {c for c in cols if c.endswith("_rel")}
        bare = {c[:-4] for c in rel_cols}
        cols = [c for c in cols if c not in bare]
    return sorted(cols)


def _embed_umap(X: np.ndarray, seed: int = 42) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="euclidean",
        random_state=seed, n_jobs=1,
    )
    return reducer.fit_transform(X)


def _scatter(ax, emb: np.ndarray, labels: pd.Series, title: str) -> None:
    for ex in EXERCISE_ORDER:
        m = (labels == ex).to_numpy()
        if not m.any():
            continue
        ax.scatter(
            emb[m, 0], emb[m, 1],
            s=4, alpha=0.35, color=EXERCISE_COLORS[ex],
            label=f"{ex} ({m.sum():,})", linewidths=0,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.tick_params(labelsize=8)
    leg = ax.legend(loc="best", fontsize=8, framealpha=0.85, markerscale=2.5)
    for h in leg.legend_handles:
        h.set_alpha(0.9)


def _prep_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    subsample: int | None,
    seed: int,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """Drop NaNs in selected features, optionally subsample stratified by exercise."""
    sub = df[feature_cols + ["exercise"]].dropna()
    if subsample is not None and len(sub) > subsample:
        sub = sub.groupby("exercise", group_keys=False).apply(
            lambda g: g.sample(
                n=min(len(g), subsample // len(EXERCISE_ORDER)),
                random_state=seed,
            )
        )
    X = StandardScaler().fit_transform(sub[feature_cols].to_numpy())
    return X, sub["exercise"], sub.index


def plot_umap(
    features_path: Path,
    recording_id: str | None,
    use_rel: bool,
    subsample: int,
    seed: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    df = pd.read_parquet(features_path)
    if recording_id is not None:
        rid = f"recording_{recording_id.zfill(3)}"
        df = df[df["recording_id"] == rid]
        if df.empty:
            raise ValueError(f"no rows for {rid}")

    df = df[df["in_active_set"] == True]  # noqa: E712
    df = df[df["exercise"].isin(EXERCISE_ORDER)]

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if recording_id is not None:
        suffix += f"_rec{recording_id.zfill(3)}"
    if use_rel:
        suffix += "_rel"

    # ---- Figure 1: all modalities together --------------------------------
    print("Computing UMAP on all features...")
    feats_all = _select_features(df, use_rel)
    X, y, _ = _prep_xy(df, feats_all, subsample, seed)
    print(f"  shape: {X.shape}")
    emb_all = _embed_umap(X, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 7))
    title = f"UMAP — all model features ({len(feats_all)} dims)"
    if recording_id is not None:
        title += f"\nrecording {recording_id}"
    _scatter(ax, emb_all, y, title=title)
    fig.tight_layout()
    despine(fig=fig)
    out_all = out_dir / f"feature_umap_all{suffix}.png"
    fig.savefig(out_all, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_all}")

    # ---- Figure 2: per modality ------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 12),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.25})
    for ax, (mod_name, prefix) in zip(axes.flatten(), MODALITIES.items()):
        feats_m = _select_features(df, use_rel, prefix=prefix)
        if len(feats_m) < 2:
            ax.text(0.5, 0.5, f"{mod_name}: <2 features",
                    ha="center", va="center", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        print(f"Computing UMAP for {mod_name} ({len(feats_m)} features)...")
        Xm, ym, _ = _prep_xy(df, feats_m, subsample, seed)
        emb_m = _embed_umap(Xm, seed=seed)
        _scatter(ax, emb_m, ym, title=f"{mod_name}  ({len(feats_m)} features)")

    title = "UMAP per modality"
    if recording_id is not None:
        title += f" — recording {recording_id}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    despine(fig=fig)
    out_per = out_dir / f"feature_umap_per_modality{suffix}.png"
    fig.savefig(out_per, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_per}")

    return out_all, out_per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH,
                        help=f"path to window_features.parquet (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--recording", type=str, default=None,
                        help="recording id (e.g. 011); omit for all recordings pooled")
    parser.add_argument("--rel", action="store_true",
                        help="prefer *_rel features over their raw versions")
    parser.add_argument("--subsample", type=int, default=20_000,
                        help="max rows total (stratified per exercise) — default 20k for speed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "inspections",
                        help="output directory (default: inspections/)")
    args = parser.parse_args()

    plot_umap(
        args.features_path,
        args.recording,
        args.rel,
        args.subsample,
        args.seed,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
