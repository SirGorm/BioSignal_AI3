"""Per-feature boxplot grid grouped by exercise.

For each engineered feature in window_features.parquet, plot the distribution
across the 4 exercises (squat / deadlift / benchpress / pullup) as side-by-side
boxplots. Lets you see at a glance which features carry exercise-discriminating
information and which are amplitude-overlapping noise.

ECG and EDA features are excluded — they are not used as model inputs (see
src/data/datasets.py:EXCLUDED_FEATURE_PREFIXES).

Usage:
    python scripts/plot_features_by_exercise.py
    python scripts/plot_features_by_exercise.py --features-path runs/<slug>/features/window_features.parquet
    python scripts/plot_features_by_exercise.py --recording 011    # one recording only
    python scripts/plot_features_by_exercise.py --rel              # use *_rel features where available
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.plot_style import apply_style, despine

apply_style()

DEFAULT_FEATURES_PATH = ROOT / "runs/20260427_110653_default/features/window_features.parquet"

# Same prefix exclusion as src/data/datasets.py — keeps plots focused on
# features that actually feed the models.
EXCLUDED_PREFIXES = ("ecg_", "eda_")

EXERCISE_ORDER = ["squat", "deadlift", "benchpress", "pullup"]
EXERCISE_COLORS = {
    "squat":      "#4C78A8",
    "deadlift":   "#F58518",
    "benchpress": "#54A24B",
    "pullup":     "#E45756",
}

MODALITY_PREFIXES = {
    "EMG":  "emg_",
    "Acc":  "acc_",
    "PPG":  "ppg_",
    "Temp": "temp_",
}


def _select_features(df: pd.DataFrame, use_rel: bool) -> list[str]:
    """Pick the model-input features, ordered by modality then alphabetically.

    use_rel=True prefers `*_rel` over the raw column when both exist (better
    cross-subject comparability).
    """
    candidate = [
        c for c in df.columns
        if not c.startswith(EXCLUDED_PREFIXES)
        and any(c.startswith(p) for p in MODALITY_PREFIXES.values())
        and df[c].dtype.kind in "fi"
    ]

    if use_rel:
        rel_cols = {c for c in candidate if c.endswith("_rel")}
        bare_cols = {c[:-4] for c in rel_cols}
        candidate = [c for c in candidate if c not in bare_cols]

    def _sort_key(col: str):
        for i, prefix in enumerate(MODALITY_PREFIXES.values()):
            if col.startswith(prefix):
                return (i, col)
        return (99, col)

    return sorted(candidate, key=_sort_key)


def _modality_of(feature: str) -> str:
    for name, prefix in MODALITY_PREFIXES.items():
        if feature.startswith(prefix):
            return name
    return "Other"


def plot_features_by_exercise(
    features_path: Path,
    recording_id: str | None,
    use_rel: bool,
    out_path: Path,
) -> Path:
    df = pd.read_parquet(features_path)

    if recording_id is not None:
        rid = f"recording_{recording_id.zfill(3)}"
        df = df[df["recording_id"] == rid]
        if df.empty:
            avail = sorted(df["recording_id"].unique())
            raise ValueError(f"no rows for {rid} (available: {avail})")

    df = df[df["in_active_set"] == True]  # noqa: E712
    df = df[df["exercise"].isin(EXERCISE_ORDER)]

    features = _select_features(df, use_rel)
    n = len(features)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols, 2.4 * n_rows + 0.6),
        gridspec_kw={"hspace": 0.55, "wspace": 0.45},
    )
    axes = axes.flatten()

    palette = [EXERCISE_COLORS[e] for e in EXERCISE_ORDER]

    for i, feat in enumerate(features):
        ax = axes[i]
        modality = _modality_of(feat)

        # Drop NaNs (some features have rests with NaN)
        sub = df[["exercise", feat]].dropna()

        sns.boxplot(
            data=sub, x="exercise", y=feat,
            hue="exercise", order=EXERCISE_ORDER,
            palette=palette, legend=False,
            ax=ax, fliersize=0, linewidth=0.9,
        )
        ax.set_title(f"{feat}  [{modality}]", fontsize=10, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    title = "Feature distributions by exercise"
    if recording_id is not None:
        title += f" — recording {recording_id}"
    title += f"  ({df.shape[0]:,} active windows)"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    despine(fig=fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH,
                        help=f"path to window_features.parquet (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--recording", type=str, default=None,
                        help="recording id (e.g. 011); omit for all recordings pooled")
    parser.add_argument("--rel", action="store_true",
                        help="prefer *_rel features over their raw versions")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path (default: inspections/feature_boxplots[_rec<id>][_rel].png)")
    args = parser.parse_args()

    if args.out is None:
        suffix = ""
        if args.recording is not None:
            suffix += f"_rec{args.recording.zfill(3)}"
        if args.rel:
            suffix += "_rel"
        args.out = ROOT / f"inspections/feature_boxplots{suffix}.png"

    plot_features_by_exercise(args.features_path, args.recording, args.rel, args.out)


if __name__ == "__main__":
    main()
