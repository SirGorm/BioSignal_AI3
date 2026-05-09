"""V17 overfit plots for the best feature models.

Single-task: only the relevant task head's train/val loss is shown.
Multi-task : all 4 task heads + total loss are shown.

Curves are aggregated across the 30 (seed * fold) histories per slug
with seaborn's lineplot (mean line + SD band).

Usage:
    python scripts/plot_v17_overfit.py
    python scripts/plot_v17_overfit.py --history results/Final/v17_all_history.csv \
                                       --out results/v17_overfit_plots
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SINGLE_TASK_SLUGS = {
    "exercise": "v17single-exercise-mlp-w1s",
    "phase":    "v17single-phase-mlp-w1s",
    "fatigue":  "v17single-fatigue-lstm-w2s",
    "reps":     "v17single-reps-lstm-w2s",
}
MULTITASK_SLUG = "v17multi-mlp-w1s"

TASK_ORDER = ["exercise", "phase", "fatigue", "reps"]

ID_COLS = ["seed", "fold", "epoch"]


def _to_long(df: pd.DataFrame, items: list[tuple[str, str]]) -> pd.DataFrame:
    """items: list of (column_in_df, label_in_plot)."""
    cols = [c for c, _ in items]
    long = df[ID_COLS + cols].melt(
        id_vars=ID_COLS, value_vars=cols, var_name="_col", value_name="value"
    )
    label_map = {c: lbl for c, lbl in items}
    long["series"] = long["_col"].map(label_map)
    return long.drop(columns="_col")


def _lineplot(ax, df: pd.DataFrame, items: list[tuple[str, str, str, str]],
              *, label_fs: int, legend_fs: int, title: str,
              title_fs: int, ylabel: str = "loss") -> None:
    """items: list of (column, label, color, linestyle).

    Uses sns.lineplot with errorbar='sd' so the band reflects the spread
    across (seed, fold) replicates per epoch.
    """
    pairs = [(c, lbl) for c, lbl, *_ in items]
    long = _to_long(df, pairs)

    palette = {lbl: color for _, lbl, color, _ in items}
    dashes = {
        lbl: ("" if ls == "-" else (4, 2))
        for _, lbl, _, ls in items
    }
    order = [lbl for _, lbl, _, _ in items]

    sns.lineplot(
        data=long,
        x="epoch", y="value",
        hue="series", style="series",
        hue_order=order, style_order=order,
        palette=palette, dashes=dashes,
        errorbar="sd",
        linewidth=1.8,
        ax=ax,
    )

    ax.set_xlabel("epoch", fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)
    ax.tick_params(labelsize=label_fs - 1)
    leg = ax.legend(loc="best", frameon=False, fontsize=legend_fs)
    if leg is not None and leg.get_title() is not None:
        leg.set_title(None)


def _arch_from_slug(slug: str) -> str:
    for a in ("cnn_lstm", "cnn1d", "lstm", "mlp", "tcn"):
        if f"-{a}-" in slug:
            return a
    return "?"


def _window_from_slug(slug: str) -> str:
    m = re.search(r"-w(\d+)s", slug)
    return f"{m.group(1)}s" if m else "?"


def plot_single_task(history: pd.DataFrame, slug: str, task: str,
                     out_dir: Path) -> Path:
    df = history[history["slug"] == slug]
    if df.empty:
        raise SystemExit(f"No history for slug={slug}")

    palette = sns.color_palette("deep")
    items = [
        (f"train_{task}", "train", palette[0], "-"),
        (f"val_{task}",   "val",   palette[3], "--"),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    _lineplot(ax, df, items, label_fs=11, legend_fs=10,
              title=f"{task} loss", title_fs=12)
    fig.suptitle(
        f"{_arch_from_slug(slug)} - features - single-task ({task}) - "
        f"window={_window_from_slug(slug)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / f"{slug}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_multitask(history: pd.DataFrame, slug: str, out_dir: Path) -> Path:
    df = history[history["slug"] == slug]
    if df.empty:
        raise SystemExit(f"No history for slug={slug}")

    palette = sns.color_palette("deep")
    train_color, val_color = palette[0], palette[3]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.4))
    panel_kw = dict(label_fs=15, legend_fs=14, title_fs=20)

    cols = [("total", "train_total", "val_total")] + [
        (t, f"train_{t}", f"val_{t}") for t in TASK_ORDER
    ]
    for ax, (title, train_col, val_col) in zip(axes, cols):
        items = [
            (train_col, "train", train_color, "-"),
            (val_col,   "val",   val_color,   "--"),
        ]
        _lineplot(ax, df, items, title=title, **panel_kw)

    fig.suptitle(
        f"{_arch_from_slug(slug)} - features - multi-task - "
        f"window={_window_from_slug(slug)}",
        fontsize=20,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out_dir / f"{slug}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_vs_multi(history: pd.DataFrame,
                          single_slug: str,
                          multi_slug: str,
                          out_dir: Path,
                          filename: str = "v17_single_vs_multi.png") -> Path:
    """4 panels (exercise, phase, fatigue, reps). Each panel overlays the
    SAME single-task model and the SAME multi-task model - train + val for
    both - on that head."""
    df_single = history[history["slug"] == single_slug]
    if df_single.empty:
        raise SystemExit(f"No history for slug={single_slug}")
    df_multi = history[history["slug"] == multi_slug]
    if df_multi.empty:
        raise SystemExit(f"No history for slug={multi_slug}")

    df_single = df_single.assign(model="single")
    df_multi  = df_multi.assign(model="multi")

    palette = sns.color_palette("deep")
    series_specs = [
        ("single", "train", palette[0], "-",  "single train"),
        ("single", "val",   palette[3], "--", "single val"),
        ("multi",  "train", palette[2], "-",  "multi train"),
        ("multi",  "val",   palette[1], "--", "multi val"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))
    title_fs, label_fs, legend_fs = 22, 16, 13

    for ax, task in zip(axes, TASK_ORDER):
        rows = []
        for model_key, split, _color, _ls, label in series_specs:
            src = df_single if model_key == "single" else df_multi
            col = f"{split}_{task}"
            sub = src[ID_COLS + [col]].rename(columns={col: "value"})
            sub["series"] = label
            rows.append(sub)
        long = pd.concat(rows, ignore_index=True)

        order = [s[-1] for s in series_specs]
        palette_map = {s[-1]: s[2] for s in series_specs}
        dashes_map  = {s[-1]: ("" if s[3] == "-" else (4, 2)) for s in series_specs}

        sns.lineplot(
            data=long,
            x="epoch", y="value",
            hue="series", style="series",
            hue_order=order, style_order=order,
            palette=palette_map, dashes=dashes_map,
            errorbar="sd",
            linewidth=1.8,
            ax=ax,
        )

        ax.set_title(task, fontsize=title_fs)
        ax.set_xlabel("epoch", fontsize=label_fs)
        ax.set_ylabel("loss", fontsize=label_fs)
        ax.tick_params(labelsize=label_fs - 2)
        leg = ax.legend(loc="best", frameon=False, fontsize=legend_fs)
        if leg is not None:
            leg.set_title(None)

    fig.suptitle(
        f"features - single-task vs multi-task - "
        f"single: {_arch_from_slug(single_slug)} "
        f"w={_window_from_slug(single_slug)}  |  "
        f"multi: {_arch_from_slug(multi_slug)} "
        f"w={_window_from_slug(multi_slug)}",
        fontsize=20,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out_dir / filename
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_task_panel(history: pd.DataFrame,
                            slugs_by_task: dict[str, str],
                            out_dir: Path,
                            filename: str = "v17single_all_heads.png") -> Path:
    """One figure with 4 panels - each shows the train/val loss of the head
    that the corresponding single-task model was trained on."""
    palette = sns.color_palette("deep")
    train_color, val_color = palette[0], palette[3]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    panel_kw = dict(label_fs=15, legend_fs=14, title_fs=20)

    for ax, task in zip(axes, TASK_ORDER):
        slug = slugs_by_task[task]
        df = history[history["slug"] == slug]
        if df.empty:
            raise SystemExit(f"No history for slug={slug}")
        items = [
            (f"train_{task}", "train", train_color, "-"),
            (f"val_{task}",   "val",   val_color,   "--"),
        ]
        title = f"{task} ({_arch_from_slug(slug)}, w={_window_from_slug(slug)})"
        _lineplot(ax, df, items, title=title, **panel_kw)

    fig.suptitle(
        "features - single-task (per-task best models, only own head)",
        fontsize=20,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out_dir / filename
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path,
                   default=Path("results/Final/v17_all_history.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("results/v17_overfit_plots"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid")

    history = pd.read_csv(args.history)

    written: list[Path] = []
    for task, slug in SINGLE_TASK_SLUGS.items():
        written.append(plot_single_task(history, slug, task, args.out))
    written.append(plot_single_task_panel(history, SINGLE_TASK_SLUGS, args.out))
    written.append(plot_multitask(history, MULTITASK_SLUG, args.out))
    written.append(plot_single_vs_multi(
        history,
        single_slug="v17single-fatigue-lstm-w2s",
        multi_slug=MULTITASK_SLUG,
        out_dir=args.out,
    ))

    for w in written:
        print(f"Wrote {w}")


if __name__ == "__main__":
    main()
