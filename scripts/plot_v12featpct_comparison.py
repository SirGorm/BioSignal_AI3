"""V12featpct comparison: heatmaps + CMs + learning curves for the
6 multi-task feature runs (multi-feat-mlp, multi-feat-lstm × 1/2/5 s)
trained with percentile normalization."""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix as sk_cm, accuracy_score,
                              balanced_accuracy_score, f1_score)

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v12featpct"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cm").mkdir(exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

WINDOWS = ["1s", "2s", "5s"]
WIN_LABELS = ["1.0", "2.0", "5.0"]
ARCHS = ["multi-feat-mlp", "multi-feat-lstm"]
ARCH_LABELS = ["feat-MLP", "feat-LSTM"]
COLORS = {"multi-feat-mlp": "#1f77b4", "multi-feat-lstm": "#ff7f0e"}
LINESTYLES = {"1s": "-", "2s": "--", "5s": ":"}
TASKS = ["exercise", "phase", "fatigue", "reps"]


def load_results():
    out = {s: {} for s in ARCHS}
    for slug in ARCHS:
        for w in WINDOWS:
            cv = next(iter((ROOT / f"runs/optuna_clean_v12featpct-w{w}-{slug}/phase2")
                            .rglob("cv_summary.json")), None)
            if cv:
                out[slug][w] = json.loads(cv.read_text())["summary"]
    return out


def get(d, *ks, default=np.nan):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d if d is not None else default


def heatmap(ax, mat, std_mat, row_labels, col_labels, title,
            fmt="{:.3f}", cmap="viridis", best="max"):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Window (s)")
    if not np.all(np.isnan(mat)):
        bi = np.unravel_index(np.nanargmax(mat) if best == "max"
                               else np.nanargmin(mat), mat.shape)
        ax.add_patch(plt.Rectangle((bi[1] - 0.5, bi[0] - 0.5), 1, 1,
                                    fill=False, edgecolor="gold", linewidth=3))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "-", ha="center", va="center", color="grey")
                continue
            txt = fmt.format(v)
            if std_mat is not None and not np.isnan(std_mat[i, j]):
                txt = f"{txt}\n+/- {std_mat[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center",
                     color="black", fontsize=8)
    plt.colorbar(im, ax=ax)


def render_heatmaps(results):
    n_a, n_w = len(ARCHS), len(WINDOWS)
    mats = {k: (np.full((n_a, n_w), np.nan), np.full((n_a, n_w), np.nan))
            for k in ["ex_F1", "ph_F1", "fa_MAE", "fa_r", "rp_MAE"]}
    for i, slug in enumerate(ARCHS):
        for j, w in enumerate(WINDOWS):
            s = results[slug].get(w)
            if not s: continue
            mats["ex_F1"][0][i, j]  = get(s, "exercise", "f1_macro", "mean")
            mats["ex_F1"][1][i, j]  = get(s, "exercise", "f1_macro", "std")
            mats["ph_F1"][0][i, j]  = get(s, "phase", "f1_macro", "mean")
            mats["ph_F1"][1][i, j]  = get(s, "phase", "f1_macro", "std")
            mats["fa_MAE"][0][i, j] = get(s, "fatigue", "mae", "mean")
            mats["fa_MAE"][1][i, j] = get(s, "fatigue", "mae", "std")
            mats["fa_r"][0][i, j]   = get(s, "fatigue", "pearson_r", "mean")
            mats["fa_r"][1][i, j]   = get(s, "fatigue", "pearson_r", "std")
            mats["rp_MAE"][0][i, j] = get(s, "reps", "mae", "mean")
            mats["rp_MAE"][1][i, j] = get(s, "reps", "mae", "std")

    panels = [
        ("exercise_f1", mats["ex_F1"],  "Exercise F1-macro",   "{:.3f}",  "Blues",    "max"),
        ("phase_f1",    mats["ph_F1"],  "Phase F1-macro",      "{:.3f}",  "Greens",   "max"),
        ("fatigue_mae", mats["fa_MAE"], "Fatigue MAE",          "{:.3f}",  "YlOrRd_r", "min"),
        ("fatigue_r",   mats["fa_r"],   "Fatigue Pearson r",    "{:+.3f}", "RdYlGn",   "max"),
        ("reps_mae",    mats["rp_MAE"], "Reps MAE",             "{:.3f}",  "YlOrRd_r", "min"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (_, (m, s), title, fmt, cmap, b) in zip(axes.flat, panels[:4]):
        heatmap(ax, m, s, ARCH_LABELS, WIN_LABELS, title, fmt, cmap, b)
    fig.suptitle("V12 percentile-norm - window x arch (Phase 2, 21 fold-runs)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "heatmap_4tasks.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'heatmap_4tasks.png'}")
    for name, (m, s), title, fmt, cmap, b in panels:
        fig, ax = plt.subplots(figsize=(7, 5))
        heatmap(ax, m, s, ARCH_LABELS, WIN_LABELS, title, fmt, cmap, b)
        fig.tight_layout()
        fig.savefig(OUT / f"heatmap_{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUT / f'heatmap_{name}.png'}")


def load_preds(run_dir, task):
    yp_all, yt_all = [], []
    for fd in (run_dir / "phase2").glob("*/seed_*/fold_*"):
        try:
            d = torch.load(fd / "test_preds.pt", weights_only=False, map_location="cpu")
        except Exception:
            continue
        m = d["masks"].get(task)
        if m is None: continue
        m = m.numpy().astype(bool)
        if not m.any(): continue
        yp = d["preds"][task].numpy()
        yt = d["targets"][task].numpy()
        if yp.ndim == 2: yp = yp.argmax(axis=1)
        if yt.ndim > 1 and task == "phase": yt = yt.argmax(axis=1)
        yp_all.append(yp[m]); yt_all.append(yt[m])
    if not yp_all:
        return None, None
    return np.concatenate(yp_all), np.concatenate(yt_all)


def plot_cm(run_dir, task, classes, label, out_path, cmap="Blues"):
    yp, yt = load_preds(run_dir, task)
    if yp is None:
        print(f"  no preds: {run_dir.name}/{task}")
        return
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    cl = [classes[i] if i < len(classes) else f"cls_{i}" for i in present]
    cm = sk_cm(yt, yp, labels=present)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = accuracy_score(yt, yp); bal = balanced_accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, labels=present, average="macro", zero_division=0)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=max(cmn.max(), 0.01))
    for i in range(len(cl)):
        for j in range(len(cl)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})",
                     ha="center", va="center", fontsize=9, color="black")
    ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, rotation=30, ha="right")
    ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{label}\nAcc={acc:.3f}  BalAcc={bal:.3f}  F1={f1m:.3f}  N={cm.sum()}",
                  fontsize=10)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}  acc={acc:.3f} F1={f1m:.3f}")


def render_cms():
    EX_CLASSES = ["benchpress", "deadlift", "pullup", "squat"]
    PH_CLASSES = ["concentric", "eccentric", "rest"]
    for slug in ARCHS:
        for w in WINDOWS:
            rd = ROOT / f"runs/optuna_clean_v12featpct-w{w}-{slug}"
            if not rd.exists(): continue
            plot_cm(rd, "exercise", EX_CLASSES,
                    f"Exercise - {slug} @ {w} (percentile)",
                    OUT / f"cm/cm_exercise_{slug}_{w}.png", cmap="Blues")
            plot_cm(rd, "phase", PH_CLASSES,
                    f"Phase - {slug} @ {w} (percentile)",
                    OUT / f"cm/cm_phase_{slug}_{w}.png", cmap="Greens")


def render_curves():
    curves = {}
    for slug in ARCHS:
        for w in WINDOWS:
            rd = ROOT / f"runs/optuna_clean_v12featpct-w{w}-{slug}/phase2"
            if not rd.exists(): continue
            by_ep = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for hp in rd.rglob("history.json"):
                try: hist = json.loads(hp.read_text())
                except Exception: continue
                for entry in hist:
                    ep = entry["epoch"]
                    for t in TASKS:
                        tr = entry.get("train", {}).get(t)
                        if tr is not None: by_ep[t]["train"][ep].append(tr)
                        vl = entry.get("val_loss", {}).get(t)
                        if vl is not None: by_ep[t]["val"][ep].append(vl)
            out = {}
            for t in TASKS:
                out[t] = {}
                for stat in ("train", "val"):
                    d = by_ep[t][stat]
                    if not d: continue
                    eps = sorted(d.keys())
                    out[t][stat] = (np.array(eps),
                                    np.array([np.mean(d[e]) for e in eps]))
            curves[(slug, w)] = out

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for row, t in enumerate(TASKS):
        for col, stat in enumerate(["train", "val"]):
            ax = axes[row, col]
            for (slug, w), c in curves.items():
                if t not in c or stat not in c[t]: continue
                x, y = c[t][stat]
                ax.plot(x, y, color=COLORS[slug], linestyle=LINESTYLES[w],
                        linewidth=1.5, alpha=0.85)
            ax.set_title(f"{t} - {stat}", fontsize=10)
            ax.set_xlabel("Epoch"); ax.set_ylabel(f"{t} loss")
            ax.grid(linestyle=":", alpha=0.4)
    arch_h = [plt.Line2D([], [], color=COLORS[s], linewidth=2, label=l)
              for s, l in zip(ARCHS, ARCH_LABELS)]
    win_h = [plt.Line2D([], [], color="black", linewidth=2,
                         linestyle=LINESTYLES[w], label=f"{w} window")
             for w in WINDOWS]
    leg1 = axes[0, 1].legend(handles=arch_h, loc="upper left",
                              bbox_to_anchor=(1.01, 1.0), fontsize=9, title="Arch")
    axes[0, 1].add_artist(leg1)
    axes[0, 1].legend(handles=win_h, loc="upper left",
                       bbox_to_anchor=(1.01, 0.4), fontsize=9, title="Window")
    fig.suptitle("V12 percentile-norm - per-task loss per epoch "
                 "(mean across 21 fold-runs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.88, 0.97))
    fig.savefig(OUT / "curves/all_curves_per_task.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'curves/all_curves_per_task.png'}")


def main():
    results = load_results()
    print(f"Loaded {sum(len(v) for v in results.values())} runs")
    render_heatmaps(results)
    render_cms()
    render_curves()
    print(f"\nAll plots in {OUT}")


if __name__ == "__main__":
    main()
