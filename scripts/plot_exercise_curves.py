"""Plot exercise-task training curves for raw 1D-CNN vs feature MLP.

Reads phase 2 history.json files from two run dirs and overlays:
  - Per-epoch exercise train loss (CE/BCE) per fold
  - Per-epoch exercise val loss per fold
  - Per-epoch exercise val F1 per fold
Mean across folds drawn bold; individual folds light.

Outputs results/v18_exercise_curves_raw_vs_feature.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    'MLP feature (Phinyomark, multitask)':
        ROOT / 'runs' / 'v18multi-mlp-phinyomark-gkf5' / 'phase2' / 'mlp',
    '1D-CNN raw (multitask, narrow 5-trial)':
        ROOT / 'runs' / 'v18multi-cnn1d_raw-gkf5-w5s' / 'phase2' / 'cnn1d_raw',
}

COLORS = {
    'MLP feature (Phinyomark, multitask)': '#1f77b4',  # blue
    '1D-CNN raw (multitask, narrow 5-trial)': '#d62728',  # red
}


def load_fold_histories(run_arch_dir: Path):
    out = []
    for fold_dir in sorted(run_arch_dir.glob('seed_*/fold_*')):
        h_path = fold_dir / 'history.json'
        if h_path.exists():
            out.append((fold_dir.parent.name + '/' + fold_dir.name,
                        json.loads(h_path.read_text())))
    return out


def extract(history, key_path):
    """Pull a per-epoch series from history. key_path is dotted, e.g.
    'train.exercise' or 'val_metrics.exercise.f1_macro'.
    """
    parts = key_path.split('.')
    out = []
    for h in history:
        v = h
        for p in parts:
            v = v[p]
        out.append(float(v))
    return np.asarray(out)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for label, path in RUNS.items():
        color = COLORS[label]
        folds = load_fold_histories(path)
        if not folds:
            print(f"[skip] {label}: no folds at {path}")
            continue
        max_len = max(len(h) for _, h in folds)

        train_loss_curves = []
        val_loss_curves = []
        f1_curves = []

        for fold_name, history in folds:
            train_l = extract(history, 'train.exercise')
            val_l = extract(history, 'val_loss.exercise')
            f1 = extract(history, 'val_metrics.exercise.f1_macro')
            ep = np.arange(len(history))

            axes[0].plot(ep, train_l, color=color, alpha=0.18, lw=0.8)
            axes[1].plot(ep, val_l, color=color, alpha=0.18, lw=0.8)
            axes[2].plot(ep, f1, color=color, alpha=0.18, lw=0.8)

            train_loss_curves.append(np.pad(train_l, (0, max_len - len(train_l)),
                                             constant_values=np.nan))
            val_loss_curves.append(np.pad(val_l, (0, max_len - len(val_l)),
                                           constant_values=np.nan))
            f1_curves.append(np.pad(f1, (0, max_len - len(f1)),
                                     constant_values=np.nan))

        # Mean across folds (ignoring nan from early-stopped folds)
        train_mean = np.nanmean(np.stack(train_loss_curves), axis=0)
        val_mean = np.nanmean(np.stack(val_loss_curves), axis=0)
        f1_mean = np.nanmean(np.stack(f1_curves), axis=0)
        ep = np.arange(max_len)
        axes[0].plot(ep, train_mean, color=color, lw=2.2, label=label)
        axes[1].plot(ep, val_mean, color=color, lw=2.2, label=label)
        axes[2].plot(ep, f1_mean, color=color, lw=2.2, label=label)

    axes[0].set_title('Exercise train loss (per fold + mean)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train loss (exercise CE)')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Exercise val loss (per fold + mean)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val loss (exercise CE)')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].set_title('Exercise val F1-macro (per fold + mean)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Val F1-macro')
    axes[2].set_ylim(0, 1)
    axes[2].axhline(0.25, color='gray', ls=':', lw=0.8,
                     label='4-class chance')
    axes[2].legend(loc='lower right', fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        'Exercise-task training curves — raw 1D-CNN vs feature MLP '
        '(GKF-5, w=5s, multitask phase 2)',
        fontsize=12,
    )
    fig.tight_layout()

    out_path = ROOT / 'results' / 'v18_exercise_curves_raw_vs_feature.png'
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'[plot] wrote {out_path}')


if __name__ == '__main__':
    main()
