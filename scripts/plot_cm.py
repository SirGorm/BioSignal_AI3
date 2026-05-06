"""Generate exercise + phase confusion matrices for any v9/v12 multi-task run.

Aggregates predictions across all seeds × folds in ``--run-dir/phase2/`` and
writes:
  <out>/cm_exercise_<tag>.png
  <out>/cm_phase_<tag>.png

Skips a task when its mask never has any True samples (e.g. fatigue-only run).
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix as sk_cm,
                              accuracy_score, balanced_accuracy_score, f1_score)

ROOT = Path(__file__).resolve().parent.parent

EX_CLASSES_DEFAULT = ['benchpress', 'deadlift', 'pullup', 'squat']
PH_CLASSES_DEFAULT = ['concentric', 'eccentric', 'rest']


def plot_cm(yt, yp, classes, title, out_path, cmap='Blues'):
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    labels = [classes[i] if i < len(classes) else f'cls_{i}' for i in present]
    cm = sk_cm(yt, yp, labels=present)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    acc = accuracy_score(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    f1_macro = f1_score(yt, yp, labels=present, average='macro', zero_division=0)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=max(cm_norm.max(), 0.01))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                     ha='center', va='center', fontsize=9,
                     color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    full_title = (f'{title}\n'
                  f'Accuracy={acc:.3f}  Balanced acc={bal_acc:.3f}  '
                  f'Macro-F1={f1_macro:.3f}  N={cm.sum()}')
    ax.set_title(full_title, fontsize=10)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}  (acc={acc:.3f}, bal_acc={bal_acc:.3f}, F1={f1_macro:.3f})')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=None,
                   help='Output dir (defaults to <run-dir>/plots/)')
    p.add_argument('--tag', default=None,
                   help='Filename tag (defaults to run-dir basename)')
    args = p.parse_args()

    rd = args.run_dir
    out = args.out or (rd / 'plots')
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or rd.name

    # Read class names from dataset_meta.json if present
    meta_path = rd / 'dataset_meta.json'
    ex_classes = EX_CLASSES_DEFAULT
    ph_classes = PH_CLASSES_DEFAULT
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ex_classes = meta.get('exercise_classes', ex_classes)
        ph_classes = meta.get('phase_classes', ph_classes)

    # Collect predictions
    p2 = rd / 'phase2'
    if not p2.exists():
        sys.exit(f'No phase2 dir: {p2}')

    all_ex_p, all_ex_t = [], []
    all_ph_p, all_ph_t = [], []
    n_files = 0
    for seed_dir in p2.glob('*/seed_*'):
        for fold_dir in sorted(seed_dir.glob('fold_*')):
            try:
                d = torch.load(fold_dir / 'test_preds.pt',
                                weights_only=False, map_location='cpu')
            except Exception:
                continue
            n_files += 1
            for task, pp, tt in [('exercise', all_ex_p, all_ex_t),
                                   ('phase',    all_ph_p, all_ph_t)]:
                pred = d['preds'].get(task)
                true = d['targets'].get(task)
                mask = d['masks'].get(task)
                if pred is None or mask is None:
                    continue
                pred = pred.numpy(); true = true.numpy()
                mask = mask.numpy().astype(bool)
                if not mask.any():
                    continue
                if pred.ndim == 2:
                    pred = pred.argmax(axis=1)
                pp.append(pred[mask]); tt.append(true[mask])

    print(f'Aggregated from {n_files} (seed,fold) prediction files in {p2}')

    if all_ex_t:
        yt = np.concatenate(all_ex_t)
        yp = np.concatenate(all_ex_p)
        plot_cm(yt, yp, ex_classes,
                 f'Exercise CM — {tag}',
                 out / f'cm_exercise_{tag}.png', cmap='Blues')
    else:
        print('  exercise: no valid samples (skipping)')

    if all_ph_t:
        yt = np.concatenate(all_ph_t)
        yp = np.concatenate(all_ph_p)
        plot_cm(yt, yp, ph_classes,
                 f'Phase CM — {tag}',
                 out / f'cm_phase_{tag}.png', cmap='Greens')
    else:
        print('  phase: no valid samples (skipping)')


if __name__ == '__main__':
    main()
