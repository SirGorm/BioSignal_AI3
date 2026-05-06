"""V14 modality ablation plots.

For each of the 3 ablated archs (feat-MLP, feat-LSTM, raw-TCN @ 1s) trained
phase-2-only with v12 best HPs, compares:
  - baseline = full 4-modality model (from v12 w1s)
  - leave-one-out (LOO) = drop one modality, keep 3
  - leave-one-in (LOI)  = keep one modality, drop 3

Outputs to runs/comparison_v14/:
  ablation_<task>.png        — per task: 3 archs × bars (baseline + 4 LOO + 4 LOI)
  ablation_drop_heatmap.png  — heatmap of LOO performance drop per (arch, mod, task)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v14"
OUT.mkdir(parents=True, exist_ok=True)

# Architectures included in the ablation
ARCHS = [
    ('multi-feat-mlp',  'feat-MLP',  'optuna_clean_v12eqw-w1s-multi-feat-mlp'),
    ('multi-feat-lstm', 'feat-LSTM', 'optuna_clean_v12eqw-w1s-multi-feat-lstm'),
    ('multi-raw-tcn',   'raw-TCN',   'optuna_clean_v12eqw-w1s-multi-raw-tcn'),
]
MODS = ['emg', 'acc', 'ppg', 'temp']
TASKS = [
    ('exercise', 'f1_macro',  'F1', 'higher = better', '#1f77b4'),
    ('phase',    'f1_macro',  'F1', 'higher = better', '#2ca02c'),
    ('fatigue',  'pearson_r', 'r',  'higher = better', '#d62728'),
    ('reps',     'mae',       'MAE','lower = better',  '#ff7f0e'),
]


def get_metric(run_dir: Path, task: str, metric: str):
    p = next(iter((run_dir / 'phase2').rglob('cv_summary.json')), None)
    if p is None:
        return None
    s = json.loads(p.read_text()).get('summary', {})
    return s.get(task, {}).get(metric, {}).get('mean')


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # Collect all results: data[arch_slug][mod][kind][task][metric] = value
    results = {}
    for arch_slug, _, baseline_dir in ARCHS:
        results[arch_slug] = {'baseline': {}}
        bd = ROOT / 'runs' / baseline_dir
        for task, metric, *_ in TASKS:
            results[arch_slug]['baseline'][(task, metric)] = get_metric(bd, task, metric)
        for mod in MODS:
            for kind in ('loo', 'loi'):
                rd = ROOT / 'runs' / f'optuna_clean_v14ablate-{arch_slug}-{kind}-{mod}'
                results[arch_slug][(mod, kind)] = {}
                for task, metric, *_ in TASKS:
                    results[arch_slug][(mod, kind)][(task, metric)] = \
                        get_metric(rd, task, metric)

    # ---- 1) Per-task figures: 4 tasks × 3 arch panels ----------------------
    for task, metric, ylabel, direction, color in TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
        for j, (arch_slug, arch_label, _) in enumerate(ARCHS):
            ax = axes[j]
            baseline = results[arch_slug]['baseline'][(task, metric)]
            loo = [results[arch_slug][(m, 'loo')][(task, metric)] for m in MODS]
            loi = [results[arch_slug][(m, 'loi')][(task, metric)] for m in MODS]
            x = np.arange(len(MODS))
            w = 0.4
            bars_loo = ax.bar(x - w/2, loo, w, color='#e74c3c', edgecolor='black',
                               label=f'leave-one-out (drop)')
            bars_loi = ax.bar(x + w/2, loi, w, color='#27ae60', edgecolor='black',
                               label=f'leave-one-in (only)')
            for b, v in zip(bars_loo, loo):
                if v is not None:
                    ax.text(b.get_x()+b.get_width()/2, v,
                             f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            for b, v in zip(bars_loi, loi):
                if v is not None:
                    ax.text(b.get_x()+b.get_width()/2, v,
                             f'{v:.2f}', ha='center', va='bottom', fontsize=8)
            if baseline is not None:
                ax.axhline(baseline, color='black', linewidth=2, linestyle='--',
                            label=f'baseline (all 4) = {baseline:.3f}')
            ax.set_xticks(x); ax.set_xticklabels(MODS)
            ax.set_xlabel('Modality')
            if j == 0:
                ax.set_ylabel(f'{task} {ylabel}')
            ax.set_title(arch_label, fontsize=11)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            ax.legend(fontsize=8, loc='best')
        fig.suptitle(f'{task} {ylabel} — modality ablation '
                      f'({direction})', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = OUT / f'ablation_{task}.png'
        fig.savefig(out, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'Wrote {out}')

    # ---- 2) Heatmap: LOO drop per (arch, modality, task) -------------------
    # Δ = (LOO − baseline) for "higher is better" metrics
    # Δ = (baseline − LOO) for "lower is better" (reps MAE, fatigue MAE)
    # In all cases: NEGATIVE Δ = drop hurts (LOO worse than baseline)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(20, 5))
    for ti, (task, metric, ylabel, direction, _) in enumerate(TASKS):
        ax = axes[ti]
        higher_better = direction.startswith('higher')
        mat = np.full((len(ARCHS), len(MODS)), np.nan)
        for ai, (arch_slug, _, _) in enumerate(ARCHS):
            base = results[arch_slug]['baseline'][(task, metric)]
            if base is None: continue
            for mj, mod in enumerate(MODS):
                v = results[arch_slug][(mod, 'loo')][(task, metric)]
                if v is None: continue
                # Sign: negative = drop hurts
                delta = (v - base) if higher_better else (base - v)
                mat[ai, mj] = delta
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                        vmin=-max(0.1, np.nanmax(np.abs(mat))),
                        vmax=max(0.1, np.nanmax(np.abs(mat))))
        ax.set_xticks(range(len(MODS))); ax.set_xticklabels(MODS)
        ax.set_yticks(range(len(ARCHS)))
        ax.set_yticklabels([lab for _, lab, _ in ARCHS])
        for i in range(len(ARCHS)):
            for j in range(len(MODS)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                             fontsize=9, color='black')
        ax.set_title(f'{task} ({ylabel}) — LOO Δ\n(neg = drop hurts)',
                      fontsize=10)
        plt.colorbar(im, ax=ax)
    fig.suptitle('Modality ablation: LOO performance drop per (arch, modality)',
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / 'ablation_drop_heatmap.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
