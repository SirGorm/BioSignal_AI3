"""V14 — critical-modality summary per task per arch.

Reduces the LOO ablation to a single question:
  "For each (task, architecture) pair, which modality is *critical* —
   i.e. the one whose removal hurts performance most?"

Outputs to runs/comparison_v14/:
  critical_modality_per_task.png — grouped bars, 4 task panels × (arch × mod),
                                   most-critical mod per arch is starred.
  critical_modality_table.png    — compact table figure with the winners.
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

ARCHS = [
    ('multi-feat-mlp',  'feat-MLP',  'optuna_clean_v12eqw-w1s-multi-feat-mlp'),
    ('multi-feat-lstm', 'feat-LSTM', 'optuna_clean_v12eqw-w1s-multi-feat-lstm'),
    ('multi-raw-tcn',   'raw-TCN',   'optuna_clean_v12eqw-w1s-multi-raw-tcn'),
]
MODS = ['emg', 'acc', 'ppg', 'temp']
TASKS = [
    ('exercise', 'f1_macro',  'F1',  True,  '#1f77b4'),
    ('phase',    'f1_macro',  'F1',  True,  '#2ca02c'),
    ('fatigue',  'pearson_r', 'r',   True,  '#d62728'),
    ('reps',     'mae',       'MAE', False, '#ff7f0e'),
]
ARCH_COLORS = {'feat-MLP': '#3498db', 'feat-LSTM': '#9b59b6', 'raw-TCN': '#e67e22'}


def get_metric(run_dir: Path, task: str, metric: str):
    p = next(iter((run_dir / 'phase2').rglob('cv_summary.json')), None)
    if p is None:
        return None
    s = json.loads(p.read_text()).get('summary', {})
    return s.get(task, {}).get(metric, {}).get('mean')


def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # results[arch_slug] = {'baseline': {(task,metric): v},
    #                      (mod,'loo'|'loi'): {(task,metric): v}}
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

    # ------------------------------------------------------------------
    # Figure 1: per-task grouped bars of LOO Δ, with the critical modality
    # per arch highlighted with a star.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    critical_table = {}  # (task, arch_label) -> (mod, delta)
    for ti, (task, metric, ylabel, higher_better, _) in enumerate(TASKS):
        ax = axes[ti]
        x = np.arange(len(MODS))
        w = 0.26
        # one group of 4 bars per arch
        for ai, (arch_slug, arch_label, _) in enumerate(ARCHS):
            base = results[arch_slug]['baseline'][(task, metric)]
            deltas = []
            for mod in MODS:
                v = results[arch_slug][(mod, 'loo')][(task, metric)]
                if v is None or base is None:
                    deltas.append(np.nan); continue
                # Sign convention: NEGATIVE Δ = drop hurt (mod is critical)
                d = (v - base) if higher_better else (base - v)
                deltas.append(d)
            deltas = np.array(deltas, dtype=float)
            offsets = (ai - 1) * w
            bars = ax.bar(x + offsets, deltas, w,
                          color=ARCH_COLORS[arch_label],
                          edgecolor='black', label=arch_label)
            # Identify the critical modality (most-negative Δ) for this arch.
            if np.isfinite(deltas).any():
                crit_idx = int(np.nanargmin(deltas))
                crit_mod = MODS[crit_idx]
                crit_delta = deltas[crit_idx]
                critical_table[(task, arch_label)] = (crit_mod, crit_delta)
                # Highlight the critical bar with a star annotation.
                bars[crit_idx].set_edgecolor('red')
                bars[crit_idx].set_linewidth(2.2)
                ymin = ax.get_ylim()[0]
                star_y = crit_delta - 0.005 if crit_delta < 0 else crit_delta + 0.005
                ax.scatter(crit_idx + offsets, crit_delta * 1.02 if crit_delta < 0 else crit_delta + 0.005,
                           marker='*', s=110, color='red', zorder=5,
                           edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(MODS)
        ax.set_xlabel('Dropped modality')
        ax.set_ylabel(f'LOO Δ ({ylabel})  — neg = drop hurt')
        direction = 'higher = better' if higher_better else 'lower = better'
        ax.set_title(f'{task} ({ylabel}, {direction})', fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.legend(fontsize=9, loc='best')

    fig.suptitle('Critical modality per (task × architecture) — '
                 'red star = the modality whose removal hurt performance most',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = OUT / 'critical_modality_per_task.png'
    fig.savefig(out1, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out1}')

    # ------------------------------------------------------------------
    # Figure 2: compact table figure — critical mod + drop per (task, arch).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.axis('off')
    arch_labels = [lab for _, lab, _ in ARCHS]
    task_labels = [t[0] for t in TASKS]
    # Build the cell text and a colour map.
    # Severity = |Δ| / |baseline| (relative drop, where defined).
    cell_text = []
    cell_colors = []
    for task, metric, ylabel, higher_better, _ in TASKS:
        row, colors = [], []
        for arch_slug, arch_label, _ in ARCHS:
            base = results[arch_slug]['baseline'][(task, metric)]
            mod, dlt = critical_table.get((task, arch_label), (None, None))
            if mod is None or dlt is None:
                row.append('—'); colors.append('#eeeeee'); continue
            if base is None or abs(base) < 1e-6:
                rel = 0.0
            else:
                rel = abs(dlt) / abs(base)
            # Format: "mod  Δ=-0.123  (-12%)"
            rel_pct = rel * 100.0
            if dlt < 0:
                txt = f'{mod}\nΔ={dlt:+.3f}  ({-rel_pct:.0f}%)'
            else:
                txt = f'{mod}\nΔ={dlt:+.3f}  (no drop)'
            row.append(txt)
            sev = max(0.0, min(1.0, rel / 0.5))  # cap at 50% drop = max red
            r = 0.95
            g = 0.95 * (1 - sev * 0.85)
            b = 0.95 * (1 - sev * 0.85)
            colors.append((r, g, b))
        cell_text.append(row)
        cell_colors.append(colors)

    tbl = ax.table(cellText=cell_text,
                   rowLabels=[f'{t[0]}\n({t[2]})' for t in TASKS],
                   colLabels=arch_labels,
                   cellColours=cell_colors,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.0)
    ax.set_title('Critical modality per (task × architecture)\n'
                 'Cell = modality whose removal hurt most | Δ = LOO change | '
                 '% = relative drop vs baseline | redder = bigger drop',
                 fontsize=11, pad=12)
    out2 = OUT / 'critical_modality_table.png'
    fig.savefig(out2, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out2}')

    # Also dump as JSON for the report
    out_json = OUT / 'critical_modality.json'
    payload = {
        'baselines': {a: {f'{t}/{m}': results[a]['baseline'][(t, m)]
                          for t, m, *_ in TASKS}
                      for a, _, _ in ARCHS},
        'critical': {f'{t}/{lab}': {'modality': m, 'delta': float(d)}
                     for (t, lab), (m, d) in critical_table.items()},
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f'Wrote {out_json}')


if __name__ == '__main__':
    main()
