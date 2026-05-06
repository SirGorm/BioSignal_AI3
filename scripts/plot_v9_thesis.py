"""Thesis Results figures — fills gaps not covered by plot_v9_comparison.py
or plot_v9_best_model.py.

Outputs to runs/comparison_v9/thesis/:
  - headline_results.png        — 4-panel: RF vs best NN per task (mean ± std)
  - window_sweep_4tasks.png     — 2x2 panel: ex_F1, ph_F1, fat_r, reps_MAE vs window
  - fatigue_scatter_tcn3s.png   — RPE scatter for actual fatigue winner
  - cm_phase_mlp1s.png          — phase CM for actual phase winner
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_cm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "comparison_v9" / "thesis"
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = ['1s', '2s', '3s', '4s', '5s']
WIN_FLOATS = [1.0, 2.0, 3.0, 4.0, 5.0]
ARCHS = [
    'multi-feat-mlp', 'multi-feat-lstm', 'multi-raw-cnn1d', 'multi-raw-lstm',
    'multi-raw-cnn_lstm', 'multi-raw-tcn', 'fatigue-raw-tcn', 'fatigue-raw-lstm',
]
ARCH_LABELS = ['feat-MLP', 'feat-LSTM', 'raw-cnn1d', 'raw-lstm',
               'raw-cnn-lstm', 'raw-tcn', 'fatigue-tcn', 'fatigue-lstm']
ARCH_COLORS = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a',
               '#9467bd', '#c5b0d5', '#d62728', '#ff9896']
FOLD_SUBJECTS = {
    0: 'Vivian', 1: 'Hytten', 2: 'kiyomi', 3: 'lucas 2',
    4: 'Tias', 5: 'Juile', 6: 'Raghild',
}
EX_CLASSES = ['benchpress', 'deadlift', 'pullup', 'squat']
PH_CLASSES = ['concentric', 'eccentric', 'rest']


def load_summary(slug: str, win: str) -> dict | None:
    rd = ROOT / 'runs' / f'optuna_clean_v9-w{win}-{slug}'
    if not rd.exists():
        return None
    cv = next(iter((rd / 'phase2').rglob('cv_summary.json')), None)
    if cv is None:
        return None
    return json.loads(cv.read_text())


def load_all_summaries():
    out = {}
    for slug in ARCHS:
        out[slug] = {}
        for w in WINDOWS:
            s = load_summary(slug, w)
            if s is not None:
                out[slug][w] = s['summary']
    return out


def load_test_preds(run_dir: Path):
    """Aggregate test_preds.pt across all seeds × folds."""
    by_fold = defaultdict(lambda: {
        'subj': None,
        'exercise': {'pred': [], 'true': []},
        'phase': {'pred': [], 'true': []},
        'fatigue': {'pred': [], 'true': []},
        'reps': {'pred': [], 'true': []},
    })
    for seed_dir in (run_dir / 'phase2').glob('*/seed_*'):
        for fold_dir in sorted(seed_dir.glob('fold_*')):
            fk = int(fold_dir.name.split('_')[1])
            try:
                d = torch.load(fold_dir / 'test_preds.pt', weights_only=False, map_location='cpu')
            except Exception:
                continue
            by_fold[fk]['subj'] = FOLD_SUBJECTS.get(fk, f'fold_{fk}')
            for task in ('exercise', 'phase', 'fatigue', 'reps'):
                pred = d['preds'].get(task)
                true = d['targets'].get(task)
                mask = d['masks'].get(task)
                if pred is None:
                    continue
                pred = pred.numpy()
                true = true.numpy()
                mask = mask.numpy().astype(bool)
                if not mask.any():
                    continue
                if task in ('exercise', 'phase') and pred.ndim == 2:
                    pred = pred.argmax(axis=1)
                by_fold[fk][task]['pred'].append(pred[mask])
                by_fold[fk][task]['true'].append(true[mask])
    for fk in by_fold:
        for task in ('exercise', 'phase', 'fatigue', 'reps'):
            if by_fold[fk][task]['pred']:
                by_fold[fk][task]['pred'] = np.concatenate(by_fold[fk][task]['pred'])
                by_fold[fk][task]['true'] = np.concatenate(by_fold[fk][task]['true'])
            else:
                by_fold[fk][task]['pred'] = np.array([])
                by_fold[fk][task]['true'] = np.array([])
    return by_fold


def headline_figure(summaries):
    """4-panel bar chart: RF vs best NN per task."""
    rf = json.loads((ROOT / 'runs/optuna_clean_v9-rf/metrics.json').read_text())

    # Locate best NN per task across (arch, window)
    def find_best(metric_key, mode='max'):
        best = None
        for slug in ARCHS:
            for w in WINDOWS:
                m = summaries.get(slug, {}).get(w)
                if m is None:
                    continue
                # navigate metric path
                if metric_key == 'ex_f1':
                    val = m['exercise']['f1_macro']
                elif metric_key == 'ph_f1':
                    val = m['phase']['f1_macro']
                elif metric_key == 'fat_r':
                    val = m['fatigue']['pearson_r']
                elif metric_key == 'fat_mae':
                    val = m['fatigue']['mae']
                elif metric_key == 'reps_mae':
                    val = m['reps']['mae']
                else:
                    continue
                v = val['mean']
                if best is None or (mode == 'max' and v > best[0]) or (mode == 'min' and v < best[0]):
                    best = (v, val.get('std', 0.0), slug, w, val.get('n', 0))
        return best

    best_ex  = find_best('ex_f1',   'max')    # F1 higher better
    best_ph  = find_best('ph_f1',   'max')
    best_fat = find_best('fat_mae', 'min')    # MAE lower better
    best_r   = find_best('fat_r',   'max')    # Pearson r higher better

    # RF baseline values
    rf_ex_f1   = rf['exercise']['f1_mean']
    rf_ex_std  = rf['exercise']['f1_std']
    rf_ph_f1   = rf['phase']['ml_f1_mean']
    rf_ph_std  = rf['phase']['ml_f1_std']
    rf_fat_mae = rf['fatigue']['mae_mean']
    rf_fat_std = rf['fatigue']['mae_std']
    rf_r_vals  = list(rf['fatigue']['pearson_r_per_subj'].values())
    rf_r_mean  = float(np.mean(rf_r_vals))
    rf_r_std   = float(np.std(rf_r_vals))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))

    panels = [
        ('Exercise — F1-macro',     axes[0], rf_ex_f1,   rf_ex_std,  best_ex,  '↑', '#3498db'),
        ('Phase — F1-macro',        axes[1], rf_ph_f1,   rf_ph_std,  best_ph,  '↑', '#27ae60'),
        ('Fatigue — RPE MAE',       axes[2], rf_fat_mae, rf_fat_std, best_fat, '↓', '#e67e22'),
        ('Fatigue — Pearson r',     axes[3], rf_r_mean,  rf_r_std,   best_r,   '↑', '#c0392b'),
    ]

    for title, ax, rf_v, rf_s, best, direction, color in panels:
        nn_v, nn_s, slug, win, n_obs = best
        nn_label = f"{slug.replace('multi-', '').replace('fatigue-', 'fat-')} @ {win}"
        labels = ['RF baseline', f'Best NN\n({nn_label})']
        means  = [rf_v, nn_v]
        stds   = [rf_s, nn_s]
        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, capsize=8,
                      color=['#7f8c8d', color], edgecolor='black',
                      linewidth=0.8, error_kw={'linewidth': 1.2})
        span = max(abs(min(means) - max(stds)), abs(max(means) + max(stds)))
        for b, v, s in zip(bars, means, stds):
            offset = 0.04 * span if v >= 0 else -0.06 * span
            va = 'bottom' if v >= 0 else 'top'
            ax.text(b.get_x() + b.get_width() / 2,
                    v + s * np.sign(v + 1e-9) + offset,
                    f'{v:+.3f}\n±{s:.3f}' if direction == '↑' and 'r' in title.lower() else f'{v:.3f}\n±{s:.3f}',
                    ha='center', va=va, fontsize=10)
        # Determine winner via direction
        winner_idx = (0 if rf_v >= nn_v else 1) if direction == '↑' else (0 if rf_v <= nn_v else 1)
        bars[winner_idx].set_edgecolor('gold')
        bars[winner_idx].set_linewidth(3.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(f'{title} ({direction})', fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)
        ax.axhline(0, color='black', linewidth=0.5)
        ymax = max(max(means) + max(stds) + 0.25 * span, 0.0)
        ymin = min(min(means) - max(stds) - 0.25 * span, 0.0)
        ax.set_ylim(ymin, ymax)

    fig.suptitle('Headline results — Random Forest baseline vs best neural network per task '
                 '(LOSO, 7 subjects, 3 seeds, mean ± std)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / 'headline_results.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def window_sweep_4tasks(summaries):
    """2x2 panel: ex_F1, ph_F1, fat_r, reps_MAE vs window length per arch."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    panels = [
        ('Exercise F1-macro (↑)', axes[0, 0], lambda m: m['exercise']['f1_macro']['mean'], None),
        ('Phase F1-macro (↑)',    axes[0, 1], lambda m: m['phase']['f1_macro']['mean'], None),
        ('Fatigue Pearson r (↑)', axes[1, 0], lambda m: m['fatigue']['pearson_r']['mean'], 0),
        ('Fatigue MAE (↓)',       axes[1, 1], lambda m: m['fatigue']['mae']['mean'], None),
    ]

    for title, ax, fn, hline in panels:
        for i, slug in enumerate(ARCHS):
            ys = []
            for w in WINDOWS:
                m = summaries.get(slug, {}).get(w)
                ys.append(fn(m) if m is not None else np.nan)
            ax.plot(WIN_FLOATS, ys, '-o', label=ARCH_LABELS[i],
                    linewidth=1.6, markersize=6, color=ARCH_COLORS[i])
        if hline is not None:
            ax.axhline(hline, color='black', linewidth=0.5)
        ax.set_xlabel('Window length (s)', fontsize=11)
        ax.set_xticks(WIN_FLOATS)
        ax.set_title(title, fontsize=12)
        ax.grid(linestyle=':', alpha=0.5)

    axes[0, 0].set_ylabel('F1-macro')
    axes[0, 1].set_ylabel('F1-macro')
    axes[1, 0].set_ylabel('Pearson r')
    axes[1, 1].set_ylabel('MAE (RPE units)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('V9 window-length sweep — all NN architectures across 4 tasks',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = OUT / 'window_sweep_4tasks.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def fatigue_scatter_tcn3s():
    """Per-set RPE scatter for fatigue-raw-tcn @ 3s (the actual fatigue winner)."""
    rd = ROOT / 'runs/optuna_clean_v9-w3s-fatigue-raw-tcn'
    by_fold = load_test_preds(rd)
    if not by_fold:
        print('SKIP fatigue_scatter_tcn3s — no test_preds')
        return

    fold_colors = ['#27ae60', '#16a085', '#3498db', '#9b59b6',
                   '#e67e22', '#e74c3c', '#f1c40f']
    fig, ax = plt.subplots(figsize=(8, 7))
    set_actual_all, set_pred_all = [], []
    for fk in sorted(by_fold):
        yp = by_fold[fk]['fatigue']['pred']
        yt = by_fold[fk]['fatigue']['true']
        if len(yp) == 0:
            continue
        # Per-set aggregate by integer RPE target
        groups = defaultdict(list)
        for t, p in zip(yt, yp):
            groups[float(t)].append(float(p))
        x = np.array(sorted(groups.keys()))
        y = np.array([np.mean(groups[t]) for t in x])
        ax.scatter(x, y, alpha=0.75, s=90, color=fold_colors[fk],
                   edgecolor='black', linewidth=0.6,
                   label=f'F{fk}: {by_fold[fk]["subj"]}')
        set_actual_all.extend(x.tolist())
        set_pred_all.extend(y.tolist())

    set_actual_all = np.array(set_actual_all)
    set_pred_all = np.array(set_pred_all)
    lo = float(set_actual_all.min()) - 0.5
    hi = float(set_actual_all.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    slope, intercept = np.polyfit(set_actual_all, set_pred_all, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, 'k:', linewidth=1.5,
            label=f'Best fit: y={slope:.2f}x+{intercept:.2f}')
    pearson = float(np.corrcoef(set_actual_all, set_pred_all)[0, 1])
    mae = float(np.mean(np.abs(set_actual_all - set_pred_all)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual RPE (1–10)', fontsize=11)
    ax.set_ylabel('Predicted RPE (per set, mean over 3 seeds)', fontsize=11)
    ax.set_title(f'Fatigue calibration — fatigue-raw-tcn @ 3s '
                 f'(n={len(set_actual_all)} sets, Pearson r={pearson:+.3f}, MAE={mae:.3f})',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = OUT / 'fatigue_scatter_tcn3s.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out} (n={len(set_actual_all)} sets, r={pearson:+.3f})')


def cm_phase_mlp1s():
    """Phase CM for feat-MLP @ 1s (the actual phase winner)."""
    rd = ROOT / 'runs/optuna_clean_v9-w1s-multi-feat-mlp'
    by_fold = load_test_preds(rd)
    if not by_fold:
        print('SKIP cm_phase_mlp1s — no test_preds')
        return

    yp_all, yt_all = [], []
    for fk in sorted(by_fold):
        if len(by_fold[fk]['phase']['pred']):
            yp_all.append(by_fold[fk]['phase']['pred'])
            yt_all.append(by_fold[fk]['phase']['true'])
    yp = np.concatenate(yp_all)
    yt = np.concatenate(yt_all)
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    labels = [PH_CLASSES[i] if i < len(PH_CLASSES) else f'cls_{i}' for i in present]
    cm = sk_cm(yt, yp, labels=present)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap='Greens', vmin=0, vmax=cm_norm.max())
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n(n={cm[i, j]})',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black',
                    fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Phase CM — feat-MLP @ 1s (n={len(yt)} windows × 3 seeds × 7 folds)',
                 fontsize=11)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out = OUT / 'cm_phase_mlp1s.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out} ({len(yt)} samples)')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    summaries = load_all_summaries()
    headline_figure(summaries)
    window_sweep_4tasks(summaries)
    fatigue_scatter_tcn3s()
    cm_phase_mlp1s()
    print(f'\nAll thesis plots in {OUT}')


if __name__ == '__main__':
    main()
