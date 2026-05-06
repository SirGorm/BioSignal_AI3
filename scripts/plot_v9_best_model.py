"""All plots for the lowest-composite-loss v9 model: multi-feat-LSTM @ 1s.

Outputs to runs/comparison_v9/best_model/:
  - cm_exercise.png             — exercise confusion matrix (4 classes)
  - cm_phase.png                — phase confusion matrix
  - fatigue_scatter.png         — predicted vs actual RPE per set
  - fatigue_per_subject.png     — Pearson r + MAE per subject
  - phase_per_subject.png       — F1 per subject
  - exercise_per_subject.png    — F1 per subject
  - reps_scatter.png            — predicted vs actual reps (soft_overlap fractions)
  - reps_per_subject.png        — MAE per subject
  - optuna_history.png          — trial-by-trial score evolution
  - hps_summary.png             — best HPs visualisation
"""
from __future__ import annotations
import json, sqlite3, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as sk_cm

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "runs" / "optuna_clean_v9-w1s-multi-feat-lstm"
OUT = ROOT / "runs" / "comparison_v9" / "best_model"
OUT.mkdir(parents=True, exist_ok=True)

FOLD_SUBJECTS = {
    0: 'Vivian', 1: 'Hytten', 2: 'kiyomi', 3: 'lucas 2',
    4: 'Tias', 5: 'Juile', 6: 'Raghild',
}
EX_CLASSES = ['benchpress', 'deadlift', 'pullup', 'squat']
PH_CLASSES_DEFAULT = ['concentric', 'eccentric', 'rest']


def collect_preds():
    """Walk phase2/*/seed_*/fold_*/test_preds.pt and aggregate per task."""
    p2 = RD / 'phase2'
    by_fold = defaultdict(lambda: {
        'subj': None,
        'exercise': {'pred': [], 'true': []},
        'phase':    {'pred': [], 'true': []},
        'fatigue':  {'pred': [], 'true': []},
        'reps':     {'pred': [], 'true': []},
    })
    n_seeds_per_fold = defaultdict(int)
    for seed_dir in p2.glob('*/seed_*'):
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
                if pred is None: continue
                pred = pred.numpy(); true = true.numpy(); mask = mask.numpy().astype(bool)
                if not mask.any(): continue
                if task in ('exercise', 'phase') and pred.ndim == 2:
                    pred = pred.argmax(axis=1)
                by_fold[fk][task]['pred'].append(pred[mask])
                by_fold[fk][task]['true'].append(true[mask])
            n_seeds_per_fold[fk] += 1
    # Concatenate across seeds for each fold/task
    for fk in by_fold:
        for task in ('exercise', 'phase', 'fatigue', 'reps'):
            if by_fold[fk][task]['pred']:
                by_fold[fk][task]['pred'] = np.concatenate(by_fold[fk][task]['pred'])
                by_fold[fk][task]['true'] = np.concatenate(by_fold[fk][task]['true'])
            else:
                by_fold[fk][task]['pred'] = np.array([])
                by_fold[fk][task]['true'] = np.array([])
    return by_fold


def plot_cm(yt, yp, classes, title, out_path, cmap):
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    labels = [classes[i] if i < len(classes) else f'cls_{i}' for i in present]
    cm = sk_cm(yt, yp, labels=present)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=cm_norm.max())
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                     color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    by_fold = collect_preds()
    print(f'Loaded {len(by_fold)} folds × ~3 seeds')

    # === 1. Confusion matrices (concat across all folds × seeds) ===
    all_ex_p, all_ex_t = [], []
    all_ph_p, all_ph_t = [], []
    for fk in sorted(by_fold):
        all_ex_p.append(by_fold[fk]['exercise']['pred'])
        all_ex_t.append(by_fold[fk]['exercise']['true'])
        all_ph_p.append(by_fold[fk]['phase']['pred'])
        all_ph_t.append(by_fold[fk]['phase']['true'])
    yp_ex = np.concatenate([a for a in all_ex_p if len(a)])
    yt_ex = np.concatenate([a for a in all_ex_t if len(a)])
    yp_ph = np.concatenate([a for a in all_ph_p if len(a)])
    yt_ph = np.concatenate([a for a in all_ph_t if len(a)])
    plot_cm(yt_ex, yp_ex, EX_CLASSES,
             f'Exercise CM — feat-LSTM @ 1s (n={len(yt_ex)} windows × 3 seeds × 7 folds)',
             OUT / 'cm_exercise.png', 'Blues')
    print(f'Wrote cm_exercise.png ({len(yt_ex)} samples)')
    plot_cm(yt_ph, yp_ph, PH_CLASSES_DEFAULT,
             f'Phase CM — feat-LSTM @ 1s (n={len(yt_ph)})',
             OUT / 'cm_phase.png', 'Greens')
    print(f'Wrote cm_phase.png ({len(yt_ph)} samples)')

    # === 2. Fatigue scatter — aggregate per (fold, target_RPE) to get per-set ===
    fold_colors = ['#27ae60', '#16a085', '#3498db', '#9b59b6',
                    '#e67e22', '#e74c3c', '#f1c40f']
    fig, ax = plt.subplots(figsize=(8, 7))
    set_actual_all, set_pred_all = [], []
    for fk in sorted(by_fold):
        yp = by_fold[fk]['fatigue']['pred']
        yt = by_fold[fk]['fatigue']['true']
        if len(yp) == 0: continue
        # Per-set aggregate: group windows by their target RPE within this fold
        groups = defaultdict(list)
        for t, p in zip(yt, yp):
            groups[float(t)].append(float(p))
        x = np.array(sorted(groups.keys()))
        y = np.array([np.mean(groups[t]) for t in x])
        ax.scatter(x, y, alpha=0.7, s=80, color=fold_colors[fk],
                    edgecolor='black', linewidth=0.5,
                    label=f'F{fk}: {by_fold[fk]["subj"]}')
        set_actual_all.extend(x); set_pred_all.extend(y)
    set_actual_all = np.array(set_actual_all); set_pred_all = np.array(set_pred_all)
    lo, hi = set_actual_all.min()-0.5, set_actual_all.max()+0.5
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    slope, intercept = np.polyfit(set_actual_all, set_pred_all, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope*xs+intercept, 'k:', linewidth=1.5,
             label=f'Best fit: y={slope:.2f}x+{intercept:.2f}')
    pearson = float(np.corrcoef(set_actual_all, set_pred_all)[0, 1])
    mae = float(np.mean(np.abs(set_actual_all - set_pred_all)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual RPE (1-10)')
    ax.set_ylabel('Predicted RPE (per set, mean over 3 seeds)')
    ax.set_title(f'Fatigue calibration — feat-LSTM @ 1s (n={len(set_actual_all)} sets, '
                  f'Pearson r={pearson:+.3f}, MAE={mae:.3f})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fatigue_scatter.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote fatigue_scatter.png (n={len(set_actual_all)} sets, r={pearson:+.3f})')

    # === 3. Per-subject metrics bar charts ===
    subj_names = [by_fold[fk]['subj'] for fk in sorted(by_fold)]
    subj_idx = list(sorted(by_fold))

    def per_subject_metric(task, metric_fn):
        return [metric_fn(by_fold[fk][task]['true'], by_fold[fk][task]['pred'])
                for fk in subj_idx]

    from sklearn.metrics import f1_score
    ex_f1 = per_subject_metric('exercise', lambda t, p: f1_score(t, p, average='macro', zero_division=0) if len(t) else 0)
    ph_f1 = per_subject_metric('phase',    lambda t, p: f1_score(t, p, average='macro', zero_division=0) if len(t) else 0)
    fat_r = per_subject_metric('fatigue',  lambda t, p: float(np.corrcoef(t, p)[0,1]) if len(t)>1 and np.std(p)>0 else 0)
    fat_mae = per_subject_metric('fatigue', lambda t, p: float(np.mean(np.abs(t-p))) if len(t) else 0)
    reps_mae = per_subject_metric('reps',  lambda t, p: float(np.mean(np.abs(t-p))) if len(t) else 0)

    def bar_plot(values, title, ylabel, out_path, hline=None, color='#3498db'):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(subj_names))
        bars = ax.bar(x, values, color=color, edgecolor='black', linewidth=0.5)
        for b, v in zip(bars, values):
            ax.text(b.get_x()+b.get_width()/2,
                     v + (0.01*max(abs(min(values)), abs(max(values))) if v >= 0 else -0.03*max(abs(min(values)), abs(max(values)))),
                     f'{v:+.2f}' if 'Pearson' in ylabel else f'{v:.3f}',
                     ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        if hline is not None:
            ax.axhline(hline, color='grey', linestyle='--', linewidth=1, label=f'baseline = {hline:.3f}')
            ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(subj_names, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)

    bar_plot(ex_f1, 'Exercise F1-macro per subject — feat-LSTM @ 1s',
              'F1-macro', OUT / 'exercise_per_subject.png',
              hline=0.123, color='#3498db')
    bar_plot(ph_f1, 'Phase F1-macro per subject — feat-LSTM @ 1s',
              'F1-macro', OUT / 'phase_per_subject.png',
              hline=0.186, color='#27ae60')
    bar_plot(fat_r, 'Fatigue Pearson r per subject — feat-LSTM @ 1s',
              'Pearson r', OUT / 'fatigue_per_subject.png',
              color='#e67e22')
    bar_plot(fat_mae, 'Fatigue MAE per subject — feat-LSTM @ 1s',
              'MAE (RPE units)', OUT / 'fatigue_mae_per_subject.png',
              hline=1.013, color='#e67e22')
    bar_plot(reps_mae, 'Reps MAE per subject — feat-LSTM @ 1s (soft_overlap fraction)',
              'MAE (fractional reps per window)', OUT / 'reps_per_subject.png',
              color='#9b59b6')
    print(f'Wrote 5 per-subject plots')

    # === 4. Reps scatter ===
    fig, ax = plt.subplots(figsize=(8, 6))
    yp_all, yt_all = [], []
    for fk in sorted(by_fold):
        yp = by_fold[fk]['reps']['pred']
        yt = by_fold[fk]['reps']['true']
        if len(yp) == 0: continue
        ax.scatter(yt, yp, alpha=0.3, s=15, color=fold_colors[fk],
                    label=f'F{fk}: {by_fold[fk]["subj"]}')
        yp_all.append(yp); yt_all.append(yt)
    yp_all = np.concatenate(yp_all); yt_all = np.concatenate(yt_all)
    lo = min(yp_all.min(), yt_all.min())
    hi = max(yp_all.max(), yt_all.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect')
    pearson_r = float(np.corrcoef(yt_all, yp_all)[0, 1])
    mae = float(np.mean(np.abs(yt_all - yp_all)))
    ax.set_xlabel('Actual reps (soft_overlap fractions per window)')
    ax.set_ylabel('Predicted reps')
    ax.set_title(f'Reps prediction — feat-LSTM @ 1s (n={len(yt_all)} windows, '
                  f'r={pearson_r:+.3f}, MAE={mae:.3f})')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / 'reps_scatter.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote reps_scatter.png (n={len(yt_all)})')

    # === 5. Optuna optimization history ===
    db = RD / 'optuna.db'
    if db.exists():
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("""SELECT t.number, tv.value FROM trials t
                       JOIN trial_values tv ON t.trial_id=tv.trial_id
                       WHERE t.state='COMPLETE' ORDER BY t.number""")
        trials = cur.fetchall()
        conn.close()
        if trials:
            ns = [t[0] for t in trials]
            vs = [t[1] for t in trials]
            best_so_far = np.minimum.accumulate(vs)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(ns, vs, alpha=0.4, s=25, color='#3498db', label='Trial score')
            ax.plot(ns, best_so_far, color='#e74c3c', linewidth=2, label='Best so far')
            ax.set_xlabel('Trial number')
            ax.set_ylabel('Optuna composite score (lower = better)')
            ax.set_title(f'Optuna optimization history — feat-LSTM @ 1s ({len(trials)} trials)')
            ax.legend()
            ax.grid(linestyle=':', alpha=0.5)
            fig.tight_layout()
            fig.savefig(OUT / 'optuna_history.png', dpi=140, bbox_inches='tight')
            plt.close(fig)
            print(f'Wrote optuna_history.png ({len(trials)} trials)')

    # === 6. Best HPs summary as text ===
    bh = json.load(open(RD / 'best_hps.json'))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    txt = (f'Best HPs (feat-LSTM @ 1s, multi-task, 75 trials)\n\n'
           f'best_score = {bh["best_score"]:.4f}\n\n')
    for k, v in bh['best_hps'].items():
        if k.startswith('_'): continue
        if isinstance(v, float):
            txt += f'  {k:<18} = {v:.6g}\n'
        else:
            txt += f'  {k:<18} = {v}\n'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va='top', fontsize=11,
             family='monospace')
    fig.tight_layout()
    fig.savefig(OUT / 'hps_summary.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote hps_summary.png')

    print(f'\nAll plots in {OUT}')


if __name__ == '__main__':
    main()
