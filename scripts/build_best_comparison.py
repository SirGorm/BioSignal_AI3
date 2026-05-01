"""Build best-models comparison page with embedded plots, raw vs features."""
import json
from pathlib import Path

OUT_DIR = Path('runs/20260429_103521_overnight-comparison/best_models')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cv(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else None


feat = {
    'mlp':      load_cv('runs/20260429_105937_short-mlp-features/mlp/cv_summary.json'),
    'cnn1d':    load_cv('runs/20260427_200403_nn-full-cnn1d/cnn1d/cv_summary.json'),
    'lstm':     load_cv('runs/20260427_232457_nn-full-lstm/lstm/cv_summary.json'),
    'cnn_lstm': load_cv('runs/20260428_010720_nn-full-cnn_lstm/cnn_lstm/cv_summary.json'),
    'tcn':      load_cv('runs/20260428_025001_nn-full-tcn/tcn/cv_summary.json'),
}
# Use the SHORT raw runs (1 fold * 1 seed * 5 epochs) — these have test_preds.pt
# and full plot suite, unlike the older raw_full run.
raw = {
    'cnn1d':    load_cv('runs/20260429_111752_short-cnn1d_raw/cnn1d_raw/cv_summary.json'),
    'lstm':     load_cv('runs/20260429_111919_short-lstm_raw/lstm_raw/cv_summary.json'),
    'cnn_lstm': load_cv('runs/20260429_112046_short-cnn_lstm_raw/cnn_lstm_raw/cv_summary.json'),
    'tcn':      load_cv('runs/20260429_112213_short-tcn_raw/tcn_raw/cv_summary.json'),
}
baseline = load_cv('runs/20260427_110653_default/lgbm/cv_summary.json')


def get(s, task, metric):
    if s is None:
        return None
    sub = s.get('summary', s).get(task, {}).get(metric, {})
    return sub if isinstance(sub, dict) and 'mean' in sub else None


def fmt(d):
    return f"{d['mean']:.3f} +/- {d.get('std', 0):.3f}" if d else '—'


def best(task, metric, sign):
    candidates = []
    for variant, models in [('features', feat), ('raw', raw)]:
        for arch, s in models.items():
            d = get(s, task, metric)
            if d:
                candidates.append((sign * d['mean'], variant, arch, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1:]


tasks = [
    ('exercise', 'f1_macro', +1, 'Exercise F1'),
    ('phase',    'f1_macro', +1, 'Phase F1'),
    ('fatigue',  'mae',      -1, 'Fatigue MAE'),
    ('reps',     'mae',      -1, 'Reps MAE'),
]

lines = []
lines.append("# Best Models — Raw input vs Features\n")
lines.append("Compares the best NN architecture per task across two input variants:")
lines.append("- **Features**: 34 hand-engineered features (HRV, EMG MNF/MDF, EDA SCL/SCR, etc.)")
lines.append("- **Raw**: stacked multimodal raw signal windows (6 modalities x time)\n")
lines.append("Baseline: LightGBM on features (per-set features for fatigue/reps, per-window for exercise/phase).\n")

lines.append("## Master comparison\n")
lines.append("Note: NN runs labelled `feat-*` use 3 seeds x 5 folds (overnight). `feat-mlp` and all `raw-*` are short verification runs (1 seed x N folds x 5 epochs) — used here primarily for plot completeness; their numbers are noisier.\n")
header_cells = ['Task', 'LightGBM', 'feat-mlp']
for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
    header_cells.append(f'feat-{arch}')
    header_cells.append(f'raw-{arch}')
header_cells.append('Best')
lines.append('| ' + ' | '.join(header_cells) + ' |')
lines.append('|' + '---|' * len(header_cells))

for task, metric, sign, label in tasks:
    row = [label, fmt(get(baseline, task, metric)), fmt(get(feat.get('mlp'), task, metric))]
    for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
        row.append(fmt(get(feat.get(arch), task, metric)))
        row.append(fmt(get(raw.get(arch), task, metric)))
    b = best(task, metric, sign)
    row.append(f"**{b[0]}-{b[1]}**" if b else '—')
    lines.append('| ' + ' | '.join(row) + ' |')

lines.append("\n## Per-task winners\n")
lines.append("| Task | Variant | Architecture | Score | vs LightGBM |")
lines.append("|---|---|---|---|---|")
for task, metric, sign, label in tasks:
    b = best(task, metric, sign)
    if not b:
        continue
    variant, arch, d = b
    lgbm = get(baseline, task, metric)
    if lgbm:
        delta = (d['mean'] - lgbm['mean']) * sign
        delta_str = f"{'+' if delta > 0 else ''}{delta:.3f}"
    else:
        delta_str = '—'
    lines.append(f"| {label} | {variant} | **{arch}** | {fmt(d)} | {delta_str} |")

lines.append("\n## Plots per winner\n")
plot_dirs = {
    ('features', 'mlp'):      'runs/20260429_105937_short-mlp-features',
    ('features', 'cnn1d'):    'runs/20260427_200403_nn-full-cnn1d',
    ('features', 'lstm'):     'runs/20260427_232457_nn-full-lstm',
    ('features', 'cnn_lstm'): 'runs/20260428_010720_nn-full-cnn_lstm',
    ('features', 'tcn'):      'runs/20260428_025001_nn-full-tcn',
    ('raw', 'cnn1d'):    'runs/20260429_111752_short-cnn1d_raw',
    ('raw', 'lstm'):     'runs/20260429_111919_short-lstm_raw',
    ('raw', 'cnn_lstm'): 'runs/20260429_112046_short-cnn_lstm_raw',
    ('raw', 'tcn'):      'runs/20260429_112213_short-tcn_raw',
}

seen = set()
for task, metric, sign, label in tasks:
    b = best(task, metric, sign)
    if not b:
        continue
    variant, arch, d = b
    key = (variant, arch)
    if key in seen:
        continue
    seen.add(key)
    run_dir = Path(plot_dirs[key])
    rel = lambda p: '../../../' + str(p).replace('\\', '/')
    lines.append(f"### {label} winner — `{variant}-{arch}` ({fmt(d)})\n")
    for plot_name, caption in [
        ('training_curves_aggregated.png', 'Training curves (mean +/- SD across folds x seeds)'),
        ('confusion_matrix_exercise.png',  'Confusion matrix — Exercise classification'),
        ('confusion_matrix_phase.png',     'Confusion matrix — Phase classification'),
        ('fatigue_calibration.png',        'Fatigue calibration (predicted vs true RPE)'),
        ('reps_evaluation.png',            'Reps evaluation (calibration + tolerance accuracy)'),
    ]:
        plot_path = run_dir / plot_name
        if plot_path.exists():
            lines.append(f"**{caption}**")
            lines.append(f"![{caption}]({rel(plot_path)})\n")
        else:
            lines.append(f"_({caption}: not available)_\n")

lines.append("\n## All models — full evaluation gallery\n")
lines.append("Every trained model below has training curves, both confusion matrices, fatigue calibration, and reps evaluation (4-tolerance bar chart).\n")
all_keys = [
    ('features', 'mlp'),      ('features', 'cnn1d'),    ('features', 'lstm'),
    ('features', 'cnn_lstm'), ('features', 'tcn'),
    ('raw', 'cnn1d'),         ('raw', 'lstm'),
    ('raw', 'cnn_lstm'),      ('raw', 'tcn'),
]
for variant, arch in all_keys:
    if (variant, arch) in seen:
        continue
    run_dir = Path(plot_dirs[(variant, arch)])
    rel = lambda p: '../../../' + str(p).replace('\\', '/')
    lines.append(f"### {variant}-{arch}\n")
    for plot_name in [
        'training_curves_aggregated.png', 'confusion_matrix_exercise.png',
        'confusion_matrix_phase.png', 'fatigue_calibration.png',
        'reps_evaluation.png',
    ]:
        p = run_dir / plot_name
        if p.exists():
            lines.append(f"![{plot_name}]({rel(p)})\n")

lines.append("\n## Raw vs Features — head-to-head per architecture (Exercise F1)\n")
lines.append("| Architecture | Features F1 | Raw F1 | Delta |")
lines.append("|---|---|---|---|")
for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
    f_d = get(feat.get(arch), 'exercise', 'f1_macro')
    r_d = get(raw.get(arch), 'exercise', 'f1_macro')
    if f_d and r_d:
        delta = r_d['mean'] - f_d['mean']
        delta_str = f"{delta:+.3f}"
    else:
        delta_str = '—'
    lines.append(f"| {arch} | {fmt(f_d)} | {fmt(r_d)} | {delta_str} |")

lines.append("\n## Caveats\n")
lines.append("- LSTM crashed on Windows process exit (STATUS_STACK_BUFFER_OVERRUN) but all 15 folds completed; cv_summary.json intact.")
lines.append("- Raw runs lack test_preds.pt — no confusion matrices for raw models.")
lines.append("- LightGBM has 5 fold values; NN runs have 3 seeds x 5 folds. Paired tests aggregate seeds to per-fold means. With n=5 subject groups, Wilcoxon minimum p = 0.0625; Bonferroni-corrected p never reaches 0.05 even when Cohen d > 3.")
lines.append("- For real-time deployment, only causal architectures are valid (TCN, unidirectional LSTM, causal-padded 1D-CNN). BiLSTM raw_lstm_p2 is research-only — see CLAUDE.md.\n")

(OUT_DIR / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
print(f"Wrote {OUT_DIR / 'comparison.md'}")
print(f"Lines: {len(lines)}")
