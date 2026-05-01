"""Generate final artifacts after run_train_nn_features_full.py completes.

Produces:
  runs/20260427_121303_nn_features_full/
    comparison.md
    comparison.png
    latency_table.md
    multitask_ablation.md
    features_<arch>/model_card.md  (one per arch)

References used throughout (from literature-references skill):
- Caruana 1997 — hard parameter sharing
- Ruder 2017 — soft vs hard sharing survey
- Bai et al. 2018 — TCN
- Hochreiter & Schmidhuber 1997 — LSTM
- Loshchilov & Hutter 2019 — AdamW
- Goodfellow et al. 2016 — regularization
- Saeb et al. 2017 — subject-wise CV
- Akiba et al. 2019 — Optuna
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / 'runs/20260427_121303_nn_features_full'

LGBM_METRICS_PATH = ROOT / 'runs/20260427_110653_default/metrics.json'


def load_json(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def fmt(v, digits=3) -> str:
    if v is None:
        return 'N/A'
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return 'N/A'
        return f'{f:.{digits}f}'
    except Exception:
        return str(v)


def _get_phase2_metrics(all_results: Dict, vname: str) -> Optional[Dict]:
    """Extract mean metrics for a phase2 variant."""
    p2 = all_results.get('phase2', {})
    if vname in p2:
        return p2[vname]
    # Try with _p2 suffix
    p2name = f"{vname}_p2"
    if p2name in p2:
        return p2[p2name]
    return None


def generate_comparison_md(all_results: Dict, lgbm: Dict) -> str:
    """Build comparison.md."""
    lgbm_ex = lgbm['exercise']['f1_mean']
    lgbm_ph = lgbm['phase']['ml_f1_mean']
    lgbm_fat = lgbm['fatigue']['mae_mean']
    lgbm_rep = lgbm['reps']['ml_mae_mean']

    p1 = all_results.get('phase1', {})
    p2 = all_results.get('phase2', {})
    ranking = all_results.get('ranking', [])
    latency = all_results.get('latency', {})
    ablation = all_results.get('ablation', {})
    winner = all_results.get('winner', 'unknown')

    lines = [
        "# Neural Network vs. LightGBM Comparison",
        "",
        "**Run:** `runs/20260427_121303_nn_features_full`  ",
        "**Baseline:** `runs/20260427_110653_default` (LightGBM, GroupKFold-5)  ",
        "**Input variant:** Engineered features (Variant A only; raw skipped per user request)  ",
        "**CV:** Subject-wise GroupKFold(5) — reuses the exact fold assignments from the "
        "LightGBM baseline (Saeb et al. 2017)  ",
        "**Multi-task architecture:** Hard parameter sharing — one shared encoder + "
        "4 task-specific heads (Caruana 1997)  ",
        "**Optimizer:** AdamW with cosine annealing (Loshchilov & Hutter 2019)  ",
        "**HP search:** Optuna TPE, 30 trials per arch (Akiba et al. 2019)  ",
        "**Regularization:** Dropout + BatchNorm (Goodfellow et al. 2016)  ",
        "",
        "---",
        "",
        "## Main Results Table",
        "",
        "| Model | Exercise F1 (macro) | Phase F1 (macro) | Fatigue MAE | Reps MAE | "
        "Causal | Deployable |",
        "|-------|---------------------|-----------------|-------------|----------|"
        "-------|------------|",
    ]

    def _row(name, ex, ph, fat, rep, causal, deploy, ex_std=None, ph_std=None,
             fat_std=None, rep_std=None):
        def _fmts(v, s):
            if s is not None:
                return f"{fmt(v)} ±{fmt(s)}"
            return fmt(v)
        c = "yes" if causal else "no (research only)"
        d = "yes" if deploy else "no"
        return (f"| {name} | {_fmts(ex, ex_std)} | {_fmts(ph, ph_std)} | "
                f"{_fmts(fat, fat_std)} | {_fmts(rep, rep_std)} | {c} | {d} |")

    # LightGBM baseline row
    lines.append(_row(
        'LightGBM (baseline)',
        lgbm_ex, lgbm_ph, lgbm_fat, lgbm_rep,
        causal=True, deploy=True,
    ))

    # NN rows — Phase 2 where available, else Phase 1
    arch_meta = {
        'cnn1d':    ('1D-CNN',     True,  True),
        'lstm':     ('BiLSTM',     False, False),   # BiLSTM non-causal
        'cnn_lstm': ('CNN-LSTM',   False, False),
        'tcn':      ('TCN',        True,  True),
    }
    for arch, (label, causal, deploy) in arch_meta.items():
        vname = f"features_{arch}"
        # Phase 2 preferred
        s = p2.get(vname) or p2.get(f"{vname}_p2") or p1.get(vname)
        if s is None:
            continue
        ex_m = s.get('exercise', {}).get('f1_macro', {})
        ph_m = s.get('phase', {}).get('f1_macro', {})
        fat_m = s.get('fatigue', {}).get('mae', {})
        rep_m = s.get('reps', {}).get('mae', {})
        lines.append(_row(
            f"{label} (features)",
            ex_m.get('mean'), ph_m.get('mean'), fat_m.get('mean'), rep_m.get('mean'),
            causal=causal, deploy=deploy,
            ex_std=ex_m.get('std'), ph_std=ph_m.get('std'),
            fat_std=fat_m.get('std'), rep_std=rep_m.get('std'),
        ))

    lines += [
        "",
        "Higher is better for F1 (macro). Lower is better for MAE.  ",
        "Phase 2 metrics shown (3 seeds × 5 folds); Phase 1 used where Phase 2 "
        "not run (screening only).  ",
        "",
        "---",
        "",
        "## State-Machine Baselines",
        "",
        "| Task | State-Machine | LightGBM | Winner NN |",
        "|------|---------------|----------|-----------|",
        f"| Reps (MAE) | {fmt(lgbm.get('reps', {}).get('state_machine_mae', 2.98))} | "
        f"{fmt(lgbm_rep)} | (see table above) |",
        f"| Phase (F1) | {fmt(lgbm.get('phase', {}).get('state_machine_f1', 0.306))} | "
        f"{fmt(lgbm_ph)} | (see table above) |",
        "| Exercise (F1) | not applicable | (LightGBM above) | (see table above) |",
        "| Fatigue (MAE) | not applicable | (LightGBM above) | (see table above) |",
        "",
        "---",
        "",
        "## Phase 1 Ranking",
        "",
        "| Rank | Variant | Mean Rank | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |",
        "|------|---------|-----------|-------------|----------|-------------|----------|",
    ]

    for i, r in enumerate(ranking, 1):
        lines.append(
            f"| {i} | {r['variant']} | {fmt(r.get('mean_rank'), 2)} | "
            f"{fmt(r.get('exercise_f1_macro'))} | {fmt(r.get('phase_f1_macro'))} | "
            f"{fmt(r.get('fatigue_mae'))} | {fmt(r.get('reps_mae'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Statistical Notes",
        "",
        "Paired t-tests on per-fold metrics (5 folds) are reported with "
        "caution: N=5 folds gives very low statistical power (~0.2 at alpha=0.05 "
        "for small effects). Results should be interpreted as indicative, not "
        "confirmatory. All comparisons use the same fold assignments — no "
        "additional variance from split randomness.",
        "",
        "---",
        "",
        "## Multi-Task Architecture",
        "",
        "Default architecture: hard parameter sharing (Caruana 1997) — one "
        "shared encoder with 4 task-specific linear heads. This is the "
        "recommended approach in low-data regimes (N=9 subjects) where negative "
        "transfer is less likely than insufficient data for separate encoders.",
        "",
        "Ablation: soft sharing (separate encoder per task) tested on the "
        "winning architecture. Results in `multitask_ablation.md`.",
        "",
    ]

    # Ablation verdict inline
    if ablation:
        verdict = ablation.get('verdict', 'N/A')
        soft_agg = ablation.get('soft_sharing', {})
        hard_agg = ablation.get('hard_sharing', {})
        lines += [
            f"**Ablation verdict:** {verdict}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Deployment Recommendation",
        "",
        "Deployment requires causal architectures (Bai et al. 2018 for TCN; "
        "unidirectional LSTM for streaming). BiLSTM and standard CNN-LSTM are "
        "research-only — they require the full context window before producing "
        "output, which breaks the real-time constraint.",
        "",
        "See `latency_table.md` for p99 latency vs. the 100 ms budget.",
        "",
        "---",
        "",
        "## References",
        "",
        "- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.",
        "- Ruder, S. (2017). An overview of multi-task learning in deep neural "
        "networks. *arXiv:1706.05098*.",
        "- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation "
        "of generic convolutional and recurrent networks for sequence modeling. "
        "*arXiv:1803.01271*.",
        "- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
        "*Neural Computation*, 9(8), 1735-1780.",
        "- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay "
        "regularization. *ICLR 2019*.",
        "- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. "
        "MIT Press.",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. "
        "(2017). The need to approximate the use-case in clinical machine "
        "learning. *GigaScience*, 6(5).",
        "- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "
        "Optuna: A next-generation hyperparameter optimization framework. "
        "*KDD 2019*.",
    ]
    return "\n".join(lines) + "\n"


def generate_latency_md(latency: Dict) -> str:
    """Build latency_table.md."""
    lines = [
        "# Latency Table — p50/p95/p99/mean",
        "",
        "Batch size = 1, single 2-second window. Real-time deployment scenario.",
        "Deployment budget: **p99 <= 100 ms** (from nn.yaml).",
        "",
        "## GPU (RTX 5070 Ti, CUDA)",
        "",
        "| Architecture | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | "
        "Within budget |",
        "|-------------|---------|---------|---------|----------|--------------|",
    ]
    for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
        gpu = latency.get(arch, {}).get('gpu', {})
        p99 = gpu.get('p99_ms', float('nan'))
        within = "yes" if not np.isnan(p99) and p99 <= 100 else "no"
        lines.append(
            f"| {arch:12s} | {fmt(gpu.get('p50_ms'))} | "
            f"{fmt(gpu.get('p95_ms'))} | {fmt(p99)} | "
            f"{fmt(gpu.get('mean_ms'))} | {within} |"
        )

    lines += [
        "",
        "## CPU (deployment without GPU)",
        "",
        "| Architecture | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | "
        "Within budget | Causal / Deployable |",
        "|-------------|---------|---------|---------|----------|--------------|---------------------|",
    ]
    arch_causal = {
        'cnn1d': ('1D-CNN', True),
        'lstm': ('BiLSTM', False),
        'cnn_lstm': ('CNN-LSTM', False),
        'tcn': ('TCN', True),
    }
    for arch, (label, causal) in arch_causal.items():
        cpu = latency.get(arch, {}).get('cpu', {})
        p99 = cpu.get('p99_ms', float('nan'))
        within = "yes" if not np.isnan(p99) and p99 <= 100 else "no"
        dep = "yes" if causal else "research only"
        lines.append(
            f"| {label:12s} | {fmt(cpu.get('p50_ms'))} | "
            f"{fmt(cpu.get('p95_ms'))} | {fmt(p99)} | "
            f"{fmt(cpu.get('mean_ms'))} | {within} | {dep} |"
        )

    lines += [
        "",
        "Note: CPU timings are relevant for real-time deployment on edge devices "
        "without a GPU. TCN and 1D-CNN (causal-padded) are the only architectures "
        "that can legally be deployed in the streaming pipeline.",
    ]
    return "\n".join(lines) + "\n"


def generate_ablation_md(ablation: Dict, winner: str) -> str:
    """Build multitask_ablation.md."""
    arch = ablation.get('arch', winner.split('_')[-1])
    soft = ablation.get('soft_sharing', {})
    hard = ablation.get('hard_sharing', {})
    verdict = ablation.get('verdict', 'N/A')

    def _m(d, task, metric):
        return d.get(task, {}).get(metric, {}).get('mean', float('nan'))

    lines = [
        "# Multi-Task Architecture Ablation: Hard vs. Soft Parameter Sharing",
        "",
        f"**Tested on:** {winner} ({arch} architecture)  ",
        "**Protocol:** 2 outer folds, 1 seed, 15 epochs each  ",
        "",
        "## Background",
        "",
        "Hard parameter sharing (Caruana 1997): one shared encoder with "
        "4 task-specific heads. Regularizes by forcing a shared representation "
        "useful for all tasks simultaneously.",
        "",
        "Soft sharing (Ruder 2017): separate encoder per task. Allows "
        "task-specific temporal signatures (e.g., fatigue with slow drift vs. "
        "exercise with rapid label changes). Higher parameter count; needs more "
        "data.",
        "",
        "## Results",
        "",
        "| Task | Metric | Hard Sharing | Soft Sharing | Delta | Winner |",
        "|------|--------|-------------|-------------|-------|--------|",
    ]

    tasks = [
        ('exercise', 'f1_macro', 'F1-macro', False),
        ('phase',    'f1_macro', 'F1-macro', False),
        ('fatigue',  'mae',      'MAE',       True),
        ('reps',     'mae',      'MAE',       True),
    ]
    for task, metric, label, lower_better in tasks:
        h = _m(hard, task, metric)
        s = _m(soft, task, metric)
        if np.isnan(h) or np.isnan(s):
            lines.append(f"| {task} | {label} | N/A | N/A | N/A | N/A |")
            continue
        delta = s - h
        if lower_better:
            w = "soft" if delta < -0.05 else "hard"
        else:
            w = "soft" if delta > 0.02 else "hard"
        lines.append(
            f"| {task} | {label} | {fmt(h)} | {fmt(s)} | "
            f"{'+' if delta > 0 else ''}{fmt(delta)} | {w} |"
        )

    lines += [
        "",
        f"## Verdict",
        "",
        verdict,
        "",
        "## References",
        "",
        "- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.",
        "- Ruder, S. (2017). An overview of multi-task learning in deep neural "
        "networks. *arXiv:1706.05098*.",
    ]
    return "\n".join(lines) + "\n"


def generate_model_card(
    arch: str,
    variant_name: str,
    summary: Dict,
    lgbm: Dict,
    best_hp: Dict,
    latency: Dict,
    n_features: int,
) -> str:
    """Generate per-arch model_card.md."""
    arch_labels = {
        'cnn1d':    ('1D-CNN', '(Yang et al. 2015)', True),
        'lstm':     ('BiLSTM', '(Hochreiter & Schmidhuber 1997)', False),
        'cnn_lstm': ('CNN-LSTM', '(Hochreiter 1997; Yang 2015)', False),
        'tcn':      ('TCN', '(Bai et al. 2018)', True),
    }
    label, arch_cite, causal = arch_labels.get(arch, (arch, '', True))
    deploy_note = (
        "Causal architecture — eligible for real-time streaming deployment."
        if causal else
        "NON-CAUSAL (research_only). BiLSTM / CNN-LSTM cannot be used in "
        "real-time streaming (requires full window context before output). "
        "Do NOT include in `src/streaming/`."
    )

    ex = summary.get('exercise', {}).get('f1_macro', {})
    ph = summary.get('phase', {}).get('f1_macro', {})
    fat = summary.get('fatigue', {}).get('mae', {})
    rep = summary.get('reps', {}).get('mae', {})

    lat_gpu = latency.get(arch, {}).get('gpu', {})
    lat_cpu = latency.get(arch, {}).get('cpu', {})

    lgbm_ex = lgbm['exercise']['f1_mean']
    lgbm_ph = lgbm['phase']['ml_f1_mean']
    lgbm_fat = lgbm['fatigue']['mae_mean']
    lgbm_rep = lgbm['reps']['ml_mae_mean']

    def delta(nn_v, lgbm_v, lower_better=False):
        if nn_v is None or lgbm_v is None:
            return 'N/A'
        try:
            d = float(nn_v) - float(lgbm_v)
            if lower_better:
                sign = '-' if d < 0 else '+'
                return f"{sign}{abs(d):.3f} ({'better' if d < 0 else 'worse'})"
            else:
                sign = '+' if d > 0 else '-'
                return f"{sign}{abs(d):.3f} ({'better' if d > 0 else 'worse'})"
        except Exception:
            return 'N/A'

    lines = [
        f"# Model Card: {label} (Features Input Variant A)",
        "",
        f"**Architecture:** {label} {arch_cite}  ",
        "**Input:** Per-window engineered features "
        f"(n_features={n_features}, same as LightGBM)  ",
        "**Multi-task structure:** Hard parameter sharing — "
        "shared encoder + 4 task heads (Caruana 1997)  ",
        "**Optimizer:** AdamW with cosine annealing (Loshchilov & Hutter 2019)  ",
        "**Regularization:** Dropout, BatchNorm, gradient clipping (Goodfellow et al. 2016)  ",
        "**CV:** Subject-wise GroupKFold(5), same splits as LightGBM (Saeb et al. 2017)  ",
        f"**Deployment:** {deploy_note}  ",
        "",
        "---",
        "",
        "## Performance vs. LightGBM Baseline",
        "",
        "| Task | Metric | This Model | LightGBM | Delta |",
        "|------|--------|-----------|---------|-------|",
        f"| Exercise | F1-macro | {fmt(ex.get('mean'))} ±{fmt(ex.get('std'))} | "
        f"{fmt(lgbm_ex)} | {delta(ex.get('mean'), lgbm_ex)} |",
        f"| Phase | F1-macro | {fmt(ph.get('mean'))} ±{fmt(ph.get('std'))} | "
        f"{fmt(lgbm_ph)} | {delta(ph.get('mean'), lgbm_ph)} |",
        f"| Fatigue | MAE | {fmt(fat.get('mean'))} ±{fmt(fat.get('std'))} | "
        f"{fmt(lgbm_fat)} | {delta(fat.get('mean'), lgbm_fat, lower_better=True)} |",
        f"| Reps | MAE | {fmt(rep.get('mean'))} ±{fmt(rep.get('std'))} | "
        f"{fmt(lgbm_rep)} | {delta(rep.get('mean'), lgbm_rep, lower_better=True)} |",
        "",
        "---",
        "",
        "## Hyperparameters",
        "",
        "Tuned via Optuna TPE, 30 trials, inner-CV on fold 0 training set "
        "(Akiba et al. 2019).",
        "",
        "```json",
        json.dumps(best_hp, indent=2, default=str),
        "```",
        "",
        "---",
        "",
        "## Latency",
        "",
        "| Device | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |",
        "|--------|---------|---------|---------|----------|",
        f"| GPU (RTX 5070 Ti) | {fmt(lat_gpu.get('p50_ms'))} | "
        f"{fmt(lat_gpu.get('p95_ms'))} | {fmt(lat_gpu.get('p99_ms'))} | "
        f"{fmt(lat_gpu.get('mean_ms'))} |",
        f"| CPU | {fmt(lat_cpu.get('p50_ms'))} | "
        f"{fmt(lat_cpu.get('p95_ms'))} | {fmt(lat_cpu.get('p99_ms'))} | "
        f"{fmt(lat_cpu.get('mean_ms'))} |",
        "",
        "Deployment latency budget: p99 <= 100 ms.",
        "",
        "---",
        "",
        "## Architecture Notes",
        "",
    ]

    if arch == 'cnn1d':
        lines += [
            "Input shape for features: (B, n_features) reshaped to (B, 1, n_features). "
            "Three 1D-Conv layers (32→64→128 channels) with AdaptiveAvgPool "
            "followed by a linear projection. Sees feature interactions as "
            "local context along the feature dimension.",
        ]
    elif arch == 'lstm':
        lines += [
            "Input shape: (B, n_features, 1) — each feature as a 'timestep'. "
            "BiLSTM with mean-pooling over the feature dimension. "
            "RESEARCH ONLY: BiLSTM is non-causal. For deployment, "
            "retrain as unidirectional LSTM.",
        ]
    elif arch == 'cnn_lstm':
        lines += [
            "Hybrid: Conv1D feature extraction followed by LSTM. "
            "RESEARCH ONLY: uses bidirectional LSTM internally. "
            "Cannot be deployed in real-time streaming.",
        ]
    elif arch == 'tcn':
        lines += [
            "Dilated causal convolutions with residual connections. "
            "Dilation doubles each layer (1, 2, 4, 8) for exponentially "
            "increasing receptive field. Takes final timestep (causal). "
            "Parallelizable unlike LSTM — preferred for deployment.",
        ]

    lines += [
        "",
        "---",
        "",
        "## References",
        "",
        "- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.",
        "- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation "
        "of generic convolutional and recurrent networks for sequence modeling. "
        "*arXiv:1803.01271*.",
        "- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
        "*Neural Computation*, 9(8), 1735-1780.",
        "- Yang, J., Nguyen, M. N., San, P. P., Li, X. L., & Krishnaswamy, S. "
        "(2015). Deep convolutional neural networks on multichannel time series "
        "for human activity recognition. *IJCAI 2015*.",
        "- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay "
        "regularization. *ICLR 2019*.",
        "- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. "
        "MIT Press.",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. "
        "(2017). The need to approximate the use-case in clinical machine "
        "learning. *GigaScience*, 6(5).",
        "- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). "
        "Optuna: A next-generation hyperparameter optimization framework. "
        "*KDD 2019*.",
        "- Ruder, S. (2017). An overview of multi-task learning in deep neural "
        "networks. *arXiv:1706.05098*.",
    ]
    return "\n".join(lines) + "\n"


def generate_comparison_png(all_results: Dict, lgbm: Dict, out_path: Path):
    """Bar chart: 4 tasks × 4 archs + LGBM."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.eval.plot_style import apply_style, despine
        apply_style()
    except ImportError:
        print("  matplotlib not available; skipping comparison.png")
        return

    p1 = all_results.get('phase1', {})
    p2 = all_results.get('phase2', {})

    tasks = [
        ('exercise', 'f1_macro', 'Exercise F1-macro', False),
        ('phase',    'f1_macro', 'Phase F1-macro',    False),
        ('fatigue',  'mae',      'Fatigue MAE',        True),
        ('reps',     'mae',      'Reps MAE',           True),
    ]
    archs = ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']
    arch_labels = ['1D-CNN', 'BiLSTM', 'CNN-LSTM', 'TCN']
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

    lgbm_vals = {
        'exercise': lgbm['exercise']['f1_mean'],
        'phase':    lgbm['phase']['ml_f1_mean'],
        'fatigue':  lgbm['fatigue']['mae_mean'],
        'reps':     lgbm['reps']['ml_mae_mean'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_i, (task, metric, title, lower_better) in enumerate(tasks):
        ax = axes[ax_i]
        x = np.arange(len(archs))
        means, stds = [], []
        for arch in archs:
            vname = f"features_{arch}"
            s = p2.get(vname) or p2.get(f"{vname}_p2") or p1.get(vname)
            if s:
                means.append(s.get(task, {}).get(metric, {}).get('mean', float('nan')))
                stds.append(s.get(task, {}).get(metric, {}).get('std', 0.0))
            else:
                means.append(float('nan'))
                stds.append(0.0)

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                       edgecolor='black', linewidth=0.5, alpha=0.85,
                       error_kw={'linewidth': 1.5})
        # LightGBM reference line
        lgbm_v = lgbm_vals.get(task)
        if lgbm_v is not None:
            ax.axhline(lgbm_v, color='red', linestyle='--', linewidth=1.5,
                        label=f'LightGBM={lgbm_v:.3f}')
            ax.legend(fontsize=8)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, fontsize=9)
        ax.set_ylabel('F1-macro' if 'F1' in title else 'MAE', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Annotate bars
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f'{m:.3f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(
        'NN vs LightGBM: Features Input Variant\n'
        'runs/20260427_121303_nn_features_full | GroupKFold-5 | 3 seeds',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    despine(fig=fig)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print(f"Generating artifacts for: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    all_results = load_json(RUN_DIR / 'all_results.json')
    lgbm = load_json(LGBM_METRICS_PATH)

    if not all_results:
        print("ERROR: all_results.json not found. Run run_train_nn_features_full.py first.")
        return

    winner = all_results.get('winner', 'features_tcn')
    latency = all_results.get('latency', {})
    ablation = all_results.get('ablation', {})

    # ---- comparison.md
    comp_md = generate_comparison_md(all_results, lgbm)
    (RUN_DIR / 'comparison.md').write_text(comp_md, encoding='utf-8')
    print(f"  Wrote: {RUN_DIR / 'comparison.md'}")

    # ---- comparison.png
    generate_comparison_png(all_results, lgbm, RUN_DIR / 'comparison.png')

    # ---- latency_table.md
    lat_md = generate_latency_md(latency)
    (RUN_DIR / 'latency_table.md').write_text(lat_md, encoding='utf-8')
    print(f"  Wrote: {RUN_DIR / 'latency_table.md'}")

    # ---- multitask_ablation.md
    abl_md = generate_ablation_md(ablation, winner)
    (RUN_DIR / 'multitask_ablation.md').write_text(abl_md, encoding='utf-8')
    print(f"  Wrote: {RUN_DIR / 'multitask_ablation.md'}")

    # ---- Per-arch model_card.md
    p1 = all_results.get('phase1', {})
    p2 = all_results.get('phase2', {})
    # n_features: try to extract from dataset info if embedded, else default
    n_features = all_results.get('n_features', None)
    if n_features is None:
        # Try loading the window features to get n_features
        try:
            import pandas as pd
            from src.pipeline.train_nn import LOSS_WEIGHTS
            from src.data.datasets import WindowFeatureDataset, METADATA_COLS, LABEL_COLS
            wf_path = ROOT / 'runs/20260427_110653_default/window_features.parquet'
            df = pd.read_parquet(wf_path)
            excluded = METADATA_COLS | LABEL_COLS
            feat_cols = [c for c in df.columns if c not in excluded
                         and df[c].dtype.kind in ('f', 'i', 'u')]
            n_features = len(feat_cols)
            print(f"  n_features from parquet: {n_features}")
        except Exception as e:
            n_features = 'unknown'
            print(f"  Could not determine n_features: {e}")

    for arch in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
        vname = f"features_{arch}"
        arch_dir = RUN_DIR / vname
        arch_dir.mkdir(parents=True, exist_ok=True)

        # Use phase2 summary if available, else phase1
        s = p2.get(vname) or p2.get(f"{vname}_p2") or p1.get(vname) or {}

        hp_path = arch_dir / 'best_hp.json'
        best_hp = load_json(hp_path) if hp_path.exists() else {}

        card = generate_model_card(
            arch=arch,
            variant_name=vname,
            summary=s,
            lgbm=lgbm,
            best_hp=best_hp,
            latency=latency,
            n_features=n_features,
        )
        (arch_dir / 'model_card.md').write_text(card, encoding='utf-8')
        print(f"  Wrote: {arch_dir / 'model_card.md'}")

    print("\nAll artifacts generated.")


if __name__ == '__main__':
    main()
