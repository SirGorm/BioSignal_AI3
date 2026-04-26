"""Aggregate results from the 4 architecture runs into a comparison table.

Run AFTER train_cnn1d, train_lstm, train_cnn_lstm, and train_tcn complete.

Reads cv_summary.json from each architecture's run dir, builds a side-by-side
table, and writes runs/<comparison_slug>/comparison.md and SUMMARY.md.

Run:
    python scripts/compare_architectures.py \\
        --runs runs/20260427_120000_nn_features_cnn1d \\
               runs/20260427_130000_nn_features_lstm \\
               runs/20260427_140000_nn_features_cnn_lstm \\
               runs/20260427_150000_nn_features_tcn \\
        --baseline-run runs/20260426_103000_lgbm_baseline \\
        --output-slug nn-features-comparison
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def load_cv_summary(run_dir: Path):
    """Find the architecture's cv_summary.json under <run_dir>/<arch>/."""
    candidates = list(run_dir.glob('*/cv_summary.json'))
    if not candidates:
        raise FileNotFoundError(f"No cv_summary.json under {run_dir}")
    return json.loads(candidates[0].read_text())


def load_lgbm_metrics(baseline_run: Optional[Path]):
    """Load LightGBM baseline metrics if provided."""
    if baseline_run is None:
        return None
    metrics_path = baseline_run / 'metrics.json'
    if not metrics_path.exists():
        print(f"[compare] WARNING: {metrics_path} not found — baseline column omitted.")
        return None
    return json.loads(metrics_path.read_text())


def fmt(mean_std: Optional[Dict]) -> str:
    if mean_std is None or mean_std.get('n', 0) == 0:
        return '—'
    m, s = mean_std['mean'], mean_std['std']
    return f"{m:.3f} ± {s:.3f}"


def fmt_baseline(metrics: Optional[Dict], task: str, key: str) -> str:
    if metrics is None:
        return '—'
    v = metrics.get(task, {}).get(key)
    if v is None:
        return '—'
    return f"{float(v):.3f}"


def build_comparison_md(arch_results: Dict[str, Dict],
                          lgbm_metrics: Optional[Dict]) -> str:
    lines = [
        "# Architecture comparison — Phase 1 (feature input)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "All 4 architecture scripts in Phase 1 trained the same multi-task MLP "
        "backbone since per-window feature input has degenerate time dimension. "
        "Architecture labels are preserved for direct comparison vs the raw-input "
        "Phase 2 variants.",
        "",
        "## Results table",
        "",
        "| Task | Metric | LightGBM | 1D-CNN | LSTM | CNN-LSTM | TCN |",
        "|------|--------|----------|--------|------|----------|-----|",
    ]

    def row(task: str, metric_key: str, label: str):
        b = fmt_baseline(lgbm_metrics, task, metric_key)
        cells = []
        for a in ['cnn1d', 'lstm', 'cnn_lstm', 'tcn']:
            r = arch_results.get(a)
            if r is None:
                cells.append('—')
            else:
                cells.append(fmt(r['summary'].get(task, {}).get(metric_key)))
        return f"| {task} | {label} | {b} | " + " | ".join(cells) + " |"

    lines += [
        row('exercise', 'f1_macro', 'F1-macro'),
        row('exercise', 'balanced_accuracy', 'Bal-Acc'),
        row('phase',    'f1_macro', 'F1-macro'),
        row('phase',    'balanced_accuracy', 'Bal-Acc'),
        row('fatigue',  'mae', 'MAE (RPE)'),
        row('fatigue',  'pearson_r', 'Pearson r'),
        row('reps',     'mae', 'MAE (count)'),
    ]

    lines += [
        "",
        "## Notes",
        "- All NN cells show mean ± std across CV folds × seeds.",
        "- LightGBM cell shows the baseline metric from its model_card.",
        "- Negative finding (NN does not beat LightGBM) is expected and "
          "publishable in a low-data regime — see Saeb et al. 2017 on the "
          "leakage-vs-signal trade-off.",
        "",
        "## References",
        "- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75.",
        "- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). "
          "The need to approximate the use-case in clinical machine learning. "
          "*GigaScience*, 6(5), gix019.",
        "- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
          "*Neural Computation*, 9(8), 1735–1780.",
        "- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of "
          "generic convolutional and recurrent networks for sequence modeling. "
          "*arXiv preprint arXiv:1803.01271*.",
        "- Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM "
          "recurrent neural networks for multimodal wearable activity recognition. "
          "*Sensors*, 16(1), 115.",
        "- Yang, J., Nguyen, M. N., San, P. P., Li, X. L., & Krishnaswamy, S. (2015). "
          "Deep convolutional neural networks on multichannel time series for "
          "human activity recognition. *Proceedings of IJCAI*, 25, 3995–4001.",
    ]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, nargs='+', required=True,
                    help='Architecture run directories (one per arch)')
    p.add_argument('--baseline-run', type=Path, default=None,
                    help='LightGBM baseline run directory')
    p.add_argument('--output-slug', type=str, default='nn-features-comparison')
    args = p.parse_args()

    arch_results: Dict[str, Dict] = {}
    for run_dir in args.runs:
        summary = load_cv_summary(run_dir)
        arch = summary['arch']
        arch_results[arch] = summary
        print(f"[compare] Loaded {arch} from {run_dir}")

    lgbm_metrics = load_lgbm_metrics(args.baseline_run)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('runs') / f"{timestamp}_{args.output_slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md = build_comparison_md(arch_results, lgbm_metrics)
    (out_dir / 'comparison.md').write_text(md)
    print(f"[compare] Wrote {out_dir / 'comparison.md'}")

    # Also save raw aggregated json
    (out_dir / 'comparison_raw.json').write_text(json.dumps({
        'archs': arch_results,
        'lgbm': lgbm_metrics,
    }, indent=2, default=str))
    print(f"[compare] Done.")


if __name__ == '__main__':
    main()
