"""Overnight orchestrator for the full /compare-all pipeline.

Runs unattended:
- GPU verification (fail-fast if no CUDA available unless --allow-cpu)
- Smoke-test gate (~3 min) — halt if it fails
- Stage A: 4 NN architectures on all features
- Stage B: 4 NN architectures on top-K features (leakage-safe)
- Stage C: modality ablation on best-from-A
- Stage D: plots
- Stage E: master comparison + significance tests

Per-step behavior:
- Resume: skip any step where cv_summary.json already exists
- Retry: failed steps retried ONCE, then skipped (continues to next)
- Logging: everything tee'd to logs/overnight_<timestamp>.log
- Status: written to logs/overnight_status_<timestamp>.json after each step

Usage:
    python scripts/run_overnight.py --baseline-run runs/<lgbm_xgb_run> --top-k 30
    python scripts/run_overnight.py --baseline-run ... --top-k 30 --skip-smoke
    python scripts/run_overnight.py --baseline-run ... --top-k 30 --allow-cpu

In the morning, read:
    logs/overnight_<ts>.log              — full stdout/stderr
    logs/overnight_status_<ts>.json      — what completed, failed, was skipped
    runs/<ts>_overnight-comparison/comparison.md — the final report
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import json
import os
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-run', type=Path, required=True,
                    help='LightGBM + XGBoost baseline run dir from /train')
    p.add_argument('--top-k', type=int, default=30,
                    help='K for top-K feature ablation (Stage B)')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 7])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--archs', type=str, nargs='+',
                    default=['cnn1d', 'lstm', 'cnn_lstm', 'tcn'])
    p.add_argument('--ablation-arch', type=str, default=None,
                    help='Arch for modality ablation. Default: best from Stage A')
    p.add_argument('--skip-smoke', action='store_true')
    p.add_argument('--skip-ablation', action='store_true')
    p.add_argument('--skip-plots', action='store_true')
    p.add_argument('--allow-cpu', action='store_true',
                    help='Continue even if no GPU is available')
    p.add_argument('--logs-dir', type=Path, default=Path('logs'))
    p.add_argument('--runs-root', type=Path, default=Path('runs'))
    return p.parse_args()


class Tee:
    """Tees messages to both stdout and a log file with timestamps."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(log_path, 'a', buffering=1)

    def write(self, msg: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def check_gpu(tee: Tee, allow_cpu: bool) -> Dict:
    info = {'cuda_available': False, 'device_count': 0, 'device_names': []}
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        info['device_count'] = torch.cuda.device_count()
        if info['cuda_available']:
            info['device_names'] = [
                torch.cuda.get_device_name(i) for i in range(info['device_count'])
            ]
            tee.write(f"GPU OK: {info['device_count']} device(s): "
                       f"{info['device_names']}")
            try:
                x = torch.zeros(8, device='cuda')
                _ = x + 1
                del x
                torch.cuda.empty_cache()
                tee.write("GPU sanity allocation OK")
            except Exception as e:
                tee.write(f"GPU detected but allocation FAILED: {e}")
                if not allow_cpu:
                    raise SystemExit(2)
        else:
            tee.write("WARNING: No CUDA-capable GPU detected.")
            if not allow_cpu:
                tee.write("Halting (use --allow-cpu to continue on CPU).")
                raise SystemExit(2)
            tee.write("Continuing on CPU (--allow-cpu set).")
    except ImportError:
        tee.write("ERROR: PyTorch not installed.")
        raise SystemExit(2)
    return info


def run_step(tee: Tee, name: str, cmd: List[str],
             cwd: Optional[Path] = None,
             env_extra: Optional[Dict[str, str]] = None) -> Dict:
    """Run cmd. Retry once on failure. Returns status dict."""
    tee.write(f"=== STEP: {name} ===")
    tee.write(f"    cmd: {' '.join(cmd)}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    started = time.time()
    for attempt in (1, 2):
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, env=env, check=False,
                capture_output=True, text=True, timeout=None,
            )
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    tee.write(f"  [stdout] {line}")
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    tee.write(f"  [stderr] {line}")

            if proc.returncode == 0:
                elapsed = time.time() - started
                status = 'ok' if attempt == 1 else 'ok_retry'
                tee.write(f"    {status} ({elapsed:.0f}s)")
                return {'name': name, 'status': status,
                         'attempts': attempt, 'elapsed_s': elapsed}
            else:
                tee.write(f"    FAILED (returncode={proc.returncode}) "
                           f"on attempt {attempt}")
                if attempt == 1:
                    tee.write("    Retrying once...")
                    time.sleep(5)
        except Exception as e:
            tee.write(f"    EXCEPTION on attempt {attempt}: {e}")
            if attempt == 1:
                time.sleep(5)
                continue
            return {'name': name, 'status': 'crashed', 'error': str(e),
                     'elapsed_s': time.time() - started}

    elapsed = time.time() - started
    tee.write(f"    SKIPPED after retry failure ({elapsed:.0f}s)")
    return {'name': name, 'status': 'skipped_failed',
             'attempts': 2, 'elapsed_s': elapsed}


def latest_run_dir(runs_root: Path, slug: str) -> Optional[Path]:
    candidates = sorted(runs_root.glob(f"*_{slug}"), reverse=True)
    return candidates[0] if candidates else None


def already_complete(run_dir: Optional[Path]) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    return any(run_dir.glob('*/cv_summary.json'))


def save_status(status_path: Path, results: List[Dict], meta: Dict):
    status_path.write_text(json.dumps({
        'meta': meta,
        'results': results,
        'last_updated': datetime.now().isoformat(timespec='seconds'),
    }, indent=2, default=str))


def pick_best_arch(stage_a_runs: Dict[str, Path], tee: Tee) -> Optional[str]:
    """Auto-select Stage A arch with best mean score across tasks."""
    arch_scores: Dict[str, float] = {}
    for arch, run_dir in stage_a_runs.items():
        cv_files = list(run_dir.glob('*/cv_summary.json'))
        if not cv_files:
            continue
        s = json.loads(cv_files[0].read_text()).get('summary', {})
        scores = []
        for task, metric, sign in [
            ('exercise', 'f1_macro', +1),
            ('phase',    'f1_macro', +1),
            ('fatigue',  'mae',      -1),
            ('reps',     'mae',      -1),
        ]:
            val = s.get(task, {}).get(metric, {}).get('mean')
            if val is not None and val == val:
                scores.append(sign * float(val))
        if scores:
            arch_scores[arch] = sum(scores) / len(scores)

    if not arch_scores:
        return None
    best = max(arch_scores, key=arch_scores.get)
    ranked = sorted(arch_scores.items(), key=lambda x: -x[1])
    tee.write(f"[stage C] Architecture ranking: {ranked}")
    return best


def summarize(results: List[Dict]) -> str:
    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r['status']] = by_status.get(r['status'], 0) + 1
    lines = ["", f"Steps: {len(results)} total"]
    for status, count in sorted(by_status.items()):
        lines.append(f"  - {status}: {count}")
    failed = [r['name'] for r in results
              if r['status'] in ('skipped_failed', 'crashed')]
    if failed:
        lines.append("")
        lines.append("FAILED OR SKIPPED:")
        for n in failed:
            lines.append(f"  - {n}")
    return '\n'.join(lines)


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = args.logs_dir / f"overnight_{timestamp}.log"
    status_path = args.logs_dir / f"overnight_status_{timestamp}.json"
    tee = Tee(log_path)

    tee.write(f"=== OVERNIGHT RUN START — {timestamp} ===")
    tee.write(f"args: {vars(args)}")
    tee.write(f"log:    {log_path}")
    tee.write(f"status: {status_path}")

    results: List[Dict] = []
    meta = {'timestamp': timestamp,
            'args': {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()}}

    # GPU check
    gpu_info = check_gpu(tee, allow_cpu=args.allow_cpu)
    meta['gpu_info'] = gpu_info
    save_status(status_path, results, meta)

    # Verify baseline exists
    if not args.baseline_run.exists():
        tee.write(f"FATAL: --baseline-run {args.baseline_run} does not exist.")
        raise SystemExit(2)
    tee.write(f"Baseline run confirmed: {args.baseline_run}")

    # Smoke-test gate
    if not args.skip_smoke:
        smoke = run_step(tee, "smoke-test (cnn1d, 1 fold × 1 seed × 3 epochs)",
                          ['python', 'scripts/train_cnn1d.py', '--smoke-test'])
        results.append(smoke)
        save_status(status_path, results, meta)
        if smoke['status'] in ('skipped_failed', 'crashed'):
            tee.write("FATAL: Smoke-test failed. Halting overnight run.")
            raise SystemExit(2)

    # Stage A: all features × 4 archs
    tee.write("\n=== STAGE A: all features × architectures ===")
    stage_a_runs: Dict[str, Path] = {}
    for arch in args.archs:
        slug = f"nn-full-{arch}"
        existing = latest_run_dir(args.runs_root, slug)
        if already_complete(existing):
            tee.write(f"[resume] Stage A {arch}: reusing {existing}")
            stage_a_runs[arch] = existing
            continue

        cmd = [
            'python', f'scripts/train_{arch}.py',
            '--run-slug', slug,
            '--seeds', *[str(s) for s in args.seeds],
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
        ]
        res = run_step(tee, f"Stage A: {arch}", cmd)
        results.append(res)
        save_status(status_path, results, meta)

        new_run = latest_run_dir(args.runs_root, slug)
        if already_complete(new_run):
            stage_a_runs[arch] = new_run
        else:
            tee.write(f"[stage A] {arch}: no completed run; "
                       f"will be omitted from comparison")

    # Stage B: top-K × 4 archs (leakage-safe)
    tee.write(f"\n=== STAGE B: top-{args.top_k} features × architectures ===")
    stage_b_runs: Dict[str, Path] = {}
    for arch in args.archs:
        slug = f"nn-top{args.top_k}-{arch}"
        existing = latest_run_dir(args.runs_root, slug)
        if already_complete(existing):
            tee.write(f"[resume] Stage B {arch}: reusing {existing}")
            stage_b_runs[arch] = existing
            continue

        cmd = [
            'python', 'scripts/train_with_top_k.py',
            '--arch', arch,
            '--top-k', str(args.top_k),
            '--leakage-safe',
            '--run-slug', slug,
            '--seeds', *[str(s) for s in args.seeds],
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
        ]
        res = run_step(tee, f"Stage B: {arch} top-{args.top_k}", cmd)
        results.append(res)
        save_status(status_path, results, meta)

        new_run = latest_run_dir(args.runs_root, slug)
        if already_complete(new_run):
            stage_b_runs[arch] = new_run

    # Stage C: modality ablation
    ablation_runs: List[Path] = []
    if not args.skip_ablation:
        tee.write("\n=== STAGE C: modality ablation ===")
        best_arch = args.ablation_arch
        if best_arch is None and stage_a_runs:
            best_arch = pick_best_arch(stage_a_runs, tee)
        if best_arch is None:
            tee.write("[stage C] No Stage A runs; skipping ablation.")
        else:
            tee.write(f"[stage C] Ablation arch: {best_arch}")
            slug = f"ablate-{best_arch}"
            existing = latest_run_dir(args.runs_root, slug)
            if already_complete(existing):
                tee.write(f"[resume] Stage C: reusing {existing}")
                ablation_runs = sorted(existing.glob('no_*'))
            else:
                cmd = [
                    'python', 'scripts/ablate_modalities.py',
                    '--arch', best_arch,
                    '--run-slug', slug,
                    '--seeds', *[str(s) for s in args.seeds],
                    '--epochs', str(args.epochs),
                    '--batch-size', str(args.batch_size),
                ]
                res = run_step(tee, f"Stage C: ablation ({best_arch})", cmd)
                results.append(res)
                save_status(status_path, results, meta)

                new_run = latest_run_dir(args.runs_root, slug)
                if new_run and new_run.exists():
                    ablation_runs = sorted(new_run.glob('no_*'))

    # Stage D: plots
    if not args.skip_plots:
        tee.write("\n=== STAGE D: plots ===")
        all_plot_dirs = (list(stage_a_runs.values())
                          + list(stage_b_runs.values())
                          + ablation_runs)
        if all_plot_dirs:
            cmd = ['python', 'scripts/generate_plots.py',
                   '--runs', *[str(d) for d in all_plot_dirs]]
            res = run_step(tee, "Stage D: plots", cmd)
            results.append(res)
            save_status(status_path, results, meta)
        else:
            tee.write("[stage D] No completed runs; skipping plots.")

    # Stage E: master comparison
    tee.write("\n=== STAGE E: master comparison ===")
    cmp_args = [
        'python', 'scripts/compare_all.py',
        '--baseline-run', str(args.baseline_run),
        '--top-k', str(args.top_k),
        '--output-slug', 'overnight-comparison',
    ]
    if stage_a_runs:
        cmp_args += ['--full-feature-runs',
                      *[str(p) for p in stage_a_runs.values()]]
    if stage_b_runs:
        cmp_args += ['--topk-runs',
                      *[str(p) for p in stage_b_runs.values()]]
    if ablation_runs:
        cmp_args += ['--ablation-runs', *[str(p) for p in ablation_runs]]

    res = run_step(tee, "Stage E: master comparison", cmp_args)
    results.append(res)
    save_status(status_path, results, meta)

    # Done
    tee.write("\n=== OVERNIGHT RUN COMPLETE ===")
    tee.write(summarize(results))
    save_status(status_path, results, meta)
    tee.close()


if __name__ == '__main__':
    main()
