"""V11: Phase-2 refit of ALL 40 v9 configs (8 archs × 5 windows) with rest
windows included. Re-uses Optuna best HPs from v9 runs.

Outputs to ``runs/optuna_clean_v11restwin-w{N}s-{slug}/``. Skips jobs where
``phase2/.../cv_summary.json`` already exists (cache-aware).

The 3 configs we already trained as v10 (feat-mlp@1s, feat-lstm@1s,
fatigue-tcn@3s) are reused by copying v10 → v11 before launch.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELED_ROOT = ROOT / "data" / "labeled_clean"
SPLITS = ROOT / "configs" / "splits_clean_loso.csv"
EXCLUDE = ["recording_004", "recording_005", "recording_007",
           "recording_008", "recording_009"]

WINDOWS = [1.0, 2.0, 3.0, 4.0, 5.0]

# (slug-suffix, arch, variant, tasks)
MULTI = [
    ('multi-feat-mlp',     'mlp',          'features', ['exercise','phase','fatigue','reps']),
    ('multi-feat-lstm',    'lstm',         'features', ['exercise','phase','fatigue','reps']),
    ('multi-raw-cnn1d',    'cnn1d_raw',    'raw',      ['exercise','phase','fatigue','reps']),
    ('multi-raw-lstm',     'lstm_raw',     'raw',      ['exercise','phase','fatigue','reps']),
    ('multi-raw-cnn_lstm', 'cnn_lstm_raw', 'raw',      ['exercise','phase','fatigue','reps']),
    ('multi-raw-tcn',      'tcn_raw',      'raw',      ['exercise','phase','fatigue','reps']),
]
FATIGUE = [
    ('fatigue-raw-tcn',  'tcn_raw',  'raw', ['fatigue']),
    ('fatigue-raw-lstm', 'lstm_raw', 'raw', ['fatigue']),
]


def _wlabel(window_s: float) -> str:
    return f"{window_s:g}s".replace('.', '_')


def cv_summary_done(run_dir: Path) -> bool:
    p2 = run_dir / 'phase2'
    if not p2.exists():
        return False
    return next(iter(p2.rglob('cv_summary.json')), None) is not None


def reuse_v10_results():
    """Copy already-done v10 phase 2 results into v11 dirs to avoid recomputing."""
    pairs = [
        ('optuna_clean_v10restwin-w1s-multi-feat-mlp',
         'optuna_clean_v11restwin-w1s-multi-feat-mlp'),
        ('optuna_clean_v10restwin-w1s-multi-feat-lstm',
         'optuna_clean_v11restwin-w1s-multi-feat-lstm'),
        ('optuna_clean_v10restwin-w3s-fatigue-raw-tcn',
         'optuna_clean_v11restwin-w3s-fatigue-raw-tcn'),
    ]
    for src, dst in pairs:
        src_dir = ROOT / 'runs' / src
        dst_dir = ROOT / 'runs' / dst
        if not src_dir.exists():
            continue
        if cv_summary_done(dst_dir):
            continue
        if dst_dir.exists():
            continue  # don't clobber a partial v11 run
        print(f"  Reusing {src} → {dst}")
        shutil.copytree(src_dir, dst_dir)


def build_jobs():
    """Yield job specs for the 40 configurations."""
    jobs = []
    for window_s in WINDOWS:
        wl = _wlabel(window_s)
        for slug_suffix, arch, variant, tasks in MULTI + FATIGUE:
            slug = f"w{wl}-{slug_suffix}"
            src_run = ROOT / 'runs' / f"optuna_clean_v9-{slug}"
            out_run = ROOT / 'runs' / f"optuna_clean_v11restwin-{slug}"
            jobs.append({
                'slug': slug,
                'arch': arch, 'variant': variant, 'window_s': window_s,
                'tasks': tasks,
                'src_run': src_run, 'out_run': out_run,
            })
    return jobs


def make_cmd(job: dict) -> list[str]:
    return [
        sys.executable, "scripts/train_phase2_only.py",
        "--arch", job['arch'], "--variant", job['variant'],
        "--src-run-dir", str(job['src_run']),
        "--out-run-dir", str(job['out_run']),
        "--include-rest",
        "--phase2-epochs", "300", "--patience", "20",
        "--phase2-seeds", "42", "1337", "7",
        "--tasks", *job['tasks'],
        "--reps-mode", "soft_overlap",
        "--window-s", str(job['window_s']),
        "--num-workers", "0",
        "--exclude-recordings", *EXCLUDE,
        "--labeled-root", str(LABELED_ROOT),
        "--splits", str(SPLITS),
    ]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-gpu-jobs', type=int, default=4)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"v11_restwin_full_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / 'status.json'

    print(f"=== V11-restwin: ALL 40 configs (8 archs × 5 windows) ===")
    print(f"max GPU jobs: {args.max_gpu_jobs}")
    print(f"Logs: {log_dir}\n")

    print("Reusing v10 results where available...")
    reuse_v10_results()

    jobs = build_jobs()

    # Check for src run dirs that don't exist (won't have best_hps.json)
    missing = [j for j in jobs if not (j['src_run'] / 'best_hps.json').exists()]
    if missing:
        print(f"\nWARNING: {len(missing)} v9 src dirs missing best_hps.json:")
        for j in missing:
            print(f"  {j['src_run'].name}")
        jobs = [j for j in jobs if j not in missing]

    pending = []
    skipped = []
    for j in jobs:
        if cv_summary_done(j['out_run']):
            skipped.append(j['slug'])
        else:
            pending.append(j)
    print(f"\nTotal jobs: {len(jobs)}  pending: {len(pending)}  "
          f"skipped (cached): {len(skipped)}")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    print()

    running: dict[str, dict] = {}
    finished: list[dict] = []

    def write_status():
        status_path.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'running': [{'slug': s, 'pid': r['pid'],
                          'elapsed_s': int(time.time() - r['started']),
                          'log': str(r['log'])}
                         for s, r in running.items()],
            'pending': [j['slug'] for j in pending],
            'finished': finished,
            'skipped_cached': skipped,
        }, indent=2))

    while pending or running:
        # Launch new jobs up to GPU cap
        while pending and len(running) < args.max_gpu_jobs:
            j = pending.pop(0)
            log_path = log_dir / f"{j['slug']}.log"
            fh = open(log_path, "w", encoding="utf-8", buffering=1)
            fh.write(f"# {j['slug']}\n# cmd: {' '.join(str(x) for x in make_cmd(j))}\n"
                     f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            proc = subprocess.Popen(make_cmd(j), stdout=fh,
                                     stderr=subprocess.STDOUT, cwd=str(ROOT))
            running[j['slug']] = {
                'pid': proc.pid, 'proc': proc, 'fh': fh, 'log': log_path,
                'started': time.time(),
            }
            print(f"[{j['slug']}] LAUNCHED pid={proc.pid}")
        write_status()

        time.sleep(30)

        # Reap finished
        done = []
        for slug, r in running.items():
            rc = r['proc'].poll()
            if rc is not None:
                r['fh'].flush(); r['fh'].close()
                elapsed = int(time.time() - r['started'])
                status = 'ok' if rc == 0 else f'failed_rc{rc}'
                print(f"[{slug}] DONE rc={rc} elapsed={elapsed}s  status={status}")
                finished.append({'slug': slug, 'status': status, 'rc': rc,
                                  'elapsed_s': elapsed, 'log': str(r['log'])})
                done.append(slug)
        for s in done:
            running.pop(s)
        write_status()

    print(f"\n=== V11-restwin DONE ===")
    ok = [f for f in finished if f['status'] == 'ok']
    fail = [f for f in finished if f['status'] != 'ok']
    print(f"  ok: {len(ok)}  failed: {len(fail)}  cached: {len(skipped)}")
    for f in fail:
        print(f"    FAILED  {f['slug']} rc={f['rc']} log={f['log']}")
    print(f"\nFull status: {status_path}")


if __name__ == '__main__':
    main()
