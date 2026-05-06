"""Launch the 3 v9-winner phase-2 reruns with rest windows included.

Re-uses Optuna best HPs from v9 runs but flips ``active_only`` off so the
model sees the rest periods between sets.

Outputs to ``runs/optuna_clean_v10restwin-...``.
"""
from __future__ import annotations

import json
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

# (slug, src v9 dir name, out v10 dir name, arch, variant, window_s, tasks)
JOBS = [
    ('feat-mlp_w1s',  'optuna_clean_v9-w1s-multi-feat-mlp',
     'optuna_clean_v10restwin-w1s-multi-feat-mlp',
     'mlp',      'features', 1.0,
     ['exercise', 'phase', 'fatigue', 'reps']),
    ('feat-lstm_w1s', 'optuna_clean_v9-w1s-multi-feat-lstm',
     'optuna_clean_v10restwin-w1s-multi-feat-lstm',
     'lstm',     'features', 1.0,
     ['exercise', 'phase', 'fatigue', 'reps']),
    ('fatigue-tcn_w3s', 'optuna_clean_v9-w3s-fatigue-raw-tcn',
     'optuna_clean_v10restwin-w3s-fatigue-raw-tcn',
     'tcn_raw',  'raw',      3.0,
     ['fatigue']),
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"v10_restwin_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== v10-restwin: {len(JOBS)} jobs ===")
    print(f"Logs: {log_dir}\n")

    procs = []
    for slug, src_name, out_name, arch, variant, ws, tasks in JOBS:
        cmd = [
            sys.executable, "scripts/train_phase2_only.py",
            "--arch", arch, "--variant", variant,
            "--src-run-dir", str(ROOT / "runs" / src_name),
            "--out-run-dir", str(ROOT / "runs" / out_name),
            "--include-rest",
            "--phase2-epochs", "300", "--patience", "20",
            "--phase2-seeds", "42", "1337", "7",
            "--tasks", *tasks,
            "--reps-mode", "soft_overlap",
            "--window-s", str(ws),
            "--num-workers", "0",
            "--exclude-recordings", *EXCLUDE,
            "--labeled-root", str(LABELED_ROOT),
            "--splits", str(SPLITS),
        ]
        log = log_dir / f"{slug}.log"
        fh = open(log, "w", encoding="utf-8", buffering=1)
        fh.write(f"# {slug}\n# cmd: {' '.join(str(x) for x in cmd)}\n"
                 f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(ROOT))
        procs.append({'slug': slug, 'proc': proc, 'fh': fh, 'log': log,
                      'started': time.time()})
        print(f"  LAUNCHED {slug} pid={proc.pid} -> {log}")
        time.sleep(3)  # stagger so dataset materializes one at a time

    status_path = log_dir / 'status.json'

    def write_status(finished):
        status_path.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'running': [{'slug': p['slug'], 'pid': p['proc'].pid,
                          'elapsed_s': int(time.time() - p['started']),
                          'log': str(p['log'])}
                         for p in procs],
            'finished': finished,
        }, indent=2))

    finished = []
    while procs:
        time.sleep(30)
        new_procs = []
        for p in procs:
            rc = p['proc'].poll()
            if rc is None:
                new_procs.append(p)
            else:
                p['fh'].flush(); p['fh'].close()
                elapsed = int(time.time() - p['started'])
                status = 'ok' if rc == 0 else f'failed_rc{rc}'
                print(f"  DONE {p['slug']} rc={rc} elapsed={elapsed}s  "
                      f"status={status}")
                finished.append({'slug': p['slug'], 'status': status,
                                  'rc': rc, 'elapsed_s': elapsed,
                                  'log': str(p['log'])})
        procs = new_procs
        write_status(finished)

    print(f"\n=== ALL DONE ===")
    for f in finished:
        print(f"  {f['slug']}: {f['status']} ({f['elapsed_s']}s)")


if __name__ == '__main__':
    main()
