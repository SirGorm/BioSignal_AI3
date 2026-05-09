"""Launch 1-trial / 1-seed phase1+phase2 for every multi-task arch at w=1s.

Throttles to MAX_PAR concurrent jobs to keep GPU memory in check.
Outputs: runs/v18-1t1s-{arch}-{variant}-w1s/phase2/.../cv_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_PAR = 3

JOBS = [
    # (arch, variant, registry-key for cv_summary subdir)
    ("mlp",           "features", "mlp"),
    ("cnn1d",         "features", "cnn1d"),
    ("lstm",          "features", "lstm"),
    ("cnn_lstm",      "features", "cnn_lstm"),
    ("tcn",           "features", "tcn"),
    ("cnn1d_raw",     "raw",      "cnn1d_raw"),
    ("lstm_raw",      "raw",      "lstm_raw"),
    ("cnn_lstm_raw",  "raw",      "cnn_lstm_raw"),
    ("tcn_raw",       "raw",      "tcn_raw"),
]


def slug_for(arch: str, variant: str) -> str:
    tag = "feat" if variant == "features" else "raw"
    return f"v18-1t1s-{arch}-{tag}-w1s"


def already_done(slug: str, arch_key: str) -> bool:
    p = ROOT / "runs" / slug / "phase2" / arch_key / "cv_summary.json"
    return p.exists()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    queue = []
    skipped = []
    for arch, variant, key in JOBS:
        slug = slug_for(arch, variant)
        if already_done(slug, key):
            skipped.append(slug)
            continue
        run_dir = ROOT / "runs" / slug
        cmd = [sys.executable, "-m", "scripts.train_optuna",
               "--variant", variant,
               "--runs-root", str(ROOT / "runs"),
               "--run-dir", str(run_dir),
               "--arch", arch,
               "--window-s", "1.0",
               "--n-trials", "1",
               "--phase2-seeds", "42"]
        log = log_dir / f"{slug}_{ts}.log"
        queue.append({"slug": slug, "cmd": cmd, "log": log,
                       "arch_key": key})

    print(f"=== v18 1-trial/1-seed multi-task sweep ===")
    print(f"Total jobs: {len(JOBS)}  pending: {len(queue)}  cached: {len(skipped)}")
    for s in skipped:
        print(f"  CACHED  {s}")
    for j in queue:
        print(f"  PENDING {j['slug']}")
    print(f"Max parallel: {MAX_PAR}\n")

    running: dict[str, dict] = {}
    finished: list[dict] = []

    while queue or running:
        while queue and len(running) < MAX_PAR:
            j = queue.pop(0)
            fh = open(j["log"], "w", encoding="utf-8", buffering=1)
            fh.write(f"# {j['slug']}\n# cmd: {' '.join(str(x) for x in j['cmd'])}\n"
                     f"# started: {datetime.now().isoformat(timespec='seconds')}\n\n")
            fh.flush()
            proc = subprocess.Popen(j["cmd"], stdout=fh,
                                     stderr=subprocess.STDOUT, cwd=str(ROOT))
            running[j["slug"]] = {"proc": proc, "fh": fh,
                                    "log": j["log"], "started": time.time(),
                                    "arch_key": j["arch_key"]}
            print(f"[LAUNCH] {j['slug']}  pid={proc.pid}", flush=True)

        time.sleep(15)

        done_slugs = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["fh"].flush(); r["fh"].close()
            elapsed = int(time.time() - r["started"])
            cv = (ROOT / "runs" / slug / "phase2" / r["arch_key"]
                  / "cv_summary.json")
            ok = cv.exists() and rc == 0
            tag = "DONE" if ok else f"FAIL(rc={rc})"
            print(f"[{tag}] {slug}  elapsed={elapsed}s  cv_summary={'YES' if cv.exists() else 'no'}",
                  flush=True)
            finished.append({"slug": slug, "status": "ok" if ok else "fail",
                              "rc": rc, "elapsed_s": elapsed})
            done_slugs.append(slug)
        for s in done_slugs:
            running.pop(s)

    print(f"\n=== sweep done ===  {len(finished)} runs")
    ok = sum(1 for f in finished if f["status"] == "ok")
    fail = [f for f in finished if f["status"] != "ok"]
    print(f"  ok: {ok}  failed: {len(fail)}  cached: {len(skipped)}")
    if fail:
        for f in fail:
            print(f"    FAIL: {f['slug']} (rc={f['rc']}, {f['elapsed_s']}s)")


if __name__ == "__main__":
    main()
