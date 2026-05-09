"""Phase-2-only re-evaluation of v17 mlp + lstm winners on the regenerated
window_features.parquet (now contains 4 new ACC amplitude descriptors:
acc_lscore, acc_mfl, acc_msr, acc_wamp).

Reuses HPs from each v17 source run (best_hps.json), runs phase 2 only:
3 seeds × 10 LOSO folds × 150 epochs.

Throttles to MAX_PAR concurrent jobs.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_PAR = 3

# (slug,                          src_run,                          arch,   variant,  window_s, tasks_or_None)
JOBS = [
    ("v18new-multi-mlp-w1s",       "v17multi-mlp-w1s",              "mlp",  "features", 1.0, None),
    ("v18new-multi-mlp-w2s",       "v17multi-mlp-w2s",              "mlp",  "features", 2.0, None),
    ("v18new-multi-mlp-w5s",       "v17multi-mlp-w5s",              "mlp",  "features", 5.0, None),
    ("v18new-multi-lstm-w1s",      "v17multi-lstm-w1s",             "lstm", "features", 1.0, None),
    ("v18new-multi-lstm-w2s",      "v17multi-lstm-w2s",             "lstm", "features", 2.0, None),
    ("v18new-multi-lstm-w5s",      "v17multi-lstm-w5s",             "lstm", "features", 5.0, None),
    ("v18new-single-exercise-mlp-w1s",  "v17single-exercise-mlp-w1s",  "mlp",  "features", 1.0, ["exercise"]),
    ("v18new-single-phase-mlp-w1s",     "v17single-phase-mlp-w1s",     "mlp",  "features", 1.0, ["phase"]),
    ("v18new-single-fatigue-lstm-w2s",  "v17single-fatigue-lstm-w2s",  "lstm", "features", 2.0, ["fatigue"]),
    ("v18new-single-reps-lstm-w2s",     "v17single-reps-lstm-w2s",     "lstm", "features", 2.0, ["reps"]),
]


def already_done(out_dir: Path, arch: str) -> bool:
    cv = out_dir / "phase2" / arch / "cv_summary.json"
    return cv.exists()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    queue: list[dict] = []
    skipped: list[str] = []
    for slug, src, arch, variant, w, tasks in JOBS:
        out_dir = ROOT / "runs" / slug
        if already_done(out_dir, arch):
            skipped.append(slug)
            continue
        src_dir = ROOT / "runs" / src
        if not (src_dir / "best_hps.json").exists():
            print(f"[skip] {slug}: source {src} missing best_hps.json")
            continue
        cmd = [sys.executable, "-m", "scripts.train_phase2_only",
               "--arch", arch, "--variant", variant,
               "--src-run-dir", str(src_dir),
               "--out-run-dir", str(out_dir),
               "--window-s", str(w)]
        if tasks:
            cmd += ["--tasks", *tasks]
        log = log_dir / f"{slug}_{ts}.log"
        queue.append({"slug": slug, "cmd": cmd, "log": log,
                       "out_dir": out_dir, "arch": arch})

    print(f"=== v18new phase2-only sweep (new ACC features) ===")
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
                                    "out_dir": j["out_dir"], "arch": j["arch"]}
            print(f"[LAUNCH] {j['slug']}  pid={proc.pid}", flush=True)

        time.sleep(15)

        done_slugs = []
        for slug, r in running.items():
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["fh"].flush(); r["fh"].close()
            elapsed = int(time.time() - r["started"])
            cv = r["out_dir"] / "phase2" / r["arch"] / "cv_summary.json"
            ok = cv.exists() and rc == 0
            tag = "DONE" if ok else f"FAIL(rc={rc})"
            print(f"[{tag}] {slug}  elapsed={elapsed}s  cv_summary={'YES' if cv.exists() else 'no'}",
                  flush=True)
            finished.append({"slug": slug, "status": "ok" if ok else "fail",
                              "rc": rc, "elapsed_s": elapsed})
            done_slugs.append(slug)
        for s in done_slugs:
            running.pop(s)

    print(f"\n=== sweep done ===")
    ok = sum(1 for f in finished if f["status"] == "ok")
    fail = [f for f in finished if f["status"] != "ok"]
    print(f"  ok: {ok}  failed: {len(fail)}  cached: {len(skipped)}")
    if fail:
        for f in fail:
            print(f"    FAIL: {f['slug']} (rc={f['rc']}, {f['elapsed_s']}s)")


if __name__ == "__main__":
    main()
