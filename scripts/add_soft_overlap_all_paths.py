"""Compute soft_overlap_reps_<W>s columns for both labeling paths.

Per recording, reads markers.json + metadata.json from dataset_aligned/ to
build rep intervals (Wang et al. 2026, J Appl Sci Eng 31:26031038, Eq. 2),
then computes the overlap-fraction sum at each timestep for window_s ∈ args
and writes a column to BOTH:

  data/labeled/<rec>/aligned_features.parquet     — for raw-variant training
  data/labeled/<rec>/window_features.parquet      — for features-variant

Column name convention:
  window_s = 2.0 → soft_overlap_reps        (legacy alias kept for backward compat)
  window_s = 1.0 → soft_overlap_reps_1s
  window_s = 5.0 → soft_overlap_reps_5s
  window_s = 2.5 → soft_overlap_reps_2_5s   (decimal escaped)

Idempotent: re-running on a parquet that already has the column overwrites
it.

Usage:
    python scripts/add_soft_overlap_all_paths.py
    python scripts/add_soft_overlap_all_paths.py --windows 1.0 2.0 5.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

REP_RE = re.compile(r"^Set:(\d+)_Rep:(\d+)$")


def build_rep_intervals(markers, kinect_sets):
    """Return {set_number: [(rep_start_unix, rep_end_unix), ...]}.

    rep_end is the next rep's start within the set, or the set's
    end_unix_time (from kinect_sets) for the last rep.
    """
    set_end = {int(s["set_number"]): float(s["end_unix_time"])
               for s in kinect_sets if "end_unix_time" in s}

    by_set: dict[int, list[tuple[int, float]]] = {}
    for m in markers:
        lab = m.get("label", "")
        match = REP_RE.match(lab)
        if not match:
            continue
        s_num = int(match.group(1))
        r_idx = int(match.group(2))
        unix = float(m["unix_time"])
        by_set.setdefault(s_num, []).append((r_idx, unix))

    rep_intervals: dict[int, list[tuple[float, float]]] = {}
    for s_num, lst in by_set.items():
        lst.sort()
        intervals = []
        for i in range(len(lst)):
            r_idx, t_start = lst[i]
            t_end = (lst[i + 1][1] if i + 1 < len(lst)
                     else set_end.get(s_num, t_start + 5.0))
            if t_end > t_start:
                intervals.append((t_start, t_end))
        if intervals:
            rep_intervals[s_num] = intervals
    return rep_intervals


def compute_soft_overlap(t_unix, set_nums, rep_intervals, window_s):
    n = len(t_unix)
    soft = np.zeros(n, dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i in range(n):
        s = set_nums[i]
        if s != s:           # NaN → not in active set
            continue
        s_int = int(round(s))
        if s_int not in rep_intervals:
            continue
        has[i] = True
        t_end = float(t_unix[i])
        t_start = t_end - window_s
        total = 0.0
        for r_start, r_end in rep_intervals[s_int]:
            ov_start = max(t_start, r_start)
            ov_end = min(t_end, r_end)
            if ov_end > ov_start:
                total += (ov_end - ov_start) / max(r_end - r_start, 1e-9)
        soft[i] = total
    return soft, has


def col_name_for(window_s: float) -> str:
    if abs(window_s - 2.0) < 1e-6:
        return "soft_overlap_reps"
    return f"soft_overlap_reps_{window_s:g}s".replace('.', '_')


def process_parquet(parquet_path: Path, rep_intervals, windows):
    """Add a soft_overlap_reps_<W>s column for each requested window_s. Also
    writes/updates has_rep_intervals (same for all window sizes)."""
    df = pd.read_parquet(parquet_path)
    if "t_unix" not in df.columns or "set_number" not in df.columns:
        return {"status": "missing_t_unix_or_set_number"}
    t_unix = df["t_unix"].to_numpy(dtype=np.float64)
    set_nums = df["set_number"].to_numpy(dtype=np.float64)
    has_any = None
    written = []
    for w in windows:
        soft, has = compute_soft_overlap(t_unix, set_nums, rep_intervals, w)
        col = col_name_for(w)
        df[col] = soft
        written.append((col, float(soft.min()), float(soft.max())))
        if has_any is None:
            has_any = has
    df["has_rep_intervals"] = has_any if has_any is not None \
        else np.zeros(len(df), dtype=bool)
    df.to_parquet(parquet_path, index=False)
    return {"status": "ok", "n_rows": len(df), "written": written}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", type=Path,
                     default=ROOT / "data" / "labeled")
    ap.add_argument("--dataset-dir", type=Path,
                     default=Path("C:/Users/skogl/Downloads/eirikgsk/"
                                    "biosignal_2/BioSignal_AI3/dataset_aligned"))
    ap.add_argument("--windows", type=float, nargs="+",
                     default=[1.0, 2.0, 5.0])
    args = ap.parse_args()

    if not args.labeled_dir.exists():
        print(f"FATAL: {args.labeled_dir} does not exist"); sys.exit(1)
    if not args.dataset_dir.exists():
        print(f"FATAL: {args.dataset_dir} does not exist (need markers.json "
              f"+ metadata.json per recording)"); sys.exit(1)

    rec_dirs = sorted(d for d in args.labeled_dir.iterdir()
                       if d.is_dir() and d.name.startswith("recording_"))
    print(f"Processing {len(rec_dirs)} recordings, "
          f"windows={args.windows}\n")

    summary = []
    for rd in rec_dirs:
        rec_id = rd.name
        src = args.dataset_dir / rec_id
        mk = src / "markers.json"
        md = src / "metadata.json"
        if not (mk.exists() and md.exists()):
            print(f"  {rec_id}: SKIP (missing markers/metadata under {src})")
            summary.append({"rec": rec_id, "status": "no_jsons"})
            continue

        markers = json.loads(mk.read_text())
        if isinstance(markers, dict):
            markers = markers.get("markers", [])
        metadata = json.loads(md.read_text())
        kinect_sets = metadata.get("kinect_sets", [])
        rep_intervals = build_rep_intervals(markers, kinect_sets)
        if not rep_intervals:
            print(f"  {rec_id}: SKIP (no per-rep markers)")
            summary.append({"rec": rec_id, "status": "no_rep_markers"})
            continue

        for parquet_name in ("aligned_features.parquet",
                               "window_features.parquet"):
            ppath = rd / parquet_name
            if not ppath.exists():
                print(f"  {rec_id}/{parquet_name}: SKIP (not present)")
                continue
            res = process_parquet(ppath, rep_intervals, args.windows)
            print(f"  {rec_id}/{parquet_name}: {res['status']}  "
                  f"rows={res.get('n_rows')}  "
                  f"cols={[c for c, _, _ in res.get('written', [])]}")
            summary.append({"rec": rec_id, "file": parquet_name, **res})

    print("\nDone.")


if __name__ == "__main__":
    main()
