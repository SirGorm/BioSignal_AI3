"""Add `soft_overlap_reps` column to all data/labeled_clean/*/aligned_features.parquet.

Implements Wang et al. 2026 (J Appl Sci Eng 31:26031038) Eq. 2:
    y_cnt(window i) = Σ_k |[s_k, e_k) ∩ [t_i, t_i+L)| / (e_k - s_k)

For each row in the parquet (representing one 10 ms timestep on the
labeled grid), we treat the window as the 2 s span ENDING at that
timestep, i.e. [t-2.0, t]. Inside each set we sum the overlap fractions
with every rep interval that overlaps that window.

Rep intervals are derived from markers.json + metadata.json:
  - rep_k_start = unix_time of the `Set:N_Rep:K` marker
  - rep_k_end   = next rep's start (within the set), or set's end_unix_time
                  (from metadata.kinect_sets) for the last rep in the set

Adds two columns to the parquet:
  soft_overlap_reps  — float, count of reps overlapping the 2 s window ending at this row
  has_rep_intervals  — bool, True only for rows in sets that have rep markers

Usage:
    python scripts/add_soft_overlap_reps.py
    python scripts/add_soft_overlap_reps.py --window-s 2.0 --labeled-dir data/labeled_clean
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


def build_rep_intervals(markers: list[dict], kinect_sets: list[dict]) -> dict[int, list[tuple[float, float]]]:
    """Return {set_number: [(rep_start_unix, rep_end_unix), ...]}.

    rep_end is the next rep's start within the set, or the set's end_unix_time
    for the last rep.
    """
    set_ends = {s["set_number"]: float(s["end_unix_time"]) for s in kinect_sets}
    by_set: dict[int, list[tuple[int, float]]] = {}
    for e in markers:
        m = REP_RE.match(e.get("label", ""))
        if not m:
            continue
        sn = int(m.group(1))
        rn = int(m.group(2))
        by_set.setdefault(sn, []).append((rn, float(e["unix_time"])))

    out: dict[int, list[tuple[float, float]]] = {}
    for sn, reps in by_set.items():
        reps.sort()  # by rep_num
        intervals: list[tuple[float, float]] = []
        for i, (rn, t_start) in enumerate(reps):
            if i + 1 < len(reps):
                t_end = reps[i + 1][1]
            else:
                t_end = set_ends.get(sn, t_start)
                if t_end <= t_start:
                    # malformed metadata — fall back to inter-rep median
                    if len(reps) >= 2:
                        med_dur = float(np.median([reps[k + 1][1] - reps[k][1]
                                                     for k in range(len(reps) - 1)]))
                        t_end = t_start + med_dur
                    else:
                        t_end = t_start + 2.0  # arbitrary 2 s fallback
            intervals.append((t_start, t_end))
        out[sn] = intervals
    return out


def compute_soft_overlap(t_unix: np.ndarray, set_nums: np.ndarray,
                          rep_intervals: dict[int, list[tuple[float, float]]],
                          window_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (soft_overlap_reps, has_rep_intervals) arrays of length len(t_unix).

    For row i, computes the sum over reps within set_nums[i] of the fraction
    of (rep_start, rep_end) that lies inside [t_unix[i] - window_s, t_unix[i]].
    """
    n = len(t_unix)
    soft = np.zeros(n, dtype=np.float32)
    has = np.zeros(n, dtype=bool)

    # Precompute set->rep intervals as numpy arrays for fast access.
    set_rep_starts: dict[int, np.ndarray] = {
        s: np.asarray([t for t, _ in iv], dtype=np.float64)
        for s, iv in rep_intervals.items()
    }
    set_rep_ends: dict[int, np.ndarray] = {
        s: np.asarray([te for _, te in iv], dtype=np.float64)
        for s, iv in rep_intervals.items()
    }
    set_rep_durs: dict[int, np.ndarray] = {
        s: np.maximum(set_rep_ends[s] - set_rep_starts[s], 1e-6)
        for s in rep_intervals
    }

    for i in range(n):
        sn_f = set_nums[i]
        if not np.isfinite(sn_f):
            continue
        sn = int(sn_f)
        if sn not in set_rep_starts:
            continue
        has[i] = True
        t_end = float(t_unix[i])
        t_start = t_end - window_s
        rs = set_rep_starts[sn]
        re_ = set_rep_ends[sn]
        rd = set_rep_durs[sn]
        # vectorized overlap: max(0, min(t_end, re) - max(t_start, rs))
        ovl = np.maximum(0.0, np.minimum(t_end, re_) - np.maximum(t_start, rs))
        soft[i] = float(np.sum(ovl / rd))

    return soft, has


def process_recording(rec_id: str, labeled_dir: Path, dataset_dir: Path,
                       window_s: float) -> dict:
    rec_labeled = labeled_dir / rec_id
    parquet_path = rec_labeled / "aligned_features.parquet"
    if not parquet_path.exists():
        return {"rec": rec_id, "status": "no_parquet"}

    src_dir = dataset_dir / rec_id
    mk_p = src_dir / "markers.json"
    md_p = src_dir / "metadata.json"
    if not (mk_p.exists() and md_p.exists()):
        return {"rec": rec_id, "status": "no_source_jsons"}

    markers = json.loads(mk_p.read_text())
    if isinstance(markers, dict):
        markers = markers.get("markers", [])
    metadata = json.loads(md_p.read_text())
    kinect_sets = metadata.get("kinect_sets", [])

    rep_intervals = build_rep_intervals(markers, kinect_sets)
    if not rep_intervals:
        return {"rec": rec_id, "status": "no_rep_markers (protocol without per-rep markers)"}

    df = pd.read_parquet(parquet_path)
    t_unix = df["t_unix"].to_numpy(dtype=np.float64)
    set_nums = df["set_number"].to_numpy(dtype=np.float64)

    soft, has = compute_soft_overlap(t_unix, set_nums, rep_intervals, window_s)

    df["soft_overlap_reps"] = soft
    df["has_rep_intervals"] = has
    df.to_parquet(parquet_path, index=False)

    n_active_with_intervals = int(has.sum())
    return {
        "rec": rec_id, "status": "ok",
        "n_rows": len(df),
        "n_active_with_rep_intervals": n_active_with_intervals,
        "soft_min": float(soft.min()),
        "soft_max": float(soft.max()),
        "soft_mean_active": float(soft[has].mean()) if n_active_with_intervals else 0.0,
        "n_sets_with_reps": len(rep_intervals),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", type=Path, default=ROOT / "data" / "labeled_clean")
    ap.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_clean")
    ap.add_argument("--window-s", type=float, default=2.0,
                    help="Window length in seconds (default 2.0 — matches WINDOW_SIZE/100Hz)")
    args = ap.parse_args()

    if not args.labeled_dir.exists():
        print(f"FATAL: {args.labeled_dir} does not exist", file=sys.stderr); sys.exit(1)

    rec_dirs = sorted(d for d in args.labeled_dir.iterdir()
                      if d.is_dir() and d.name.startswith("recording_"))
    print(f"Processing {len(rec_dirs)} recordings (window={args.window_s} s)\n")
    for rd in rec_dirs:
        rec_id = rd.name
        rep = process_recording(rec_id, args.labeled_dir, args.dataset_dir, args.window_s)
        if rep["status"] == "ok":
            print(f"  {rec_id}: rows={rep['n_rows']}  "
                  f"active_with_reps={rep['n_active_with_rep_intervals']}  "
                  f"soft_range=[{rep['soft_min']:.2f}, {rep['soft_max']:.2f}]  "
                  f"mean_active={rep['soft_mean_active']:.2f}  "
                  f"sets_with_reps={rep['n_sets_with_reps']}")
        else:
            print(f"  {rec_id}: SKIP — {rep['status']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
