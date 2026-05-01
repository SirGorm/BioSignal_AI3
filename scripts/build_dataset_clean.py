"""
Build dataset_clean/ — high-quality merged dataset.

Policy (per user 2026-05-01):
  - Output: NEW directory dataset_clean/ (does not touch dataset_aligned/).
  - Source choice: per-recording, per-modality QC comparison between
    dataset/<rec>/ and dataset_memory/<rec>_memory/. Pick source with
    fewest gaps and best fs adherence; tie-break to dataset/.
  - Recordings: ALL (union dataset ∪ memory). Includes 001..014.
  - Sets: keep ALL kinect_sets in metadata; flag false-start / no-rep / short
    sets in set_quality.json so the labeler can filter.
  - No filtering, no resampling, no notch — raw samples preserved.

Schema unification (output is always `timestamp,<channel>` with Unix-epoch t):
  - dataset/ rec_001-005:  `Time (s),<ch>,Sampling Rate:..,Recording Start Unix Time:..`
                           col 0 is session-relative; Unix start parsed from header.
  - dataset/ rec_006-014:  `timestamp,<ch>` — already Unix epoch.
  - memory/ rec_005:       `device_<ts>_<modality>.csv` with `time,<ch>` columns.
                           Multiple SD sessions; concatenated per modality if used.
                           NOTE: rec_005 memory uses combined imu (ax/ay/az) and
                           combined ppg files — schema differs from rec_006-014.
                           Currently memory NOT used for rec_005 (dataset only).
  - memory/ rec_006-014:   `timestamp,<ch>` — Unix epoch on sensor clock; we
                           subtract KNOWN_OFFSETS[rec] to align to PC clock.

Originals (dataset/, dataset_memory/) are READ-ONLY.

Outputs per recording dataset_clean/recording_NNN/:
  ecg.csv, emg.csv, eda.csv, ax.csv, ay.csv, az.csv, temperature.csv,
  ppg_blue.csv, ppg_green.csv, ppg_red.csv, ppg_ir.csv,
  markers.json, metadata.json, recording_NN_joints.json (verbatim from dataset/),
  clock_alignment.json   — per-modality decisions, offsets, sources
  quality_report.json    — per-modality QC metrics for both sources
  set_quality.json       — per-set flags for downstream filtering

Master:
  dataset_clean/alignment_offsets.json  — overview across all recordings
  dataset_clean/BUILD_LOG.md            — human-readable summary
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset")
MEM = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory")
OUT = ROOT / "dataset_clean"
OUT.mkdir(parents=True, exist_ok=True)

# Per-recording memory→dataset clock offsets (from existing
# dataset_aligned/alignment_offsets.json, verified by cross-correlation).
KNOWN_OFFSETS = {
    "006": 2519.158, "007": 2360.100, "008": 2784.200, "009": 2391.427,
    "010": 2698.113, "011": 5428.085, "012": 2204.103, "013": 2305.805,
    "014": 2598.984,
}

EXPECTED_FS = {
    "ecg": 500.0, "emg": 2000.0, "eda": 50.0, "temperature": 1.0,
    "ax": 100.0, "ay": 100.0, "az": 100.0,
    "ppg_blue": 100.0, "ppg_green": 100.0, "ppg_red": 100.0, "ppg_ir": 100.0,
}

CSV_FILES = list(EXPECTED_FS.keys())
LABEL_FILES = ["markers.json", "metadata.json"]


# ---------------------------------------------------------------------------
# Unified CSV reader (returns DataFrame with `timestamp, <channel>` Unix epoch)
# ---------------------------------------------------------------------------

_HDR_OLD = re.compile(
    r"Time\s*\(s\)\s*,\s*(?P<ch>\w+)\s*,\s*Sampling\s+Rate:\s*"
    r"(?P<fs>[\d.]+)\s*Hz\s*,\s*Recording\s+Start\s+Unix\s+Time:\s*"
    r"(?P<unix>[\d.]+)"
)


def read_csv_to_unix(path: Path) -> Optional[pd.DataFrame]:
    """Read a biosignal CSV in any of the 3 known schemas and return a
    DataFrame with two columns: `timestamp` (Unix epoch sec) and the
    original channel name. Returns None if path missing/empty/unreadable.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "r") as f:
        first_line = f.readline().rstrip("\n").rstrip("\r")
    if not first_line:
        return None

    m = _HDR_OLD.match(first_line)
    if m:
        ch = m.group("ch")
        unix_start = float(m.group("unix"))
        df = pd.read_csv(path, skiprows=1, names=["t_rel", ch])
        df["timestamp"] = df["t_rel"] + unix_start
        return df[["timestamp", ch]].reset_index(drop=True)

    cols = [c.strip() for c in first_line.split(",")]
    if len(cols) < 2:
        return None
    if cols[0].lower() in ("timestamp", "time"):
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if cols[0] != "timestamp":
            df = df.rename(columns={cols[0]: "timestamp"})
        ch = [c for c in df.columns if c != "timestamp"]
        if not ch:
            return None
        return df[["timestamp", ch[0]]].reset_index(drop=True)

    # Unknown schema: try generic 2-column read.
    try:
        df = pd.read_csv(path, names=["timestamp", "value"], skiprows=1)
        if df["timestamp"].iloc[0] > 1e9:
            return df
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# rec_005 memory: concatenate 3 SD sessions per modality. Memory uses combined
# imu and ppg files — split them out into ax/ay/az and ppg_*.
# ---------------------------------------------------------------------------

def read_rec005_memory_modality(modality: str) -> Optional[pd.DataFrame]:
    """Aggregate rec_005 memory sessions for a given modality.

    Memory file naming: device_<timestamp>_<groupname>.csv
      groupname is one of: ecg, emg, eda, temperature, imu, ppg.
    For ax/ay/az we split imu; for ppg_<color> we split ppg.
    """
    base = MEM / "recording_005_memory"
    if not base.exists():
        return None
    if modality in ("ax", "ay", "az"):
        group = "imu"; col = modality
    elif modality.startswith("ppg_"):
        group = "ppg"; col = modality
    else:
        group = modality; col = modality
    files = sorted(base.glob(f"device_*_{group}.csv"))
    if not files:
        return None
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "time" not in df.columns:
            continue
        if col not in df.columns:
            continue
        parts.append(df[["time", col]].rename(columns={"time": "timestamp"}))
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna().sort_values("timestamp").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# QC primitives
# ---------------------------------------------------------------------------

def qc_dataframe(df: Optional[pd.DataFrame], expected_fs: float) -> dict:
    """Compute QC metrics for a DataFrame produced by read_csv_to_unix."""
    if df is None or len(df) == 0:
        return {"available": False, "reason": "empty_or_unreadable"}
    if "timestamp" not in df.columns:
        return {"available": False, "reason": "no_timestamp_column"}

    t = df["timestamp"].to_numpy(dtype=float)
    n = int(len(t))
    t0, t1 = float(t[0]), float(t[-1])
    dur = t1 - t0
    fs_est = (n - 1) / dur if dur > 0 else 0.0

    if n > 1:
        dts = np.diff(t)
        med = float(np.median(dts))
        gap_mask = (dts > max(med * 5, 0.5))
        n_gaps = int(gap_mask.sum())
        max_gap = float(dts.max()) if len(dts) else 0.0
        total_gap_s = float(dts[gap_mask].sum())
    else:
        med = 0.0; n_gaps = 0; max_gap = 0.0; total_gap_s = 0.0

    val_col = next((c for c in df.columns if c != "timestamp"), None)
    if val_col is None:
        std = 0.0; n_nan = n; v_min = 0.0; v_max = 0.0
    else:
        vals = df[val_col].to_numpy()
        if vals.dtype.kind == "f":
            n_nan = int(np.isnan(vals).sum())
            v_finite = vals[np.isfinite(vals)]
        else:
            n_nan = 0
            v_finite = vals
        if len(v_finite):
            std = float(np.std(v_finite))
            v_min = float(np.min(v_finite))
            v_max = float(np.max(v_finite))
        else:
            std = 0.0; v_min = 0.0; v_max = 0.0

    t_unix_ok = bool(t0 > 1e9)

    return {
        "available": True,
        "n_samples": n,
        "t_first": t0,
        "t_last": t1,
        "duration_s": dur,
        "fs_estimated": fs_est,
        "fs_expected": expected_fs,
        "fs_ratio": fs_est / expected_fs if expected_fs else 0.0,
        "median_dt_s": med,
        "n_gaps": n_gaps,
        "max_gap_s": max_gap,
        "total_gap_s": total_gap_s,
        "gap_pct": (total_gap_s / dur * 100.0) if dur > 0 else 0.0,
        "value_std": std,
        "value_min": v_min,
        "value_max": v_max,
        "n_nan": n_nan,
        "t_unix_ok": t_unix_ok,
        "value_col": val_col,
    }


def qc_score(qc: dict) -> tuple[float, list[str]]:
    """Lower is better. Returns (score, reasons-list)."""
    if not qc.get("available"):
        return float("inf"), [f"unavailable: {qc.get('reason', '?')}"]
    reasons = []
    s = 0.0
    s += qc["total_gap_s"] * 1000.0
    if qc["total_gap_s"] > 0:
        reasons.append(f"gap={qc['total_gap_s']:.1f}s")
    s += qc["n_gaps"] * 100.0
    if qc["n_gaps"] > 0:
        reasons.append(f"n_gaps={qc['n_gaps']}")
    fs_dev = abs(qc["fs_ratio"] - 1.0)
    s += fs_dev * 50.0
    if fs_dev > 0.02:
        reasons.append(f"fs_dev={fs_dev*100:.1f}%")
    if qc["value_std"] == 0.0:
        s += 10000
        reasons.append("dead_signal_std=0")
    if not qc["t_unix_ok"]:
        s += 10000
        reasons.append("t_first_not_unix")
    s += qc["n_nan"] * 1.0
    if qc["n_nan"] > 0:
        reasons.append(f"n_nan={qc['n_nan']}")
    return s, reasons


def pick_best_source(rec: str, modality: str, ds_qc: dict, mem_qc: dict,
                     offset: Optional[float]) -> dict:
    ds_score, ds_reasons = qc_score(ds_qc)
    mem_score, mem_reasons = qc_score(mem_qc)

    if ds_score < float("inf") and mem_score < float("inf"):
        # Bias toward dataset by 1.0 unless memory is meaningfully better.
        pick = "dataset" if (ds_score - 1.0) <= mem_score else "memory"
    elif ds_score < float("inf"):
        pick = "dataset"
    elif mem_score < float("inf"):
        if offset is None:
            return {"choice": "none", "reason": "memory_available_but_no_offset",
                    "ds_score": ds_score, "mem_score": mem_score,
                    "ds_reasons": ds_reasons, "mem_reasons": mem_reasons}
        pick = "memory"
    else:
        return {"choice": "none", "reason": "both_unavailable",
                "ds_score": ds_score, "mem_score": mem_score,
                "ds_reasons": ds_reasons, "mem_reasons": mem_reasons}

    if pick == "memory" and offset is None:
        if ds_score < float("inf"):
            pick = "dataset"
            ds_reasons.append("fallback_no_offset_for_memory")
        else:
            return {"choice": "none", "reason": "memory_chosen_but_no_offset",
                    "ds_score": ds_score, "mem_score": mem_score,
                    "ds_reasons": ds_reasons, "mem_reasons": mem_reasons}

    return {
        "choice": pick,
        "ds_score": ds_score,
        "mem_score": mem_score,
        "ds_reasons": ds_reasons,
        "mem_reasons": mem_reasons,
    }


# ---------------------------------------------------------------------------
# Set-quality flagger
# ---------------------------------------------------------------------------

def build_set_quality(rec_dir: Path) -> dict:
    rec = rec_dir.name.replace("recording_", "")
    md_p = rec_dir / "metadata.json"
    mk_p = rec_dir / "markers.json"
    if not md_p.exists():
        return {"recording": rec, "sets": [], "policy": "metadata.json missing"}
    md = json.loads(md_p.read_text())
    sets_md = md.get("kinect_sets", [])

    rep_counts: dict[int, int] = {}
    rep_markers_present_overall = False
    if mk_p.exists():
        mk = json.loads(mk_p.read_text())
        markers = mk if isinstance(mk, list) else mk.get("markers", [])
        for e in markers:
            m = re.match(r"^Set:(\d+)_Rep:\d+$", e.get("label", ""))
            if m:
                sn = int(m.group(1))
                rep_counts[sn] = rep_counts.get(sn, 0) + 1
                rep_markers_present_overall = True

    out = {"recording": rec,
           "policy": "flag-only; downstream labeler decides ok_for_training",
           "thresholds": {"very_short_s": 10.0, "short_s": 15.0, "low_reps": 3},
           "rep_markers_present": rep_markers_present_overall,
           "sets": []}
    for s in sets_md:
        sn = s["set_number"]
        dur = float(s["end_unix_time"] - s["start_unix_time"])
        nr = rep_counts.get(sn, 0)
        joint_p = rec_dir / f"recording_{sn:02d}_joints.json"
        if not joint_p.exists():
            joint_p_alt = rec_dir / f"recording_{sn}_joints.json"
            has_joint = joint_p_alt.exists()
        else:
            has_joint = True

        flags = []
        # Only flag "no_reps_in_markers" when the recording protocol DID log
        # reps for some sets (otherwise rec_001/002 would all be flagged).
        if nr == 0 and rep_markers_present_overall:
            flags.append("no_reps_in_markers")
        if not rep_markers_present_overall:
            flags.append("rep_markers_unavailable_in_protocol")
        if dur < 10.0: flags.append("very_short_<10s")
        elif dur < 15.0: flags.append("short_<15s")
        if 0 < nr < 3: flags.append("low_reps_<3")
        if not has_joint: flags.append("no_joint_file")
        # ok_for_training: dur OK, joint file exists, and either reps>=3 or
        # protocol didn't log reps at all.
        ok = (dur >= 10.0 and has_joint
              and (nr >= 3 or not rep_markers_present_overall)
              and "no_reps_in_markers" not in flags)
        out["sets"].append({
            "set_number": sn,
            "duration_s": round(dur, 3),
            "rep_count_markers": nr,
            "has_joint_file": has_joint,
            "flags": flags,
            "ok_for_training": ok,
        })
    return out


# ---------------------------------------------------------------------------
# Recording builder
# ---------------------------------------------------------------------------

def write_unified(df: pd.DataFrame, out_path: Path) -> None:
    """Write df with columns [timestamp, <ch>] to CSV with 6-decimal floats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")


def build_recording(rec: str) -> dict:
    src_ds = DS / f"recording_{rec}"
    src_mem = MEM / f"recording_{rec}_memory"
    out_dir = OUT / f"recording_{rec}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_present = src_ds.exists()
    mem_present = src_mem.exists()
    offset = KNOWN_OFFSETS.get(rec)

    rec_report = {
        "recording": rec,
        "ds_present": ds_present,
        "mem_present": mem_present,
        "offset_s_memory_minus_dataset": offset,
        "modalities": {},
        "labels": {},
        "joint_files": {},
        "warnings": [],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    is_rec005 = (rec == "005")

    for mod in CSV_FILES:
        # --- read both sources via unified reader ---
        ds_df = None
        if ds_present:
            ds_df = read_csv_to_unix(src_ds / f"{mod}.csv")
        mem_df = None
        if mem_present:
            if is_rec005:
                # rec_005 memory has device_<timestamp>_<group>.csv with combined
                # imu/ppg. Concatenate the 3 sessions and split into per-channel
                # rows. Memory clock is sensor-side; without a verified offset
                # we won't actually USE this output, but we QC it for visibility.
                mem_df = read_rec005_memory_modality(mod)
            else:
                mem_df = read_csv_to_unix(src_mem / f"{mod}.csv")

        ds_qc = qc_dataframe(ds_df, EXPECTED_FS[mod]) if ds_present else \
                {"available": False, "reason": "no_dataset_dir"}
        mem_qc = qc_dataframe(mem_df, EXPECTED_FS[mod]) if mem_present else \
                 {"available": False, "reason": "no_memory_dir"}

        # rec_005 has no known offset → memory is QC-only, never picked.
        eff_offset = offset if not is_rec005 else None
        decision = pick_best_source(rec, mod, ds_qc, mem_qc, eff_offset)
        rec_report["modalities"][mod] = {
            "dataset": ds_qc, "memory": mem_qc, "decision": decision,
        }

        out_csv = out_dir / f"{mod}.csv"
        if decision["choice"] == "dataset" and ds_df is not None:
            write_unified(ds_df, out_csv)
        elif decision["choice"] == "memory" and mem_df is not None and offset is not None:
            mem_df = mem_df.copy()
            mem_df["timestamp"] = mem_df["timestamp"] - offset
            write_unified(mem_df, out_csv)
        else:
            rec_report["warnings"].append(
                f"{mod}: no_source_chosen — file not written ({decision.get('reason', '?')})"
            )

    # Pass-through label files from dataset/.
    if ds_present:
        for fname in LABEL_FILES:
            sp = src_ds / fname
            dp = out_dir / fname
            if sp.exists():
                shutil.copy2(sp, dp)
                rec_report["labels"][fname] = "copied"
            else:
                rec_report["labels"][fname] = "missing_in_dataset"

        # Glob both 'recording_*_joints.json' and 'recording_*_joints_feil.json'
        # patterns; filter out any name containing 'feil' (Norwegian: error)
        # since those mark known-bad Kinect captures (verified in rec_005).
        joints_all = sorted(set(src_ds.glob("recording_*joints*.json")))
        joints = [j for j in joints_all if "feil" not in j.name.lower()]
        skipped_feil = [j.name for j in joints_all if "feil" in j.name.lower()]
        names = []
        for j in joints:
            shutil.copy2(j, out_dir / j.name)
            names.append(j.name)
        rec_report["joint_files"] = {"n": len(joints), "names": names,
                                     "skipped_feil": skipped_feil}
    else:
        rec_report["warnings"].append("dataset/ source missing — no labels/joints copied")
        rec_report["labels"] = {f: "missing_no_dataset" for f in LABEL_FILES}
        rec_report["joint_files"] = {"n": 0, "names": []}

    (out_dir / "clock_alignment.json").write_text(json.dumps(rec_report, indent=2))
    (out_dir / "quality_report.json").write_text(json.dumps(rec_report["modalities"], indent=2))
    sq = build_set_quality(out_dir)
    (out_dir / "set_quality.json").write_text(json.dumps(sq, indent=2))

    return rec_report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    print(f"=== build_dataset_clean: {OUT} ===\n")

    ds_recs = {d.name.replace("recording_", "") for d in DS.iterdir()
               if d.is_dir() and d.name.startswith("recording_")}
    mem_recs = {d.name.replace("recording_", "").replace("_memory", "")
                for d in MEM.iterdir() if d.is_dir() and d.name.startswith("recording_")}
    all_recs = sorted(ds_recs | mem_recs)
    print(f"Found {len(all_recs)} recordings: {all_recs}\n")

    master = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DS),
        "source_memory": str(MEM),
        "output_root": str(OUT),
        "policy": {
            "selection": "per-modality QC: gaps, fs deviation, dead-signal; tie-break to dataset/",
            "offset_application": "memory timestamps shifted by KNOWN_OFFSETS[rec]; dataset timestamps verbatim or schema-converted",
            "set_handling": "all kinect_sets retained; flags in set_quality.json",
            "filtering_resampling": "none",
            "rec_005_memory": "QC-only (multi-session SD-card layout, no verified offset); not used as source",
        },
        "known_offsets": KNOWN_OFFSETS,
        "recordings": {},
    }

    for rec in all_recs:
        print(f"--- recording_{rec} ---")
        rep = build_recording(rec)
        compact = {
            "ds_present": rep["ds_present"],
            "mem_present": rep["mem_present"],
            "offset_s_memory_minus_dataset": rep["offset_s_memory_minus_dataset"],
            "modality_choice": {m: v["decision"]["choice"]
                                for m, v in rep["modalities"].items()},
            "n_joint_files": rep["joint_files"]["n"],
            "labels_status": rep["labels"],
            "warnings": rep["warnings"],
        }
        master["recordings"][rec] = compact
        for m, v in rep["modalities"].items():
            d = v["decision"]
            ds_s = d.get("ds_score", "?")
            mem_s = d.get("mem_score", "?")
            ds_s_str = "inf" if ds_s == float("inf") else f"{ds_s:.0f}"
            mem_s_str = "inf" if mem_s == float("inf") else f"{mem_s:.0f}"
            print(f"  {m:12s} -> {d['choice']:8s}  ds={ds_s_str:>6}  mem={mem_s_str:>6}")
        if rep["warnings"]:
            for w in rep["warnings"]:
                print(f"  WARN: {w}")
        print()

    (OUT / "alignment_offsets.json").write_text(json.dumps(master, indent=2))

    lines = [
        "# dataset_clean BUILD_LOG",
        f"Generated: {master['generated_utc']}",
        f"Sources: dataset={DS}, memory={MEM}",
        f"Recordings: {len(all_recs)} ({', '.join(all_recs)})",
        "",
        "## Per-modality source choice",
        "",
        "| rec | ecg | emg | eda | temp | acc | ppg_g | ppg_b | ppg_r | ppg_ir | warnings |",
        "|-----|-----|-----|-----|------|-----|-------|-------|-------|--------|----------|",
    ]
    for rec in all_recs:
        c = master["recordings"][rec]["modality_choice"]
        def short(x):
            return {"dataset": "ds", "memory": "mem", "none": "—"}.get(x, x)
        acc = ",".join(short(c.get(k, "—")) for k in ("ax", "ay", "az"))
        warns = "; ".join(master["recordings"][rec]["warnings"]) or "ok"
        lines.append(
            f"| {rec} | {short(c.get('ecg','—'))} | {short(c.get('emg','—'))} | "
            f"{short(c.get('eda','—'))} | {short(c.get('temperature','—'))} | "
            f"{acc} | {short(c.get('ppg_green','—'))} | {short(c.get('ppg_blue','—'))} | "
            f"{short(c.get('ppg_red','—'))} | {short(c.get('ppg_ir','—'))} | {warns} |"
        )
    (OUT / "BUILD_LOG.md").write_text("\n".join(lines))

    elapsed = time.time() - t0
    print(f"=== DONE in {elapsed:.1f}s ===")
    print(f"  alignment_offsets.json: {OUT/'alignment_offsets.json'}")
    print(f"  BUILD_LOG.md:           {OUT/'BUILD_LOG.md'}")


if __name__ == "__main__":
    main()
