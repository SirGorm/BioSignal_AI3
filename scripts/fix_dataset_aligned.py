"""
Fix surgically-detected inconsistencies in dataset_aligned/.

Detected issues (see conversation 2026-05-01):
  1. rec_006: markers.json has Set:13 entries that don't match metadata's 12 sets.
  2. rec_008: markers.json has Set:8 + Set:13 that don't match metadata's
     11 sets (numbers 1..7, 9..12); recording_08_joints.json is an orphan
     (corresponds to dropped src set 8).
  3. rec_009: recording_13_joints.json + recording_14_joints.json are post-session
     orphans (no matching kinect_set or marker).
  4. rec_011, rec_014: dataset_aligned/.../temperature.csv ends ~63-97s before
     the biosignal end because the memory-source range was shorter than the
     dataset clock range. Extend with last-value forward-fill at 1 Hz so the
     temperature column covers the full signal duration.
  5. rec_003: not registered in alignment_offsets.json. Add it as a passthrough
     entry (no memory variant existed). Also document the legitimate 401s
     recording-pause @ t=737s and the 50 Hz PPG-green deviation.

Original dataset/ and dataset_memory/ are READ-ONLY. All edits are confined
to dataset_aligned/.

Reversibility:
  Every modified file is first backed up to dataset_aligned/_backup_pre_fix/<rel>.
  Run scripts/restore_dataset_aligned_backup.py (manual one-liner) to roll back.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALIGNED = ROOT / "dataset_aligned"
BACKUP = ALIGNED / "_backup_pre_fix"
SRC_DATASET = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset")
SRC_MEMORY = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory")

if not ALIGNED.exists():
    print(f"FATAL: {ALIGNED} missing"); sys.exit(1)


def _backup(p: Path) -> None:
    """Copy p to BACKUP/<relative-to-ALIGNED> before editing."""
    rel = p.relative_to(ALIGNED)
    dst = BACKUP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():  # never overwrite an existing backup
        shutil.copy2(p, dst)


def fix_markers_drop_sets(rec: str, drop_set_nums: list[int]) -> dict:
    """Remove Set:N_Start, Set_N_End, and Set:N_Rep:* entries for N in drop_set_nums.

    Markers.json format: {"markers": [{"unix_time", "time", "label", "color"}, ...], "total_markers"}.
    Renumbering is NOT performed — set_number gaps are now consistent with metadata.
    """
    p = ALIGNED / f"recording_{rec}" / "markers.json"
    _backup(p)
    data = json.loads(p.read_text())
    if isinstance(data, list):
        markers = data
        wrap = False
    else:
        markers = data.get("markers", [])
        wrap = True
    n_before = len(markers)
    drop_set = set(drop_set_nums)
    pat_start = re.compile(r"^Set:(\d+)_Start$")
    pat_end = re.compile(r"^Set_(\d+)_End$")
    pat_rep = re.compile(r"^Set:(\d+)_Rep:\d+$")
    keep = []
    dropped = []
    for e in markers:
        lbl = e.get("label", "")
        sn = None
        for pat in (pat_start, pat_end, pat_rep):
            m = pat.match(lbl)
            if m:
                sn = int(m.group(1)); break
        if sn is not None and sn in drop_set:
            dropped.append(lbl)
            continue
        keep.append(e)
    if wrap:
        data["markers"] = keep
        data["total_markers"] = len(keep)
        out = data
    else:
        out = keep
    p.write_text(json.dumps(out, indent=2))
    return {"recording": rec, "n_markers_before": n_before, "n_markers_after": len(keep),
            "dropped_labels": dropped, "drop_set_nums": drop_set_nums}


def delete_orphan_joint_file(rec: str, set_num: int) -> dict:
    p = ALIGNED / f"recording_{rec}" / f"recording_{set_num:02d}_joints.json"
    if not p.exists():
        return {"recording": rec, "set_num": set_num, "status": "not_found"}
    _backup(p)
    size = p.stat().st_size
    p.unlink()
    return {"recording": rec, "set_num": set_num, "status": "deleted", "bytes_freed": size}


def extend_temperature_with_forward_fill(rec: str) -> dict:
    """Re-shift memory temperature, but if its range falls short of the signal
    range, append forward-filled samples at 1 Hz so timestamps cover the full
    biosignal duration.

    Source of truth for biosignal range: dst ecg.csv first/last timestamp.
    Source of truth for offset: alignment_offsets.json (or memory file's own range).
    """
    rec_dir = ALIGNED / f"recording_{rec}"
    align = json.loads((ALIGNED / "alignment_offsets.json").read_text())
    rec_info = align["recordings"].get(rec)
    if rec_info is None:
        return {"recording": rec, "status": "not_in_alignment_offsets"}
    offset = float(rec_info["offset_s_for_temperature_only"])

    # Authoritative biosignal range from dst ecg.csv (verbatim from src).
    ecg_p = rec_dir / "ecg.csv"
    ecg_t = pd.read_csv(ecg_p, usecols=["timestamp"])["timestamp"].values
    t0 = float(ecg_t[0]); t1 = float(ecg_t[-1])

    # Re-shift temperature from source memory.
    src_mem_p = SRC_MEMORY / f"recording_{rec}_memory" / "temperature.csv"
    if not src_mem_p.exists():
        return {"recording": rec, "status": "no_memory_source"}
    df = pd.read_csv(src_mem_p)
    df["timestamp"] = df["timestamp"] - offset
    df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)].reset_index(drop=True)
    n_real = len(df)
    if n_real == 0:
        return {"recording": rec, "status": "no_overlap"}

    last_t = float(df["timestamp"].iloc[-1])
    last_v = float(df["temperature"].iloc[-1])
    pad_rows = []
    n_pad = int((t1 - last_t)) if t1 > last_t else 0
    for k in range(1, n_pad + 1):
        pad_rows.append({"timestamp": last_t + k, "temperature": last_v})
    if pad_rows:
        df = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)
    # Also forward-fill at the START if memory range began after t0.
    first_t = float(df["timestamp"].iloc[0])
    pre_rows = []
    if first_t > t0:
        first_v = float(df["temperature"].iloc[0])
        n_pre = int((first_t - t0))
        for k in range(n_pre, 0, -1):
            pre_rows.append({"timestamp": first_t - k, "temperature": first_v})
        if pre_rows:
            df = pd.concat([pd.DataFrame(pre_rows), df], ignore_index=True)

    dst = rec_dir / "temperature.csv"
    _backup(dst)
    df.to_csv(dst, index=False, float_format="%.6f")
    return {
        "recording": rec, "status": "extended",
        "ecg_t_first": t0, "ecg_t_last": t1,
        "temp_t_first": float(df["timestamp"].iloc[0]),
        "temp_t_last": float(df["timestamp"].iloc[-1]),
        "n_real_samples": n_real,
        "n_pad_after": n_pad,
        "n_pad_before": len(pre_rows),
        "policy": "shift+trim+forward_fill_to_signal_range",
    }


def add_recording_003_to_offsets() -> dict:
    """Add recording_003 to alignment_offsets.json as a passthrough (no memory variant)."""
    ap = ALIGNED / "alignment_offsets.json"
    _backup(ap)
    data = json.loads(ap.read_text())
    if "003" in data["recordings"]:
        return {"status": "already_present"}

    # Authoritative biosignal range from dst ecg.csv (already produced by manual build).
    ecg_p = ALIGNED / "recording_003" / "ecg.csv"
    ecg_t = pd.read_csv(ecg_p, usecols=["timestamp"])["timestamp"].values

    # Joint-file count.
    rec_dir = ALIGNED / "recording_003"
    n_joints = sum(1 for f in rec_dir.iterdir() if "joints" in f.name)

    data["recordings"]["003"] = {
        "offset_s_for_temperature_only": None,
        "confidence": "n/a",
        "ds_t_unix_first": float(ecg_t[0]),
        "ds_t_unix_last": float(ecg_t[-1]),
        "n_biosignal_csvs_copied": 10,
        "temperature_status": "from_dataset_passthrough",
        "n_joint_files": n_joints,
        "notes": [
            "No dataset_memory variant exists for this recording — temperature was sourced from dataset/ (verbatim, no offset).",
            "Biosignals contain a legitimate 401s recording pause at session-relative t≈737s (all modalities). Time vector is correct (Unix epoch) but apparent fs computed as (n-1)/(t_last-t_first) under-reports the true instantaneous rate by ~13% for this reason.",
            "PPG-green sensor logged at 50 Hz (not 100 Hz like recordings 006-014). Verified via median dt = 0.02s. Feature extractors that hardcode 100 Hz must check fs per recording.",
        ],
    }
    ap.write_text(json.dumps(data, indent=2))
    return {"status": "added", "ds_t_unix_first": float(ecg_t[0]), "ds_t_unix_last": float(ecg_t[-1])}


# ---------------------------------------------------------------------------
report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "actions": []}

print("=== fix_dataset_aligned: starting surgical fixes ===\n")

# Fix 1: rec_006 markers — drop Set:13
print("[1/5] rec_006: dropping Set:13 markers...")
r = fix_markers_drop_sets("006", [13])
print(f"      n_markers {r['n_markers_before']} -> {r['n_markers_after']}, dropped: {r['dropped_labels']}")
report["actions"].append({"fix": "markers_drop_sets", **r})

# Fix 2a: rec_008 markers — drop Set:8 and Set:13
print("\n[2a/5] rec_008: dropping Set:8 and Set:13 markers...")
r = fix_markers_drop_sets("008", [8, 13])
print(f"       n_markers {r['n_markers_before']} -> {r['n_markers_after']}, dropped: {r['dropped_labels']}")
report["actions"].append({"fix": "markers_drop_sets", **r})

# Fix 2b: rec_008 — delete orphan joint file 08
print("\n[2b/5] rec_008: deleting orphan recording_08_joints.json...")
r = delete_orphan_joint_file("008", 8)
print(f"       {r}")
report["actions"].append({"fix": "delete_orphan_joint_file", **r})

# Fix 3: rec_009 — delete orphan joint files 13, 14
print("\n[3/5] rec_009: deleting orphan joint files 13, 14...")
for sn in (13, 14):
    r = delete_orphan_joint_file("009", sn)
    print(f"      set {sn}: {r}")
    report["actions"].append({"fix": "delete_orphan_joint_file", **r})

# Fix 4: rec_011, rec_014 — extend temperature with forward fill
print("\n[4/5] rec_011, rec_014: extending temperature.csv with forward-fill...")
for rec in ("011", "014"):
    r = extend_temperature_with_forward_fill(rec)
    print(f"      rec_{rec}: {r}")
    report["actions"].append({"fix": "extend_temperature_with_forward_fill", **r})

# Fix 5: register rec_003 in alignment_offsets.json
print("\n[5/5] rec_003: registering in alignment_offsets.json...")
r = add_recording_003_to_offsets()
print(f"      {r}")
report["actions"].append({"fix": "add_recording_003_to_offsets", **r})

# Persist a per-run report.
report_path = ALIGNED / "_fix_dataset_aligned_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"\n=== DONE ===  Backups in {BACKUP}\nReport: {report_path}")
