"""
Build dataset_aligned/ — perfectly time-aligned training set per recording.

Source policy:
  - Biosignals (ecg, emg, eda, ax, ay, az, ppg_*) come from dataset/recording_NNN/
    (PC-streamed; already on dataset clock; no modification).
  - temperature.csv comes from dataset_memory/recording_NNN_memory/ because
    dataset/'s temperature.csv is empty or missing for ALL 9 recordings.
    Memory's temperature timestamps are SHIFTED by per-recording offset
    (memory_clock - offset = dataset_clock) and trimmed to dataset's time-range.
  - markers.json, metadata.json, recording_NN_joints.json copied verbatim
    from dataset/ (already on dataset clock).

Source dirs (READ-ONLY, never modified):
  C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/recording_NNN/
  C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset_memory/recording_NNN_memory/

Output dir:
  C:/Users/skogl/Downloads/eirikgsk/biosignal_2/BioSignal_AI3/dataset_aligned/
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DS = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset")
MEM = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory")
OUT_ROOT = Path(r"c:\Users\skogl\Downloads\eirikgsk\biosignal_2\BioSignal_AI3\dataset_aligned")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Per-recording offsets (memory_clock - dataset_clock), in seconds.
# Confirmed via cross-correlation in scripts/verify_memory_dataset_match.py and
# scripts/check_special_recordings.py. Used ONLY for temperature.csv shifting.
OFFSETS = {
    "006": {"offset_s": 2519.158, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
    "007": {"offset_s": 2360.100, "pearson": 1.00, "probes_agree": "8/10", "confidence": "high"},
    "008": {"offset_s": 2784.200, "pearson": 1.00, "probes_agree": "6/10", "confidence": "medium"},
    "009": {"offset_s": 2391.427, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
    "010": {"offset_s": 2698.113, "pearson": 1.00, "probes_agree": "6/10", "confidence": "medium"},
    "011": {"offset_s": 5428.085, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
    "012": {"offset_s": 2204.103, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
    "013": {"offset_s": 2305.805, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
    "014": {"offset_s": 2598.984, "pearson": 1.00, "probes_agree": "3/3", "confidence": "high"},
}

# Biosignals copied verbatim from dataset/ (already on dataset clock)
DS_BIOSIGNAL_CSVS = [
    "ecg.csv", "emg.csv", "eda.csv",
    "ax.csv", "ay.csv", "az.csv",
    "ppg_blue.csv", "ppg_green.csv", "ppg_red.csv", "ppg_ir.csv",
]

# Ground-truth files copied verbatim from dataset/
DS_LABEL_FILES = ["markers.json", "metadata.json"]


def get_dataset_time_range(rec: str) -> tuple[float, float]:
    """ECG.csv first & last timestamp from dataset = canonical time range."""
    p = DS / f"recording_{rec}" / "ecg.csv"
    df = pd.read_csv(p, usecols=["timestamp"])
    return float(df["timestamp"].iloc[0]), float(df["timestamp"].iloc[-1])


def shift_and_trim_temperature(src: Path, dst: Path, offset: float, ds_t0: float, ds_t1: float) -> dict:
    df = pd.read_csv(src)
    if "timestamp" not in df.columns:
        raise SystemExit(f"FATAL: {src} has no 'timestamp' column")
    n_before = len(df)
    df["timestamp"] = df["timestamp"] - offset
    mask = (df["timestamp"] >= ds_t0) & (df["timestamp"] <= ds_t1)
    df = df.loc[mask].reset_index(drop=True)
    n_after = len(df)
    df.to_csv(dst, index=False, float_format="%.6f")
    return {
        "src": str(src),
        "dst": str(dst),
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped": n_before - n_after,
        "t_first": float(df["timestamp"].iloc[0]) if n_after else None,
        "t_last": float(df["timestamp"].iloc[-1]) if n_after else None,
    }


summary = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_dataset": str(DS),
    "source_memory": str(MEM),
    "output_root": str(OUT_ROOT),
    "policy": {
        "biosignals_except_temperature": "copied verbatim from dataset/",
        "temperature": "from dataset_memory/, timestamp -= per-recording offset, trimmed to dataset range",
        "markers_metadata_joints": "copied verbatim from dataset/",
    },
    "recordings": {},
}

for rec, off_info in OFFSETS.items():
    src_ds = DS / f"recording_{rec}"
    src_mem = MEM / f"recording_{rec}_memory"
    out_dir = OUT_ROOT / f"recording_{rec}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_ds.exists():
        print(f"SKIP rec_{rec}: dataset folder missing")
        continue

    ds_t0, ds_t1 = get_dataset_time_range(rec)
    offset = off_info["offset_s"]
    print(f"\n=== rec_{rec} (offset {offset:.3f}s, ds=[{ds_t0:.3f}, {ds_t1:.3f}]) ===")

    # 1. Copy biosignal CSVs verbatim from dataset/
    csv_reports = {}
    for name in DS_BIOSIGNAL_CSVS:
        src = src_ds / name
        dst = out_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            n = sum(1 for _ in open(dst, "rb")) - 1  # rough line count
            csv_reports[name] = {"status": "copied_from_dataset", "n_lines": n}
            print(f"  copy dataset/{name:18s}  n_rows~={n}")
        else:
            csv_reports[name] = {"status": "missing_in_dataset"}
            print(f"  WARN missing dataset/{name}")

    # 2. Temperature: from memory, with offset shift + trim
    src_temp = src_mem / "temperature.csv"
    dst_temp = out_dir / "temperature.csv"
    if src_temp.exists():
        rep = shift_and_trim_temperature(src_temp, dst_temp, offset, ds_t0, ds_t1)
        csv_reports["temperature.csv"] = {"status": "from_memory_shifted", **rep}
        print(f"  shift memory/temperature.csv: in={rep['n_before']} out={rep['n_after']} dropped={rep['n_dropped']}")
    else:
        csv_reports["temperature.csv"] = {"status": "missing_in_memory"}
        print("  WARN memory temperature.csv missing")

    # 3. Copy ground-truth JSONs verbatim from dataset/
    json_reports = {}
    for jf in DS_LABEL_FILES:
        src = src_ds / jf
        dst = out_dir / jf
        if src.exists():
            shutil.copy2(src, dst)
            json_reports[jf] = {"status": "copied", "src": str(src)}
            print(f"  copy {jf}")
        else:
            json_reports[jf] = {"status": "missing"}
            print(f"  WARN missing {jf}")

    joint_files = sorted(src_ds.glob("recording_*_joints.json"))
    for jf in joint_files:
        shutil.copy2(jf, out_dir / jf.name)
    json_reports["joint_files"] = {
        "n": len(joint_files),
        "names": [j.name for j in joint_files],
    }
    print(f"  copy {len(joint_files)} joint json files")

    # 4. clock_alignment.json (per recording)
    rec_payload = {
        "recording": rec,
        "policy": "biosignals from dataset/ (verbatim); temperature from memory (offset-shifted); labels from dataset/ (verbatim)",
        "offset_s_memory_minus_dataset": offset,
        "verification_pearson": off_info["pearson"],
        "verification_probes_agree": off_info["probes_agree"],
        "verification_confidence": off_info["confidence"],
        "source_dataset_dir": str(src_ds),
        "source_memory_dir": str(src_mem),
        "dataset_t_unix_first": ds_t0,
        "dataset_t_unix_last": ds_t1,
        "csvs": csv_reports,
        "passthrough": json_reports,
    }
    (out_dir / "clock_alignment.json").write_text(json.dumps(rec_payload, indent=2))

    summary["recordings"][rec] = {
        "offset_s_for_temperature_only": offset,
        "confidence": off_info["confidence"],
        "ds_t_unix_first": ds_t0,
        "ds_t_unix_last": ds_t1,
        "n_biosignal_csvs_copied": sum(
            1 for v in csv_reports.values() if v.get("status") == "copied_from_dataset"
        ),
        "temperature_status": csv_reports.get("temperature.csv", {}).get("status"),
        "n_joint_files": len(joint_files),
    }

(OUT_ROOT / "alignment_offsets.json").write_text(json.dumps(summary, indent=2))
print(f"\n=== DONE ===\nWrote {OUT_ROOT/'alignment_offsets.json'}")
print(f"Built {len(summary['recordings'])} recordings under {OUT_ROOT}")
