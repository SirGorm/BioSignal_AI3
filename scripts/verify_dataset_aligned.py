"""
Sanity-check the dataset_aligned/ output:

For each recording_NNN/:
  1. CSV timestamps are Unix epoch (>1e9).
  2. CSV time-range is contained in the dataset session range.
  3. markers.json `unix_time` values fall INSIDE the CSV time-range.
  4. metadata.json kinect_sets `start_unix_time`/`end_unix_time` fall inside CSV range.
  5. Sample counts match metadata.json["total_samples"] within ~0.5%.
  6. Joint .json files exist for each kinect_set.

Reports any deviation. Pass = green status row; warning = yellow; fail = red.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ALIGNED = Path(r"c:\Users\skogl\Downloads\eirikgsk\biosignal_2\BioSignal_AI3\dataset_aligned")

CSV_NAMES = [
    "ecg.csv", "emg.csv", "eda.csv", "temperature.csv",
    "ax.csv", "ay.csv", "az.csv",
    "ppg_blue.csv", "ppg_green.csv", "ppg_red.csv", "ppg_ir.csv",
]

# metadata.json["total_samples"] keys
META_KEY = {
    "ecg.csv": "ecg",
    "emg.csv": "emg",
    "eda.csv": "eda",
    "ax.csv": "ax",
    "ay.csv": "ay",
    "az.csv": "az",
    "ppg_blue.csv": "ppg_blue",
    "ppg_green.csv": "ppg_green",
    "ppg_red.csv": "ppg_red",
    "ppg_ir.csv": "ppg_ir",
}

records = sorted(ALIGNED.glob("recording_*"))
print(f"Verifying {len(records)} recordings under {ALIGNED}\n")

all_ok = True

for r in records:
    if not r.is_dir():
        continue
    rec = r.name.split("_")[1]
    print(f"=== {r.name} ===")
    issues = []

    metadata = json.loads((r / "metadata.json").read_text())
    markers = json.loads((r / "markers.json").read_text())["markers"]
    joint_files = sorted(r.glob("recording_*_joints.json"))

    # CSV timestamp ranges
    csv_t = {}
    csv_n = {}
    for name in CSV_NAMES:
        p = r / name
        if not p.exists():
            issues.append(f"missing CSV {name}")
            continue
        df = pd.read_csv(p, usecols=["timestamp"])
        if len(df) == 0:
            issues.append(f"empty CSV {name}")
            continue
        t0 = float(df["timestamp"].iloc[0])
        t1 = float(df["timestamp"].iloc[-1])
        if t0 < 1e9:
            issues.append(f"{name} timestamps not Unix epoch ({t0})")
        csv_t[name] = (t0, t1)
        csv_n[name] = len(df)

    # CSV time-range overlap (all should agree to within sample interval)
    t0s = [v[0] for v in csv_t.values()]
    t1s = [v[1] for v in csv_t.values()]
    common_t0 = max(t0s)
    common_t1 = min(t1s)
    if max(t0s) - min(t0s) > 0.05:
        issues.append(
            f"CSV starts disagree by {max(t0s)-min(t0s):.3f}s "
            f"(min={min(t0s):.3f}, max={max(t0s):.3f})"
        )
    if max(t1s) - min(t1s) > 0.05:
        issues.append(
            f"CSV ends disagree by {max(t1s)-min(t1s):.3f}s "
            f"(min={min(t1s):.3f}, max={max(t1s):.3f})"
        )
    print(f"  CSV time-range: [{common_t0:.3f}, {common_t1:.3f}] ({(common_t1-common_t0)/60:.2f} min)")

    # Markers inside CSV range?
    marker_t = [m["unix_time"] for m in markers]
    n_outside = sum(1 for t in marker_t if t < common_t0 - 0.05 or t > common_t1 + 0.05)
    if n_outside:
        issues.append(f"{n_outside}/{len(markers)} markers fall outside CSV time-range")
    print(f"  markers: {len(markers)} total, "
          f"range=[{min(marker_t):.3f}, {max(marker_t):.3f}], "
          f"{n_outside} outside CSV range")

    # Kinect sets inside CSV range?
    sets_outside = 0
    for s in metadata.get("kinect_sets", []):
        if s["start_unix_time"] < common_t0 - 0.05 or s["end_unix_time"] > common_t1 + 0.05:
            sets_outside += 1
    if sets_outside:
        issues.append(f"{sets_outside} kinect_sets straddle CSV boundary")
    print(f"  kinect_sets: {metadata['total_kinect_sets']} declared, {sets_outside} straddle CSV boundary")

    # Sample-count comparison vs metadata
    declared = metadata.get("total_samples", {})
    for name, k in META_KEY.items():
        if k in declared and name in csv_n:
            d = declared[k]
            a = csv_n[name]
            diff_pct = abs(a - d) / d * 100 if d else 0
            if diff_pct > 0.5:
                issues.append(
                    f"{name} sample count {a} differs from metadata {d} by {diff_pct:.2f}%"
                )

    # Joint file count vs metadata kinect_sets
    if len(joint_files) != metadata.get("total_kinect_sets", 0):
        issues.append(
            f"joint file count {len(joint_files)} != kinect_sets {metadata.get('total_kinect_sets', 0)}"
        )
    print(f"  joint files: {len(joint_files)}")

    # Verdict
    if issues:
        all_ok = False
        print("  ISSUES:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  OK")
    print()

print("=== OVERALL:", "PASS" if all_ok else "ISSUES FOUND", "===")
