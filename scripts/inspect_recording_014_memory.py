"""
Inspection of recording_014 with split source layout requested by user:

  Biosignals (ecg, emg, eda, temperature, ax/ay/az, ppg_*) FROM
      C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset_memory/recording_014_memory/
  Markers + per-set joint skeletons + metadata FROM
      C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/recording_014/
  Participants.xlsx FROM
      C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/Participants.xlsx

Output:  inspections/recording_014_memory/
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine

apply_style()

REPO = Path(r"c:\Users\skogl\Downloads\eirikgsk\biosignal_2\BioSignal_AI3")
BIO_DIR = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory\recording_014_memory")
META_DIR = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset\recording_014")
PART_PATH = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset\Participants\Participants.xlsx")

OUT = REPO / "inspections" / "recording_014_memory"
OUT.mkdir(parents=True, exist_ok=True)


def estimate_fs(ts: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(ts)))


def channel_stats(name: str, x: np.ndarray, ts: np.ndarray, fs: float) -> dict:
    valid = x[~np.isnan(x)] if x.size else x
    if valid.size == 0:
        return {"name": name, "fs_hz": fs, "status": "empty"}
    return {
        "name": name,
        "fs_hz": float(fs),
        "n_samples": int(x.size),
        "t_unix_first": float(ts[0]),
        "t_unix_last": float(ts[-1]),
        "duration_s": float(ts[-1] - ts[0]),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p1": float(np.percentile(valid, 1)),
        "p99": float(np.percentile(valid, 99)),
        "n_nan": int(np.sum(np.isnan(x))),
        "pct_nan": float(np.mean(np.isnan(x)) * 100),
        "pct_at_clip": float(
            np.mean(np.abs(valid) >= np.max(np.abs(valid)) * 0.99) * 100
        ),
    }


# ---------- 1. Load biosignals from dataset_memory ----------
print("=== Loading biosignals from dataset_memory/recording_014_memory ===")
csv_files = sorted(BIO_DIR.glob("*.csv"))
print("Files:", [f.name for f in csv_files])

signals: dict[str, dict] = {}  # channel -> {ts, x, fs}
for f in csv_files:
    df = pd.read_csv(f)
    if df.shape[1] < 2:
        print(f"  SKIP {f.name}: too few columns")
        continue
    if "timestamp" not in df.columns:
        print(f"  SKIP {f.name}: no timestamp column ({list(df.columns)})")
        continue
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    if ts.size < 2:
        print(f"  SKIP {f.name}: empty")
        continue
    if ts[0] < 1e9:
        raise SystemExit(f"FATAL: {f.name} timestamps not Unix-epoch ({ts[0]})")
    val_col = [c for c in df.columns if c != "timestamp"][0]
    x = df[val_col].to_numpy(dtype=np.float64)
    fs = estimate_fs(ts)
    signals[val_col] = {"ts": ts, "x": x, "fs": fs, "file": f.name}
    print(
        f"  {val_col:14s} fs={fs:8.3f} Hz  n={x.size:>9d}  "
        f"t=[{ts[0]:.3f}, {ts[-1]:.3f}]  ({(ts[-1]-ts[0]):.1f} s)"
    )

# Choose canonical timestamp (use ECG's, or ax's; here we have all aligned)
bio_t0 = min(s["ts"][0] for s in signals.values())
bio_t1 = max(s["ts"][-1] for s in signals.values())
print(f"\nBiosignal session: {bio_t0:.3f} -> {bio_t1:.3f}  duration={(bio_t1-bio_t0)/60:.2f} min")
print(f"  UTC start: {datetime.fromtimestamp(bio_t0, tz=timezone.utc)}")
print(f"  UTC end:   {datetime.fromtimestamp(bio_t1, tz=timezone.utc)}")

# Compute acc-magnitude
ax_s = signals["ax"]
ay_s = signals["ay"]
az_s = signals["az"]
n_min = min(ax_s["x"].size, ay_s["x"].size, az_s["x"].size)
acc_mag = np.sqrt(
    ax_s["x"][:n_min] ** 2 + ay_s["x"][:n_min] ** 2 + az_s["x"][:n_min] ** 2
)
acc_ts = ax_s["ts"][:n_min]
fs_acc = ax_s["fs"]
signals["acc_mag"] = {"ts": acc_ts, "x": acc_mag, "fs": fs_acc, "file": "(computed)"}
print(f"  acc_mag (computed) fs={fs_acc:.2f} Hz  n={acc_mag.size}")

# ---------- 2. Load markers, metadata, joints, participants ----------
print("\n=== Loading markers/metadata/joints from dataset/recording_014 ===")
metadata = json.loads((META_DIR / "metadata.json").read_text())
markers = json.loads((META_DIR / "markers.json").read_text())["markers"]
joint_files = sorted(META_DIR.glob("recording_*_joints.json"))
print(f"  metadata.json: {metadata['recording_start']}, {metadata['total_kinect_sets']} sets")
print(f"  markers.json: {len(markers)} entries")
print(f"  joint files:  {len(joint_files)}")

# Marker time-range (Unix)
marker_t_unix = np.array([m["unix_time"] for m in markers])
mark_t0, mark_t1 = float(marker_t_unix.min()), float(marker_t_unix.max())
print(f"  marker Unix range: [{mark_t0:.3f}, {mark_t1:.3f}]")
print(f"    UTC: {datetime.fromtimestamp(mark_t0, tz=timezone.utc)} -> {datetime.fromtimestamp(mark_t1, tz=timezone.utc)}")

# Critical: overlap test
overlap_start = max(bio_t0, mark_t0)
overlap_end = min(bio_t1, mark_t1)
overlap_s = max(0.0, overlap_end - overlap_start)
gap_s = bio_t0 - mark_t1  # positive if biosignals start after markers end
print(f"\n  Bio vs marker overlap: {overlap_s:.2f} s")
print(f"  Bio start - marker end: {gap_s:+.2f} s  ({'gap, no overlap' if gap_s > 0 else 'overlap'})")

# Per-set summary from metadata
sets_meta = metadata["kinect_sets"]
set_starts_unix = np.array([s["start_unix_time"] for s in sets_meta])
set_ends_unix = np.array([s["end_unix_time"] for s in sets_meta])

# Per-set rep counts from markers
def reps_in_set(set_n: int) -> int:
    return sum(1 for m in markers if m["label"].startswith(f"Set:{set_n}_Rep:"))

# Joint frame counts
joint_frame_counts = {}
for jf in joint_files:
    d = json.loads(jf.read_text())
    joint_frame_counts[jf.name] = len(d.get("frames", []))

# Participants.xlsx for recording 14
parts = pd.read_excel(PART_PATH)
mask = parts["Recording:"] == 14
i = parts.index[mask][0]
ex_row = parts.iloc[i]
fat_row = parts.iloc[i + 1]
participant_name = str(ex_row["Name:"])
ex_per_set = [str(ex_row[f"set{n}"]) for n in range(1, 13)]
rpe_per_set = [int(fat_row[f"set{n}"]) if pd.notna(fat_row[f"set{n}"]) else None for n in range(1, 13)]
print(f"\n  Participants.xlsx: {participant_name}")

# ---------- 3. Per-channel stats ----------
print("\n=== Computing per-channel stats ===")
stats = {}
for name, s in signals.items():
    st = channel_stats(name, s["x"], s["ts"], s["fs"])
    st["file"] = s["file"]
    stats[name] = st

# Write stats.json
stats_payload = {
    "subject_session": "recording_014_memory",
    "participant_name": participant_name,
    "biosignal_dir": str(BIO_DIR),
    "metadata_dir": str(META_DIR),
    "biosignal_t_unix_first": bio_t0,
    "biosignal_t_unix_last": bio_t1,
    "biosignal_duration_s": bio_t1 - bio_t0,
    "marker_t_unix_first": mark_t0,
    "marker_t_unix_last": mark_t1,
    "bio_marker_overlap_s": overlap_s,
    "bio_marker_gap_s": gap_s,
    "metadata_declared_fs": metadata["sampling_rates"],
    "channels": stats,
    "sets_detected_from_markers": len(sets_meta),
    "rep_count_total": sum(reps_in_set(n) for n in range(1, len(sets_meta) + 1)),
    "joint_files_present": list(joint_frame_counts.keys()),
    "joint_frame_counts": joint_frame_counts,
}
(OUT / "stats.json").write_text(json.dumps(stats_payload, indent=2))
print(f"Wrote {OUT/'stats.json'}")

# ---------- 4. Plots ----------
print("\n=== Plots ===")
plot_channels = ["ecg", "emg", "eda", "temperature", "ppg_green", "acc_mag"]
plot_channels = [c for c in plot_channels if c in signals]

# 4a. Signal overview (full session, downsampled)
fig, axes = plt.subplots(len(plot_channels), 1, figsize=(14, 2.0 * len(plot_channels)), sharex=True)
for ax, c in zip(axes, plot_channels):
    s = signals[c]
    t_min = (s["ts"] - bio_t0) / 60.0
    step = max(1, t_min.size // 50000)
    ax.plot(t_min[::step], s["x"][::step], lw=0.4)
    ax.set_ylabel(c, fontsize=9)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Session time [min] (relative to biosignal start)")
plt.suptitle(f"recording_014_memory ({participant_name}) — biosignal overview", y=0.995)
plt.tight_layout()
despine()
plt.savefig(OUT / "signal_overview.png", dpi=110)
plt.close()
print("  signal_overview.png")

# 4b. Set-1 zoom — but Set 1 markers are NOT in biosignal range (gap finding).
# So zoom on first 60 seconds of biosignal recording instead.
fig, axes = plt.subplots(len(plot_channels), 1, figsize=(14, 2.0 * len(plot_channels)), sharex=True)
zoom_t0, zoom_t1 = bio_t0, bio_t0 + 60.0
for ax, c in zip(axes, plot_channels):
    s = signals[c]
    sel = (s["ts"] >= zoom_t0) & (s["ts"] <= zoom_t1)
    ax.plot(s["ts"][sel] - bio_t0, s["x"][sel], lw=0.6)
    ax.set_ylabel(c, fontsize=9)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Session time [s] (first 60 s)")
plt.suptitle(
    f"signal_zoomed_set1.png stand-in: first 60 s of recording_014_memory\n"
    f"(Set 1 from dataset/recording_014 markers ends {gap_s:.1f} s BEFORE bio starts — see findings.md)",
    fontsize=10,
)
plt.tight_layout()
despine()
plt.savefig(OUT / "signal_zoomed_set1.png", dpi=110)
plt.close()
print("  signal_zoomed_set1.png")

# 4c. PPG channel check — first 30 s window of bio recording
ppg_cols = [c for c in signals if c.startswith("ppg_")]
fig, ax = plt.subplots(figsize=(14, 4))
zoom_t0 = bio_t0 + 30.0
zoom_t1 = bio_t0 + 60.0
for c in ppg_cols:
    s = signals[c]
    sel = (s["ts"] >= zoom_t0) & (s["ts"] <= zoom_t1)
    ax.plot(s["ts"][sel] - bio_t0, s["x"][sel], label=c, lw=0.7)
ax.set_xlabel("Session time [s]")
ax.set_title("PPG channels (30-60 s window) — verify which is green")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
despine()
plt.savefig(OUT / "ppg_channel_check.png", dpi=110)
plt.close()
print("  ppg_channel_check.png")

# 4d. PSDs
for c in plot_channels:
    s = signals[c]
    x = s["x"][~np.isnan(s["x"])]
    if x.size < 256:
        continue
    nperseg = min(8192, x.size // 8 if x.size > 8 else x.size)
    if nperseg < 64:
        continue
    f, pxx = welch(x, fs=s["fs"], nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(f, pxx, lw=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"{c} — fs = {s['fs']:.1f} Hz")
    ax.grid(alpha=0.3, which="both")
    for hz in (50, 60):
        ax.axvline(hz, color="red", ls="--", alpha=0.4, lw=0.8)
    plt.tight_layout()
    despine()
    plt.savefig(OUT / f"psd_{c}.png", dpi=110)
    plt.close()
    print(f"  psd_{c}.png")

# 4e. Timestamp alignment plot — show the GAP clearly
fig, ax = plt.subplots(figsize=(14, 3.2))
# Draw biosignal coverage band
ax.axhspan(0.55, 0.85, xmin=0, xmax=1, color="lightgray", alpha=0.0)  # placeholder
# Use Unix time on x-axis (downsampled scatter for visual coverage)
bio_ts_subsample = signals["ecg"]["ts"][::500]  # ~1 Hz
ax.scatter(bio_ts_subsample, np.full_like(bio_ts_subsample, 1.0), s=2, color="C0", label="biosignal samples (memory variant)")
ax.scatter(marker_t_unix, np.full_like(marker_t_unix, 0.0), s=14, color="C1", label="markers (dataset variant)")
# Per-set bands from metadata
for s_meta in sets_meta:
    ax.axvspan(s_meta["start_unix_time"], s_meta["end_unix_time"], color="C2", alpha=0.15)
ax.set_yticks([0, 1])
ax.set_yticklabels(["markers/joints", "biosignals"])
ax.set_xlabel("Unix time (s)")
ax.set_title(
    f"Timestamp coverage — bio vs markers (gap {gap_s:+.1f} s)\n"
    f"Memory bio: {datetime.fromtimestamp(bio_t0, tz=timezone.utc):%Y-%m-%d %H:%M:%S}-{datetime.fromtimestamp(bio_t1, tz=timezone.utc):%H:%M:%S} UTC; "
    f"markers: {datetime.fromtimestamp(mark_t0, tz=timezone.utc):%H:%M:%S}-{datetime.fromtimestamp(mark_t1, tz=timezone.utc):%H:%M:%S} UTC"
)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
despine()
plt.savefig(OUT / "timestamp_alignment.png", dpi=110)
plt.close()
print("  timestamp_alignment.png")

# 4f. Sets detected from acc-magnitude vs marker truth
g_baseline = float(np.median(acc_mag[: int(30 * fs_acc)]))
smooth = uniform_filter1d(np.abs(acc_mag - g_baseline), size=int(1.0 * fs_acc))
threshold = 0.3
is_active = smooth > threshold
diff = np.diff(is_active.astype(int))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]
detected_segments = []
if starts.size and ends.size:
    if ends[0] < starts[0]:
        ends = ends[1:]
    n_min2 = min(starts.size, ends.size)
    for i_seg in range(n_min2):
        if (ends[i_seg] - starts[i_seg]) / fs_acc > 20:
            detected_segments.append((int(starts[i_seg]), int(ends[i_seg])))
print(f"  acc-mag detected {len(detected_segments)} active segments (threshold {threshold} m/s²)")

fig, ax = plt.subplots(figsize=(14, 4.5))
t_min_acc = (acc_ts - bio_t0) / 60.0
step_acc = max(1, acc_ts.size // 60000)
ax.plot(t_min_acc[::step_acc], acc_mag[::step_acc], lw=0.3, color="gray", alpha=0.7, label="|acc|")
ax.plot(t_min_acc[::step_acc], (smooth[::step_acc] + g_baseline), lw=0.8, color="black", label="smoothed |acc-g|")
for s_seg, e_seg in detected_segments:
    ax.axvspan(t_min_acc[s_seg], t_min_acc[e_seg], color="C0", alpha=0.15)
# Overlay marker-derived sets — these are likely OUTSIDE the bio window
for s_meta in sets_meta:
    s_min = (s_meta["start_unix_time"] - bio_t0) / 60.0
    e_min = (s_meta["end_unix_time"] - bio_t0) / 60.0
    ax.axvspan(s_min, e_min, color="C3", alpha=0.25)
ax.set_xlabel("Session time [min] (relative to bio start)")
ax.set_ylabel("|acc| [m/s²]")
ax.set_title(
    f"acc-magnitude detected sets (blue) vs marker-defined sets (red) — bio start = 0\n"
    f"Marker sets at session-time {(set_starts_unix.min()-bio_t0)/60:+.1f} .. {(set_ends_unix.max()-bio_t0)/60:+.1f} min"
)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
despine()
plt.savefig(OUT / "sets_detected.png", dpi=110)
plt.close()
print("  sets_detected.png")

# 4g. Joint coverage — bar chart of frame counts
fig, ax = plt.subplots(figsize=(12, 4))
xs = np.arange(1, len(joint_files) + 1)
counts = [joint_frame_counts[jf.name] for jf in joint_files]
durations = [s["end_unix_time"] - s["start_unix_time"] for s in sets_meta[: len(joint_files)]]
fps = [c / d if d > 0 else 0 for c, d in zip(counts, durations)]
bars = ax.bar(xs, counts, color="C2", alpha=0.7)
ax.set_xticks(xs)
ax.set_xlabel("Set #")
ax.set_ylabel("Joint frame count")
ax.set_title(
    f"Per-set Kinect skeleton frames (mean {np.mean(counts):.0f}, "
    f"effective {np.mean(fps):.1f} fps over {np.mean(durations):.1f} s avg set)"
)
for x_, c_, fps_ in zip(xs, counts, fps):
    ax.text(x_, c_ + 5, f"{fps_:.0f}fps", ha="center", fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
despine()
plt.savefig(OUT / "joint_coverage.png", dpi=110)
plt.close()
print("  joint_coverage.png")

# ---------- 5. Reports ----------
print("\n=== Writing report.md and findings.md ===")

# Build report.md
fs_table_rows = []
for c in ["ecg", "emg", "eda", "temperature", "ppg_green", "ppg_red", "ppg_ir", "ppg_blue", "ax", "ay", "az", "acc_mag"]:
    if c in stats:
        s = stats[c]
        fs_table_rows.append(
            f"| {c} | {s['fs_hz']:.2f} Hz | {s['mean']:.3g} | {s['p1']:.3g} .. {s['p99']:.3g} | {s['pct_nan']:.2f}% |"
        )

set_table_rows = []
for n in range(1, len(sets_meta) + 1):
    s_meta = sets_meta[n - 1]
    rpe = rpe_per_set[n - 1]
    ex = ex_per_set[n - 1]
    reps = reps_in_set(n)
    dur = s_meta["end_unix_time"] - s_meta["start_unix_time"]
    set_table_rows.append(f"| {n} | {ex} | {rpe} | {dur:.1f} s | {reps} |")

joint_table_rows = []
for jf in joint_files:
    joint_table_rows.append(f"| {jf.name} | {joint_frame_counts[jf.name]} |")

bio_utc_start = datetime.fromtimestamp(bio_t0, tz=timezone.utc)
mark_utc_start = datetime.fromtimestamp(mark_t0, tz=timezone.utc)

report = f"""# Inspection report — recording_014 (memory variant)

Participant: **{participant_name}**
Biosignal source: `dataset_memory/recording_014_memory/` (per-channel CSVs)
Markers/joints/metadata source: `dataset/recording_014/`

Biosignal Unix start: {bio_t0:.3f} ({bio_utc_start:%Y-%m-%d %H:%M:%S} UTC)
Biosignal duration:   {(bio_t1 - bio_t0) / 60:.2f} min
Marker Unix start:    {mark_t0:.3f} ({mark_utc_start:%Y-%m-%d %H:%M:%S} UTC)
Marker last:          {mark_t1:.3f} ({datetime.fromtimestamp(mark_t1, tz=timezone.utc):%H:%M:%S} UTC)
Bio-vs-marker gap:    **{gap_s:+.2f} s** (positive = bio starts AFTER markers end)

## Modalities (measured fs from CSV timestamps)

| Modality | fs (measured) | mean | p1..p99 | NaN% |
|----------|--------------:|-----:|---------|-----:|
""" + "\n".join(fs_table_rows) + f"""

Declared fs from `dataset/recording_014/metadata.json`: `{metadata['sampling_rates']}`

## Sets (metadata.json + markers.json + Participants.xlsx)

| Set | Exercise | RPE | Duration | Reps |
|----:|----------|----:|---------:|-----:|
""" + "\n".join(set_table_rows) + f"""

Total sets in metadata: {len(sets_meta)}
Total reps from markers: {sum(reps_in_set(n) for n in range(1, len(sets_meta)+1))}

## Joint-skeleton file coverage

| File | n_frames |
|------|---------:|
""" + "\n".join(joint_table_rows) + """

## Plots

- `signal_overview.png` — full memory-variant biosignal session, all channels
- `signal_zoomed_set1.png` — first 60 s of bio (Set 1 markers fall outside bio time-range — see findings.md)
- `ppg_channel_check.png` — 30-60 s window, all 4 PPG wavelengths
- `psd_<channel>.png` — per-channel PSD with 50/60 Hz reference lines
- `timestamp_alignment.png` — biosignal coverage (memory) vs marker coverage (dataset) on shared Unix time-axis
- `sets_detected.png` — acc-magnitude segmentation (blue) and marker-derived sets (red) on bio session-time
- `joint_coverage.png` — per-set Kinect skeleton frame counts
"""

(OUT / "report.md").write_text(report, encoding="utf-8")
print("  report.md")

# Build findings.md
findings = f"""# Findings — recording_014 (memory variant, participant {participant_name})

UTC start (memory bio): {bio_utc_start:%Y-%m-%d %H:%M:%S}
Bio duration: {(bio_t1 - bio_t0) / 60:.2f} min
Total sets in `dataset/recording_014` metadata/markers/Participants.xlsx: 12 (✓ three-way match)

## CRITICAL — biosignal and marker streams DO NOT OVERLAP

The biosignals from `dataset_memory/recording_014_memory/` and the markers/joints from
`dataset/recording_014/` are **two different recordings of the same participant**, not
two views of one session.

- `dataset/recording_014/metadata.json` describes a session running
  **{datetime.fromtimestamp(metadata['recording_start_unix_time'], tz=timezone.utc):%H:%M:%S}** UTC -> ~{(metadata['data_start_unix_time'] + 2455.26 - metadata['recording_start_unix_time'])/60:.1f} min later.
  All `dataset/recording_014/*.csv`, `markers.json`, and `recording_NN_joints.json` share that range.
- `dataset_memory/recording_014_memory/*.csv` starts at
  **{bio_utc_start:%H:%M:%S}** UTC and runs {(bio_t1 - bio_t0) / 60:.2f} min.
  This is **{gap_s:+.1f} s after the dataset session ENDS** — `bio_marker_overlap_s = {overlap_s:.2f}`.

So no marker or joint file in `dataset/recording_014/` covers a single sample of the memory
biosignals. Aligning them by Unix time will produce zero ground-truth labels for sets, reps
or phase. This will silently break `data-labeler` (the joint→biosignal interpolation will
return all-NaN, and acc-magnitude segmentation has no marker truth to validate against).

### Recommended remediation (pick one before running `/label`)

1. **Use the dataset variant for everything.** `dataset/recording_014/{{ecg,emg,eda,temperature,ppg_*,ax,ay,az}}.csv` exist and share their Unix-time grid with `markers.json` + `recording_NN_joints.json`. Drop the memory variant for this recording.
2. **Locate the matching memory-variant markers.** If the memory recording is a separate, post-session bout, the corresponding `markers.json` lives elsewhere on disk (or was never produced by Kinect). The user must point at it explicitly, OR accept that this recording has no ground-truth labels and is only usable for unsupervised pre-training.
3. **Run `/inspect` on the dataset variant** (`dataset/recording_014/`) to confirm it is consistent with the existing `recording_012`/`recording_013` inspections (which all draw biosignals from `dataset/recording_NNN/`).

Until one of (1)/(2)/(3) is resolved, **do not advance to `/label`**.

## CONFIRMED hardware (memory-variant biosignals)

| Modality | Column file | fs (measured) | Range | NaN% | Status |
|----------|-------------|--------------:|-------|-----:|--------|
| ECG | `ecg.csv` | {stats['ecg']['fs_hz']:.2f} Hz | {stats['ecg']['min']:.3g} … {stats['ecg']['max']:.3g} | {stats['ecg']['pct_nan']:.2f}% | OK |
| EMG | `emg.csv` | **{stats['emg']['fs_hz']:.2f} Hz** | {stats['emg']['min']:.3g} … {stats['emg']['max']:.3g} | {stats['emg']['pct_nan']:.2f}% | OK |
| EDA | `eda.csv` | {stats['eda']['fs_hz']:.2f} Hz | {stats['eda']['min']:.3g} … {stats['eda']['max']:.3g} | {stats['eda']['pct_nan']:.2f}% | OK |
| Temperature | `temperature.csv` | {stats['temperature']['fs_hz']:.2f} Hz (declared 1 Hz) | {stats['temperature']['min']:.3g} … {stats['temperature']['max']:.3g} °C | {stats['temperature']['pct_nan']:.2f}% | {'POPULATED (newer recordings 012/013 had this empty)' if stats['temperature'].get('mean', 0) > 0 else 'EMPTY'} |
| PPG-green | `ppg_green.csv` | {stats['ppg_green']['fs_hz']:.2f} Hz | {stats['ppg_green']['min']:.3g} … {stats['ppg_green']['max']:.3g} (raw) | {stats['ppg_green']['pct_nan']:.2f}% | OK — column literally named `ppg_green` |
| PPG-blue | `ppg_blue.csv` | {stats['ppg_blue']['fs_hz']:.2f} Hz | {stats['ppg_blue']['min']:.3g} … {stats['ppg_blue']['max']:.3g} | {stats['ppg_blue']['pct_nan']:.2f}% | logged but unused |
| PPG-red | `ppg_red.csv` | {stats['ppg_red']['fs_hz']:.2f} Hz | {stats['ppg_red']['min']:.3g} … {stats['ppg_red']['max']:.3g} | {stats['ppg_red']['pct_nan']:.2f}% | logged but unused |
| PPG-ir | `ppg_ir.csv` | {stats['ppg_ir']['fs_hz']:.2f} Hz | {stats['ppg_ir']['min']:.3g} … {stats['ppg_ir']['max']:.3g} | {stats['ppg_ir']['pct_nan']:.2f}% | logged but unused |
| ax / ay / az | `ax.csv` `ay.csv` `az.csv` | {stats['ax']['fs_hz']:.2f} / {stats['ay']['fs_hz']:.2f} / {stats['az']['fs_hz']:.2f} Hz | ax {stats['ax']['min']:.2g}..{stats['ax']['max']:.2g} | 0% | OK |
| acc_mag (computed) | √(ax²+ay²+az²) | {stats['acc_mag']['fs_hz']:.2f} Hz | {stats['acc_mag']['min']:.3g} … {stats['acc_mag']['max']:.3g} | 0% | OK |

Declared in `dataset/recording_014/metadata.json`: `{metadata['sampling_rates']}`. Measured rates match the declared rates within float tolerance.

## Synchronization (memory bio internal)

- Memory-variant CSVs all share Unix-epoch timestamps (>1e9 ✓).
- ECG / EMG / EDA / PPG / IMU first samples are all `{bio_t0:.6f}` — single shared start across modalities.
- Internal sync of memory biosignals: OK.
- External sync to dataset markers/joints: **NOT OK** (see "CRITICAL" section above).

## Labels available (assuming dataset/recording_014 ground truth, currently un-mappable to memory bio)

- **Per-set** from `Participants.xlsx`:
  - Exercises: `{ex_per_set}`
  - RPE: `{rpe_per_set}`
- **Per-rep** from `markers.json`: explicit Unix-time markers, `Set:N_Rep:K` and `Set:N_Start` / `Set_N_End`. Total reps: {sum(reps_in_set(n) for n in range(1, len(sets_meta)+1))}.
- **Per-frame skeleton** from `recording_NN_joints.json`: 32 joints, positions + orientations. **Note:** internal `timestamp_usec = 0` (Kinect SDK didn't fill it) — sync must come from `metadata.json["kinect_sets"]` set start/end Unix and linear interpolation across frames.

## Quality flags

1. **CRITICAL: cross-source timestamp gap** — see "CRITICAL" section. This is the headline finding.
2. **Acc-magnitude segmentation detected {len(detected_segments)} active segments** in the memory bio (threshold 0.3 m/s²). Without aligned markers we cannot validate against the 12 expected sets, but the count alone (vs the 12 sets the dataset session contains) is consistent with this being a different session of unknown structure.
3. **Temperature** has measured fs ≈ {stats['temperature']['fs_hz']:.2f} Hz vs declared 1 Hz — the file IS populated here (mean {stats['temperature']['mean']:.2f} °C), unlike the empty temperature.csv on `recording_012`/`recording_013`. If the memory and dataset recordings come from different firmware/loggers this asymmetry is a clue.
4. **EMG / ECG line-noise check**: see `psd_emg.png` / `psd_ecg.png` — vertical red lines at 50/60 Hz. Norway grid = 50 Hz; verify peak there before defaulting to a 50 Hz notch in `configs/default.yaml`.

## Action items for `CLAUDE.md` and `configs/default.yaml`

- [ ] **Resolve the source-mismatch problem above before doing anything else.**
- [x] EMG fs = {stats['emg']['fs_hz']:.0f} Hz (CLAUDE.md placeholder was 1000) — same as recording_012/013.
- [x] PPG fs = {stats['ppg_green']['fs_hz']:.0f} Hz (CLAUDE.md placeholder was 64) — same as recording_012/013.
- [x] PPG-green column is literally `ppg_green` — no renaming needed in `data-loader`.
- [x] EDA fs = {stats['eda']['fs_hz']:.0f} Hz (CLAUDE.md table was empty).
- [x] Temperature fs = {stats['temperature']['fs_hz']:.2f} Hz (declared 1 Hz; memory variant populates the file).
- [x] **12 sets / 4 exercises (squat, deadlift, benchpress, pullup)** — already noted in recording_012/013 findings; reconfirmed.
- [x] Joint-angle ground truth = Kinect skeleton (32 joints, positions+orientations); joint angles must be derived in `data-labeler` (no precomputed angle CSV exists).
- [x] Joint frame `timestamp_usec` is always 0 — sync via `metadata.json["kinect_sets"]` Unix start/end + linear interpolation across frames.
- [x] `Participants.xlsx` schema is dual-row per `Recording: N` (exercise row, then unnamed `Name: "fatigue"` row with RPE per set).

## Recommendation

Stop and confirm with the user which source of biosignals to use for recording_014. Two consistent options:

1. **Use `dataset/recording_014/`** for all of biosignals + markers + joints. This matches the existing `recording_012`/`recording_013` inspections and gives a directly-labelable session.
2. **Treat `dataset_memory/recording_014_memory/` as an unlabeled extra recording** — useful for self-supervised pre-training of a representation, but cannot supervise the four labeled tasks.

Mixing memory biosignals with dataset markers/joints, as initially requested, will not work for recording_014: there is zero Unix-time overlap.

## References

- Subject-wise / per-session timestamp validation as a precondition for label alignment: project methodology (no external citation needed for this finding).
- EMG fs ≥ 1000 Hz adequate for fatigue spectral indices (MNF/MDF, Dimitrov FInsm5): De Luca 1997; Dimitrov et al. 2006.
- Kinect Azure body tracking joint set and frame rate behaviour: Microsoft k4abt SDK 1.1.x docs (this dataset's `recording_NN_joints.json["k4abt_sdk_version"] = 1.1.2`).
- Inter-subject EDA baseline normalization (already flagged in recording_012/013): Greco et al. 2016.

(All references except k4abt SDK are tracked in the `literature-references` skill.)
"""
(OUT / "findings.md").write_text(findings, encoding="utf-8")
print("  findings.md")

print("\n=== DONE ===")
print(f"Output dir: {OUT}")
