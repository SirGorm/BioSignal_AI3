"""
Inspect a single recording directory.

Adapted to the actual dataset layout:
  dataset/recording_NNN/
    {ecg,emg,eda,temperature,ppg_blue,ppg_green,ppg_red,ppg_ir,ax,ay,az}.csv
    a_combined.csv
    metadata.json
    markers.json
    recording_NN_joints.json   (one per set; raw Kinect skeleton)
  dataset/Participants/Participants.xlsx

Usage:
  python scripts/inspect_recording.py 012
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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

ROOT = Path("c:/MasterProject/Code_v1/strength-rt-v2")
DATASET = ROOT / "dataset"
PARTICIPANTS_XLSX = DATASET / "Participants" / "Participants.xlsx"

MODALITIES = {
    "ecg": "ecg.csv",
    "emg": "emg.csv",
    "eda": "eda.csv",
    "temperature": "temperature.csv",
    "ppg_green": "ppg_green.csv",
    "ppg_blue": "ppg_blue.csv",
    "ppg_red": "ppg_red.csv",
    "ppg_ir": "ppg_ir.csv",
    "ax": "ax.csv",
    "ay": "ay.csv",
    "az": "az.csv",
}


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or len(df.columns) < 2:
        return None
    return df


def fs_from_timestamps(t: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(t)))


def channel_stats(name: str, t: np.ndarray, x: np.ndarray) -> dict:
    fs = fs_from_timestamps(t)
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return {"name": name, "status": "empty", "fs_hz": fs}
    abs_max = float(np.max(np.abs(valid))) if np.max(np.abs(valid)) > 0 else 1.0
    return {
        "name": name,
        "fs_hz": fs,
        "n_samples": int(len(valid)),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "t_unix_start": float(t[0]) if len(t) else float("nan"),
        "t_unix_end": float(t[-1]) if len(t) else float("nan"),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p1": float(np.percentile(valid, 1)),
        "p99": float(np.percentile(valid, 99)),
        "n_nan": int(np.sum(np.isnan(x))),
        "pct_nan": float(np.mean(np.isnan(x)) * 100),
        "pct_at_clip": float(np.mean(np.abs(valid) >= abs_max * 0.999) * 100),
    }


def participants_lookup(rec_num: int) -> dict:
    df = pd.read_excel(PARTICIPANTS_XLSX)
    idx = df.index[df["Recording:"] == rec_num].tolist()
    if not idx:
        return {"name": None, "exercises": [], "rpe": []}
    i = idx[0]
    return {
        "name": str(df.iloc[i]["Name:"]),
        "exercises": [str(v) for v in df.iloc[i, 2:].tolist()],
        "rpe": [int(v) if pd.notna(v) else None for v in df.iloc[i + 1, 2:].tolist()],
    }


def parse_markers(markers_path: Path) -> tuple[list[dict], list[dict]]:
    with open(markers_path) as f:
        data = json.load(f)
    sets, reps = [], []
    cur_set = None
    for m in data["markers"]:
        label = m["label"]
        if label.endswith("_Start") and "Set:" in label:
            n = int(label.split(":")[1].split("_")[0])
            cur_set = {"set_number": n, "start_unix": m["unix_time"], "end_unix": None, "reps": []}
        elif label.startswith("Set_") and label.endswith("_End"):
            n = int(label.split("_")[1])
            if cur_set and cur_set["set_number"] == n:
                cur_set["end_unix"] = m["unix_time"]
                sets.append(cur_set)
                cur_set = None
        elif "_Rep:" in label:
            rep_n = int(label.split("Rep:")[1])
            set_n = int(label.split(":")[1].split("_")[0])
            reps.append({"set_number": set_n, "rep": rep_n, "unix": m["unix_time"]})
            if cur_set and cur_set["set_number"] == set_n:
                cur_set["reps"].append(m["unix_time"])
    return sets, reps


def detect_active_segments_from_acc(t: np.ndarray, mag: np.ndarray, fs: float) -> list[tuple[float, float]]:
    if len(t) < int(60 * fs):
        return []
    baseline = np.median(mag[: int(30 * fs)])
    smooth = uniform_filter1d(np.abs(mag - baseline), size=max(1, int(1.0 * fs)))
    is_active = smooth > 0.3
    diff = np.diff(is_active.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return []
    if ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    segs = []
    for i in range(n):
        s_t, e_t = t[starts[i]], t[ends[i]]
        if e_t - s_t > 10:
            segs.append((s_t, e_t))
    return segs


def joint_coverage(rec_dir: Path, sets: list[dict]) -> list[dict]:
    """For each set, look up the corresponding recording_NN_joints.json and report frame count."""
    coverage = []
    for s in sets:
        # joint files are 1-indexed and may not match set number directly
        # Try multiple possible filenames
        candidates = [
            rec_dir / f"recording_{s['set_number']:02d}_joints.json",
        ]
        joint_file = next((p for p in candidates if p.exists()), None)
        if joint_file is None:
            coverage.append({"set_number": s["set_number"], "joint_file": None, "n_frames": 0})
            continue
        try:
            with open(joint_file) as f:
                jd = json.load(f)
            n_frames = len(jd.get("frames", []))
        except Exception:
            n_frames = 0
        coverage.append(
            {"set_number": s["set_number"], "joint_file": joint_file.name, "n_frames": n_frames}
        )
    return coverage


def inspect(rec_id: str) -> None:
    rec_dir = DATASET / f"recording_{rec_id}"
    if not rec_dir.exists():
        raise SystemExit(f"Recording dir not found: {rec_dir}")

    out_dir = ROOT / "inspections" / f"recording_{rec_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Inspecting {rec_dir.name} ===")
    print(f"Output: {out_dir}")

    # Metadata
    with open(rec_dir / "metadata.json") as f:
        meta = json.load(f)
    declared_fs = meta["sampling_rates"]
    print(f"Declared sample rates: {declared_fs}")

    # Participants
    rec_num = int(rec_id)
    participants = participants_lookup(rec_num)
    print(f"Participant: {participants['name']}")

    # Markers (rep ground truth)
    sets, reps = parse_markers(rec_dir / "markers.json")
    print(f"Markers: {len(sets)} sets, {len(reps)} reps total")

    # Load each modality at native rate
    channels: dict[str, dict] = {}
    for mod, fname in MODALITIES.items():
        df = load_csv(rec_dir / fname)
        if df is None:
            channels[mod] = {"present": False}
            continue
        t = df.iloc[:, 0].values.astype(float)
        x = df.iloc[:, 1].values.astype(float)
        channels[mod] = {"present": True, "t": t, "x": x, "df": df}
    present = [m for m, v in channels.items() if v.get("present")]
    print(f"Channels present: {present}")

    # Stats
    stats = {"recording": rec_dir.name, "participant": participants["name"], "channels": {}}
    for mod, v in channels.items():
        if not v.get("present"):
            stats["channels"][mod] = {"name": mod, "status": "empty_or_missing"}
            continue
        stats["channels"][mod] = channel_stats(mod, v["t"], v["x"])

    # Verify Unix timestamps on present modalities
    for mod, v in channels.items():
        if not v.get("present"):
            continue
        t0 = v["t"][0]
        if t0 < 1e9:
            print(f"  WARNING: {mod} timestamps look NOT-Unix (first={t0})")

    # Compute acc magnitude (uses ax/ay/az aligned to common timestamps)
    if all(channels[c].get("present") for c in ["ax", "ay", "az"]):
        ax_t = channels["ax"]["t"]
        ax_v = channels["ax"]["x"]
        ay_v = channels["ay"]["x"]
        az_v = channels["az"]["x"]
        n = min(len(ax_v), len(ay_v), len(az_v))
        acc_mag = np.sqrt(ax_v[:n] ** 2 + ay_v[:n] ** 2 + az_v[:n] ** 2)
        acc_t = ax_t[:n]
        fs_acc = fs_from_timestamps(acc_t)
        stats["channels"]["acc_mag"] = channel_stats("acc_mag", acc_t, acc_mag)
    else:
        acc_mag, acc_t, fs_acc = None, None, float("nan")

    # Detect sets from acc-magnitude
    acc_segments = []
    if acc_mag is not None:
        acc_segments = detect_active_segments_from_acc(acc_t, acc_mag, fs_acc)
    print(f"Acc-mag detected segments: {len(acc_segments)} (markers say {len(sets)})")

    # Joint coverage (per set)
    j_cov = joint_coverage(rec_dir, sets)

    # ----- Plots -----
    bio_t0 = channels["ecg"]["t"][0] if channels["ecg"].get("present") else (
        meta.get("data_start_unix_time") or 0.0
    )

    # 1) signal_overview.png
    plot_channels = [m for m in ["ecg", "emg", "eda", "temperature", "ppg_green"] if channels.get(m, {}).get("present")]
    if acc_mag is not None:
        plot_channels.append("acc_mag")
    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(14, 1.8 * len(plot_channels)), sharex=True)
    if len(plot_channels) == 1:
        axes = [axes]
    for ax, mod in zip(axes, plot_channels):
        if mod == "acc_mag":
            t_plot = (acc_t - bio_t0) / 60
            v_plot = acc_mag
        else:
            v = channels[mod]
            t_plot = (v["t"] - bio_t0) / 60
            v_plot = v["x"]
        step = max(1, len(t_plot) // 80_000)
        ax.plot(t_plot[::step], v_plot[::step], lw=0.4)
        ax.set_ylabel(mod)
        ax.grid(alpha=0.3)
        # mark sets
        for s in sets:
            ax.axvspan(
                (s["start_unix"] - bio_t0) / 60,
                (s["end_unix"] - bio_t0) / 60,
                color="C2",
                alpha=0.15,
            )
    axes[-1].set_xlabel("Session time [min]")
    plt.suptitle(f"{rec_dir.name} — full session ({participants['name']})")
    plt.tight_layout()
    despine()
    plt.savefig(out_dir / "signal_overview.png", dpi=110)
    plt.close()

    # 2) signal_zoomed_set1.png
    if sets:
        s1 = sets[0]
        win = (s1["start_unix"] - 5, s1["end_unix"] + 5)
        fig, axes = plt.subplots(len(plot_channels), 1, figsize=(12, 1.6 * len(plot_channels)), sharex=True)
        if len(plot_channels) == 1:
            axes = [axes]
        for ax, mod in zip(axes, plot_channels):
            if mod == "acc_mag":
                t = acc_t
                v = acc_mag
            else:
                t = channels[mod]["t"]
                v = channels[mod]["x"]
            mask = (t >= win[0]) & (t <= win[1])
            ax.plot(t[mask] - s1["start_unix"], v[mask], lw=0.6)
            ax.set_ylabel(mod)
            ax.grid(alpha=0.3)
            ax.axvspan(0, s1["end_unix"] - s1["start_unix"], color="C2", alpha=0.15)
            for r in s1.get("reps", []):
                ax.axvline(r - s1["start_unix"], color="red", ls="--", alpha=0.4, lw=0.6)
        axes[-1].set_xlabel(f"Time relative to set 1 start [s]  (set duration {s1['end_unix'] - s1['start_unix']:.1f}s)")
        plt.suptitle(f"{rec_dir.name} — set 1 zoom (red lines = reps)")
        plt.tight_layout()
        despine()
        plt.savefig(out_dir / "signal_zoomed_set1.png", dpi=110)
        plt.close()

    # 3) ppg_channel_check.png
    ppg_mods = [m for m in ["ppg_green", "ppg_blue", "ppg_red", "ppg_ir"] if channels.get(m, {}).get("present")]
    if len(ppg_mods) >= 2:
        fig, ax = plt.subplots(figsize=(14, 4))
        # 30 s window from minute 1 (resting period)
        ref_t0 = bio_t0 + 60
        ref_t1 = ref_t0 + 30
        for mod in ppg_mods:
            t = channels[mod]["t"]
            v = channels[mod]["x"]
            mask = (t >= ref_t0) & (t <= ref_t1)
            ax.plot(t[mask] - ref_t0, v[mask], label=mod, lw=0.7)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("PPG raw")
        ax.set_title(f"PPG channels (30s rest @ ~min 1) — {rec_dir.name}")
        ax.legend()
        plt.tight_layout()
        despine()
        plt.savefig(out_dir / "ppg_channel_check.png", dpi=110)
        plt.close()

    # 4) PSDs
    for mod in plot_channels:
        if mod == "acc_mag":
            x = acc_mag
            fs_c = fs_acc
        else:
            x = channels[mod]["x"]
            fs_c = stats["channels"][mod]["fs_hz"]
        x = x[~np.isnan(x)]
        if len(x) < 1024 or not np.isfinite(fs_c):
            continue
        f, pxx = welch(x, fs=fs_c, nperseg=min(8192, len(x) // 4))
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.semilogy(f, pxx)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")
        ax.set_title(f"{mod} — fs ≈ {fs_c:.1f} Hz")
        for hz in [50, 60]:
            if hz < fs_c / 2:
                ax.axvline(hz, color="red", ls="--", alpha=0.4, lw=0.8)
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        despine()
        plt.savefig(out_dir / f"psd_{mod}.png", dpi=110)
        plt.close()

    # 5) timestamp_alignment.png — biosignal coverage vs joint files
    fig, ax = plt.subplots(figsize=(14, 3))
    if channels["ecg"].get("present"):
        t = channels["ecg"]["t"]
        ax.scatter((t[::5000] - bio_t0) / 60, np.ones(len(t[::5000])) * 1, s=2, alpha=0.5, label="ecg sampled")
    # set boundaries from markers
    for s in sets:
        ax.axvspan((s["start_unix"] - bio_t0) / 60, (s["end_unix"] - bio_t0) / 60,
                   color="C2", alpha=0.2)
    ax.set_yticks([1])
    ax.set_yticklabels(["bio"])
    ax.set_xlabel("Session time [min]")
    ax.set_title(f"Timestamp coverage (Unix anchored)  — bio start {datetime.fromtimestamp(bio_t0, tz=timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    plt.tight_layout()
    despine()
    plt.savefig(out_dir / "timestamp_alignment.png", dpi=110)
    plt.close()

    # 6) sets_detected.png
    if acc_mag is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        step = max(1, len(acc_t) // 80_000)
        ax.plot((acc_t[::step] - bio_t0) / 60, acc_mag[::step], lw=0.3, color="gray", alpha=0.6, label="|acc|")
        # marker-derived sets
        for s in sets:
            ax.axvspan((s["start_unix"] - bio_t0) / 60, (s["end_unix"] - bio_t0) / 60,
                       color="C2", alpha=0.25, label="markers set" if s["set_number"] == 1 else None)
        # acc-segmented
        for i, (s, e) in enumerate(acc_segments):
            ax.axvspan((s - bio_t0) / 60, (e - bio_t0) / 60,
                       color="C1", alpha=0.15, label="acc-detected" if i == 0 else None)
        ax.set_xlabel("Session time [min]")
        ax.set_ylabel("|acc| [m/s²]")
        ax.set_title(f"Sets: markers={len(sets)}, acc-detected={len(acc_segments)}")
        ax.legend(loc="upper right")
        plt.tight_layout()
        despine()
        plt.savefig(out_dir / "sets_detected.png", dpi=110)
        plt.close()

    # 7) joint_coverage.png
    fig, ax = plt.subplots(figsize=(14, 3))
    for s in sets:
        ax.axvspan((s["start_unix"] - bio_t0) / 60, (s["end_unix"] - bio_t0) / 60,
                   color="C2", alpha=0.2)
    for cov, s in zip(j_cov, sets):
        n_f = cov["n_frames"]
        cx = ((s["start_unix"] + s["end_unix"]) / 2 - bio_t0) / 60
        ax.text(cx, 0.5, f"{n_f} frames\n{cov['joint_file'] or 'MISSING'}",
                ha="center", va="center", fontsize=7,
                color="black" if cov["joint_file"] else "red")
    ax.set_xlabel("Session time [min]")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Joint-skeleton file coverage per set — {rec_dir.name}")
    plt.tight_layout()
    despine()
    plt.savefig(out_dir / "joint_coverage.png", dpi=110)
    plt.close()

    # ----- stats.json -----
    stats["sets_from_markers"] = len(sets)
    stats["sets_from_acc"] = len(acc_segments)
    stats["sets_from_metadata"] = meta.get("total_kinect_sets")
    stats["sets_from_participants"] = sum(1 for x in participants["exercises"] if isinstance(x, str) and x != "nan")
    stats["bio_t0_unix"] = bio_t0
    stats["bio_duration_s"] = float(channels["ecg"]["t"][-1] - bio_t0) if channels["ecg"].get("present") else None
    stats["joint_coverage"] = j_cov
    stats["sets"] = [{"set_number": s["set_number"], "start_unix": s["start_unix"],
                      "end_unix": s["end_unix"], "duration_s": s["end_unix"] - s["start_unix"],
                      "n_reps": len(s["reps"])} for s in sets]
    stats["participants"] = participants

    # serialize-safe
    def _safe(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=_safe)

    # ----- report.md -----
    bio_start_dt = datetime.fromtimestamp(bio_t0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    duration_min = (stats["bio_duration_s"] or 0) / 60

    rows_hw = []
    for mod in ["ecg", "emg", "eda", "temperature", "ppg_green", "ax", "ay", "az"]:
        cs = stats["channels"].get(mod, {})
        if cs.get("status") == "empty_or_missing":
            rows_hw.append(f"| {mod} | empty/missing | — | — | — |")
        else:
            rows_hw.append(
                f"| {mod} | {cs.get('fs_hz', float('nan')):.1f} Hz | {cs.get('mean', 0):.3g} | "
                f"{cs.get('p1', 0):.3g} .. {cs.get('p99', 0):.3g} | {cs.get('pct_nan', 0):.2f}% |"
            )

    set_rows = []
    for s, ex, rpe in zip(stats["sets"], participants["exercises"], participants["rpe"]):
        set_rows.append(
            f"| {s['set_number']} | {ex} | {rpe} | {s['duration_s']:.1f} s | {s['n_reps']} |"
        )

    j_rows = []
    for cov in j_cov:
        j_rows.append(f"| {cov['set_number']} | {cov['joint_file'] or 'MISSING'} | {cov['n_frames']} |")

    report = f"""# Inspection report — {rec_dir.name}

Participant: **{participants['name']}**
Recording UTC start: {bio_start_dt}
Duration: {duration_min:.1f} min
Total sets (metadata.json): {meta.get('total_kinect_sets')}
Total sets (markers.json):  {len(sets)}
Total sets (Participants.xlsx): {stats['sets_from_participants']}

## Modalities (measured fs from CSV timestamps)

| Modality | fs (measured) | mean | p1..p99 | NaN% |
|----------|--------------:|-----:|---------|-----:|
{chr(10).join(rows_hw)}

Declared fs from metadata.json: `{declared_fs}`

## Sets (markers.json + Participants.xlsx)

| Set | Exercise | RPE | Duration | Reps |
|----:|----------|----:|---------:|-----:|
{chr(10).join(set_rows)}

## Joint-skeleton file coverage (1 file per set)

| Set | Joint file | n_frames |
|----:|------------|---------:|
{chr(10).join(j_rows)}

## Plots

- `signal_overview.png` — all channels with set windows shaded
- `signal_zoomed_set1.png` — set 1 zoom, red dashed = rep markers
- `ppg_channel_check.png` — 30 s rest window, all 4 PPG wavelengths
- `psd_<channel>.png` — per-channel PSD with 50/60 Hz lines marked
- `timestamp_alignment.png` — biosignal Unix-time coverage
- `sets_detected.png` — markers vs acc-magnitude segmentation
- `joint_coverage.png` — per-set Kinect skeleton frame counts
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote {out_dir / 'stats.json'}")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Plots: {sorted(p.name for p in out_dir.glob('*.png'))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_recording.py <NNN> [<NNN> ...]")
        sys.exit(1)
    for rec_id in sys.argv[1:]:
        inspect(rec_id)
