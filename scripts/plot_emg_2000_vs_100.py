"""Plot raw 2000 Hz EMG vs 100 Hz RMS-envelope for one full set.

Uses the same envelope pipeline as src/labeling/align.py so the plot reflects
what the aligned parquet (and all raw NN models) actually see.

Usage
-----
    python scripts/plot_emg_2000_vs_100.py [recording] [set_number] [pad_s]

Defaults: recording_012, set 1, pad 1.0 s before/after kinect_set boundaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labeling.align import emg_envelope, make_100hz_grid, FS_EMG_NATIVE


def main(recording: str = "recording_012", set_number: int = 1, pad_s: float = 1.0) -> None:
    rec_dir = ROOT / "dataset_aligned" / recording
    emg_csv = rec_dir / "emg.csv"
    meta = json.loads((rec_dir / "metadata.json").read_text())

    sets = {int(s["set_number"]): s for s in meta["kinect_sets"]}
    if set_number not in sets:
        raise SystemExit(f"set {set_number} not in {sorted(sets)}")
    s = sets[set_number]
    set_start = float(s["start_unix_time"])
    set_end = float(s["end_unix_time"])

    df = pd.read_csv(emg_csv)
    t_raw = df["timestamp"].to_numpy(dtype=float)
    x_raw = df["emg"].to_numpy(dtype=float)

    # Same envelope pipeline as align.py (BP 20-450 + 50 Hz notch + 50 ms RMS)
    env_2000 = emg_envelope(x_raw, fs=FS_EMG_NATIVE)
    grid_t = make_100hz_grid(t_raw[0], t_raw[-1])
    env_100 = np.interp(grid_t, t_raw, env_2000, left=np.nan, right=np.nan)

    t0 = set_start - pad_s
    t1 = set_end + pad_s
    raw_mask = (t_raw >= t0) & (t_raw <= t1)
    env_mask = (grid_t >= t0) & (grid_t <= t1)

    set_dur = set_end - set_start

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    axes[0].plot(t_raw[raw_mask] - set_start, x_raw[raw_mask] * 1e6,
                 lw=0.4, color="C0")
    axes[0].set_ylabel("Raw EMG (µV)")
    axes[0].set_title(f"{recording} — set {set_number} "
                      f"({set_dur:.1f} s, {pad_s:.0f} s pad)  "
                      f"raw 2000 Hz vs 100 Hz RMS envelope")
    axes[0].axvspan(0, set_dur, color="0.85", alpha=0.4, zorder=-1,
                    label="kinect_set window")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_raw[raw_mask] - set_start, env_2000[raw_mask] * 1e6,
                 lw=0.5, color="0.5", alpha=0.7,
                 label="envelope @ 2000 Hz (pre-decimation)")
    axes[1].plot(grid_t[env_mask] - set_start, env_100[env_mask] * 1e6,
                 lw=1.4, color="C3", label="envelope @ 100 Hz (in parquet)")
    axes[1].axvspan(0, set_dur, color="0.85", alpha=0.4, zorder=-1)
    axes[1].set_ylabel("Envelope (µV)")
    axes[1].set_xlabel(f"Time relative to set start (s)  —  set duration = {set_dur:.2f} s")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    out = ROOT / "results" / f"emg_2000_vs_100_{recording}_set{set_number:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    rec = sys.argv[1] if len(sys.argv) > 1 else "recording_012"
    sn = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    pad = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    main(rec, sn, pad)
