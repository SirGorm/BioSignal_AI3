"""Interactive accelerometer filter explorer.

Loads ax/ay/az from dataset_aligned/recording_NNN/, computes acc_mag,
then renders raw vs. bandpass-filtered signal with live sliders for
lowcut, highcut, and Butterworth order.

Usage:
    python scripts/acc_filter_explorer.py 012
    python scripts/acc_filter_explorer.py 012 --duration 60 --start 120
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from scipy.signal import butter, sosfiltfilt

FS_ACC = 100  # Hz (per CLAUDE.md)
NYQ = FS_ACC / 2.0


def load_acc_mag(recording_id: str, start_s: float, duration_s: float | None) -> tuple[np.ndarray, np.ndarray]:
    base = Path(__file__).resolve().parents[1] / "dataset_aligned" / f"recording_{recording_id}"
    if not base.exists():
        raise FileNotFoundError(f"Recording not found: {base}")

    ax = pd.read_csv(base / "ax.csv")
    ay = pd.read_csv(base / "ay.csv")
    az = pd.read_csv(base / "az.csv")

    n = min(len(ax), len(ay), len(az))
    ax, ay, az = ax.iloc[:n], ay.iloc[:n], az.iloc[:n]
    t_unix = ax["timestamp"].to_numpy()
    acc_mag = np.sqrt(ax["ax"].to_numpy() ** 2 + ay["ay"].to_numpy() ** 2 + az["az"].to_numpy() ** 2)

    t_rel = t_unix - t_unix[0]
    mask = t_rel >= start_s
    if duration_s is not None:
        mask &= t_rel < start_s + duration_s
    return t_rel[mask], acc_mag[mask]


def filter_acc(acc_mag: np.ndarray, lowcut: float, highcut: float, order: int) -> np.ndarray:
    lowcut = max(0.01, min(lowcut, NYQ - 0.1))
    highcut = max(lowcut + 0.05, min(highcut, NYQ - 0.01))
    sos = butter(order, [lowcut, highcut], btype="band", fs=FS_ACC, output="sos")
    return sosfiltfilt(sos, acc_mag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_id", help="e.g. 012")
    parser.add_argument("--start", type=float, default=0.0, help="seconds from recording start")
    parser.add_argument("--duration", type=float, default=120.0, help="seconds to plot (None = all)")
    args = parser.parse_args()

    rec_id = args.recording_id.zfill(3)
    duration = None if args.duration <= 0 else args.duration
    t, acc_mag = load_acc_mag(rec_id, args.start, duration)
    print(f"Loaded recording_{rec_id}: {len(acc_mag)} samples ({t[-1]-t[0]:.1f} s)")

    init_low, init_high, init_order = 0.5, 20.0, 4
    filtered = filter_acc(acc_mag, init_low, init_high, init_order)

    fig, (ax_raw, ax_filt) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.28, top=0.95, hspace=0.25)

    ax_raw.plot(t, acc_mag, lw=0.6, color="#888")
    ax_raw.set_ylabel("acc_mag (raw)")
    ax_raw.set_title(f"recording_{rec_id} — accelerometer magnitude")
    ax_raw.grid(alpha=0.3)

    (line_filt,) = ax_filt.plot(t, filtered, lw=0.7, color="#1f77b4")
    ax_filt.set_ylabel("acc_mag (bandpass)")
    ax_filt.set_xlabel("time (s)")
    ax_filt.grid(alpha=0.3)
    title = ax_filt.set_title("")

    def update_title(low: float, high: float, order: int) -> None:
        title.set_text(f"bandpass {low:.2f}–{high:.2f} Hz, order {order}")

    update_title(init_low, init_high, init_order)

    ax_low = plt.axes([0.10, 0.15, 0.80, 0.03])
    ax_high = plt.axes([0.10, 0.10, 0.80, 0.03])
    ax_order = plt.axes([0.10, 0.05, 0.80, 0.03])

    s_low = Slider(ax_low, "lowcut (Hz)", 0.05, 10.0, valinit=init_low, valstep=0.05)
    s_high = Slider(ax_high, "highcut (Hz)", 1.0, NYQ - 0.5, valinit=init_high, valstep=0.5)
    s_order = Slider(ax_order, "order", 1, 8, valinit=init_order, valstep=1)

    def on_change(_val: float) -> None:
        low, high, order = s_low.val, s_high.val, int(s_order.val)
        if high <= low:
            high = low + 0.5
            s_high.eventson = False
            s_high.set_val(high)
            s_high.eventson = True
        try:
            new = filter_acc(acc_mag, low, high, order)
        except ValueError as e:
            title.set_text(f"filter error: {e}")
            fig.canvas.draw_idle()
            return
        line_filt.set_ydata(new)
        ymin, ymax = float(new.min()), float(new.max())
        pad = 0.05 * (ymax - ymin + 1e-9)
        ax_filt.set_ylim(ymin - pad, ymax + pad)
        update_title(low, high, order)
        fig.canvas.draw_idle()

    s_low.on_changed(on_change)
    s_high.on_changed(on_change)
    s_order.on_changed(on_change)

    plt.show()


if __name__ == "__main__":
    main()
