"""EMG power spectrogram per set — visualizes spectral compression (fatigue).

Reads raw 2000 Hz EMG from dataset_aligned/recording_NNN/emg.csv, applies the
offline 20-450 Hz bandpass + 50 Hz notch (src/features/emg_features.py), then
computes a Welch-style spectrogram (scipy.signal.spectrogram). Overlays MNF
from window_features.parquet so you can see the spectral centroid drift down
as the set progresses (Cifrek 2009; De Luca 1997).

Single set:
    python scripts/plot_emg_power.py --recording 014 --set 1

All sets of one exercise (side-by-side):
    python scripts/plot_emg_power.py --recording 014 --exercise pullup

All 12 sets in one figure:
    python scripts/plot_emg_power.py --recording 014 --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine
from src.features.emg_features import FS_EMG, _filter_emg_offline
from scripts.plot_set_emg_acc_rms import (
    EXERCISE_COLORS,
    load_rep_times,
    load_window_features,
)

apply_style()

FREQ_MAX_HZ = 500.0  # display cap; bandpass cuts at 450 anyway


def _load_metadata(recording_id: str) -> dict:
    path = Path(f"dataset_aligned/recording_{recording_id}/metadata.json")
    return json.loads(path.read_text())


def _load_raw_emg(recording_id: str) -> pd.DataFrame:
    path = Path(f"dataset_aligned/recording_{recording_id}/emg.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


def _slice_set(emg_df: pd.DataFrame, t_unix_start: float, t_unix_end: float,
               pad_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_session_s, emg) for the active set ± pad. t_session is relative
    to the set start so x=0 marks the first set sample."""
    mask = ((emg_df["timestamp"] >= t_unix_start - pad_s) &
            (emg_df["timestamp"] <= t_unix_end + pad_s))
    seg = emg_df.loc[mask]
    t = seg["timestamp"].to_numpy() - t_unix_start
    y = seg["emg"].to_numpy().astype(float)
    return t, y


def _spectrogram_db(emg: np.ndarray, fs: int = FS_EMG,
                    win_s: float = 0.5, hop_s: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (f, t, Sxx_dB). 500 ms window matches MNF/MDF feature window."""
    nperseg = int(win_s * fs)
    noverlap = nperseg - int(hop_s * fs)
    f, t, Sxx = spectrogram(emg, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            scaling="density", mode="psd")
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


def _set_unix_bounds(meta: dict, set_number: int) -> tuple[float, float, float]:
    """Returns (t_unix_start, t_unix_end, t_session_start) for the set."""
    sets = {int(s["set_number"]): s for s in meta["kinect_sets"]}
    if set_number not in sets:
        raise ValueError(f"set {set_number} not in metadata.kinect_sets")
    s = sets[set_number]
    t0 = float(meta["data_start_unix_time"])
    return float(s["start_unix_time"]), float(s["end_unix_time"]), float(s["start_unix_time"]) - t0


def _draw_spectrogram(ax, t_rel: np.ndarray, f: np.ndarray, Sxx_db: np.ndarray,
                      t_set_start: float, t_set_end: float,
                      mnf_t: np.ndarray | None, mnf_y: np.ndarray | None,
                      rep_times_rel: list[float],
                      vmin: float, vmax: float) -> None:
    fmask = f <= FREQ_MAX_HZ
    im = ax.pcolormesh(t_rel, f[fmask], Sxx_db[fmask], shading="gouraud",
                       cmap="magma", vmin=vmin, vmax=vmax)
    ax.axvline(t_set_start, color="white", linewidth=1.0, alpha=0.7, linestyle=":")
    ax.axvline(t_set_end, color="white", linewidth=1.0, alpha=0.7, linestyle=":")
    for rt in rep_times_rel:
        ax.axvline(rt, color="white", linewidth=0.6, alpha=0.45, linestyle="--")
    if mnf_t is not None and mnf_y is not None and len(mnf_t) > 0:
        ax.plot(mnf_t, mnf_y, color="#7CFC00", linewidth=1.8, alpha=0.95,
                label="MNF (Hz)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_ylim(0, FREQ_MAX_HZ)
    return im


def _mnf_overlay(df_win: pd.DataFrame, t_session_set_start: float,
                 t_unix_start: float, t_unix_end: float,
                 pad_s: float) -> tuple[np.ndarray, np.ndarray]:
    """MNF trace from window_features, x in seconds relative to set start."""
    t0 = t_session_set_start - pad_s
    t1 = t_session_set_start + (t_unix_end - t_unix_start) + pad_s
    sl = df_win[(df_win["t_session_s"] >= t0) & (df_win["t_session_s"] <= t1)]
    return (sl["t_session_s"].to_numpy() - t_session_set_start,
            sl["emg_mnf"].to_numpy())


def plot_set_spectrogram(recording_id: str, set_number: int,
                         pad_s: float = 2.0) -> Path:
    meta = _load_metadata(recording_id)
    emg_df = _load_raw_emg(recording_id)
    df_win = load_window_features(recording_id)

    t_unix_start, t_unix_end, t_sess_start = _set_unix_bounds(meta, set_number)
    t_rel, y = _slice_set(emg_df, t_unix_start, t_unix_end, pad_s)
    if len(y) < int(0.5 * FS_EMG):
        raise RuntimeError(f"too few EMG samples for set {set_number} ({len(y)})")
    y_filt = _filter_emg_offline(y, fs=FS_EMG)

    f, t_spec, Sxx_db = _spectrogram_db(y_filt, fs=FS_EMG)
    t_spec = t_spec + t_rel[0]

    rep_times = load_rep_times(recording_id, set_number)
    rep_rel = [rt - t_sess_start for rt in rep_times]
    mnf_t, mnf_y = _mnf_overlay(df_win, t_sess_start, t_unix_start, t_unix_end, pad_s)

    set_row = df_win[df_win["set_number"] == float(set_number)].iloc[0]
    exercise = str(set_row["exercise"])
    rpe = int(set_row["rpe_for_this_set"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    vmin = float(np.nanpercentile(Sxx_db, 5))
    vmax = float(np.nanpercentile(Sxx_db, 99))
    im = _draw_spectrogram(ax, t_spec, f, Sxx_db, t_set_start=0.0,
                           t_set_end=t_unix_end - t_unix_start,
                           mnf_t=mnf_t, mnf_y=mnf_y, rep_times_rel=rep_rel,
                           vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Power (dB / Hz)", fontsize=11)
    ax.set_xlabel("Tid relativt til set-start (s)")
    ax.set_ylabel("Frekvens (Hz)")
    ax.set_title(f"Recording {recording_id} — set {set_number} — {exercise} — RPE {rpe}")
    fig.tight_layout()
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"set{set_number:02d}_emg_power_spectrogram.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_exercise_spectrogram(recording_id: str, exercise: str,
                              pad_s: float = 2.0) -> Path:
    meta = _load_metadata(recording_id)
    emg_df = _load_raw_emg(recording_id)
    df_win = load_window_features(recording_id)

    ex_rows = df_win[df_win["exercise"] == exercise]
    if ex_rows.empty:
        avail = sorted(df_win["exercise"].dropna().unique().tolist())
        raise ValueError(f"exercise {exercise!r} not found (available: {avail})")
    set_numbers = sorted(int(n) for n in ex_rows["set_number"].dropna().unique())

    panels = []
    for set_n in set_numbers:
        t_unix_start, t_unix_end, t_sess_start = _set_unix_bounds(meta, set_n)
        t_rel, y = _slice_set(emg_df, t_unix_start, t_unix_end, pad_s)
        y_filt = _filter_emg_offline(y, fs=FS_EMG)
        f, t_spec, Sxx_db = _spectrogram_db(y_filt, fs=FS_EMG)
        t_spec = t_spec + t_rel[0]
        rep_rel = [rt - t_sess_start for rt in load_rep_times(recording_id, set_n)]
        mnf_t, mnf_y = _mnf_overlay(df_win, t_sess_start, t_unix_start, t_unix_end, pad_s)
        rpe = int(df_win[df_win["set_number"] == float(set_n)]["rpe_for_this_set"].iloc[0])
        panels.append({
            "set_n": set_n, "rpe": rpe,
            "t": t_spec, "f": f, "S": Sxx_db,
            "set_end_rel": t_unix_end - t_unix_start,
            "rep_rel": rep_rel, "mnf_t": mnf_t, "mnf_y": mnf_y,
        })

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.5),
                             sharey=True, gridspec_kw={"wspace": 0.05})
    if n == 1:
        axes = [axes]

    all_S = np.concatenate([p["S"].ravel() for p in panels])
    vmin = float(np.nanpercentile(all_S, 5))
    vmax = float(np.nanpercentile(all_S, 99))

    im = None
    for ax, p in zip(axes, panels):
        im = _draw_spectrogram(ax, p["t"], p["f"], p["S"], t_set_start=0.0,
                               t_set_end=p["set_end_rel"],
                               mnf_t=p["mnf_t"], mnf_y=p["mnf_y"],
                               rep_times_rel=p["rep_rel"],
                               vmin=vmin, vmax=vmax)
        ax.set_title(f"Set {p['set_n']} — RPE {p['rpe']}",
                     fontsize=14, fontweight="bold",
                     color=EXERCISE_COLORS.get(exercise, "#222"))
        ax.set_xlabel("Tid (s)")
    axes[0].set_ylabel("Frekvens (Hz)")
    cbar = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.025)
    cbar.set_label("Power (dB / Hz)", fontsize=11)

    fig.suptitle(f"Recording {recording_id} — {exercise} — EMG power spectrogram",
                 fontsize=18)
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{exercise}_emg_power_spectrogram.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_all_sets_spectrogram(recording_id: str, pad_s: float = 2.0) -> Path:
    meta = _load_metadata(recording_id)
    emg_df = _load_raw_emg(recording_id)
    df_win = load_window_features(recording_id)

    set_numbers = sorted(int(s["set_number"]) for s in meta["kinect_sets"])
    panels = []
    for set_n in set_numbers:
        t_unix_start, t_unix_end, t_sess_start = _set_unix_bounds(meta, set_n)
        t_rel, y = _slice_set(emg_df, t_unix_start, t_unix_end, pad_s)
        if len(y) < int(0.5 * FS_EMG):
            continue
        y_filt = _filter_emg_offline(y, fs=FS_EMG)
        f, t_spec, Sxx_db = _spectrogram_db(y_filt, fs=FS_EMG)
        t_spec = t_spec + t_rel[0]
        rep_rel = [rt - t_sess_start for rt in load_rep_times(recording_id, set_n)]
        mnf_t, mnf_y = _mnf_overlay(df_win, t_sess_start, t_unix_start, t_unix_end, pad_s)
        srow = df_win[df_win["set_number"] == float(set_n)].iloc[0]
        panels.append({
            "set_n": set_n, "rpe": int(srow["rpe_for_this_set"]),
            "exercise": str(srow["exercise"]),
            "t": t_spec, "f": f, "S": Sxx_db,
            "set_end_rel": t_unix_end - t_unix_start,
            "rep_rel": rep_rel, "mnf_t": mnf_t, "mnf_y": mnf_y,
        })

    n = len(panels)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.6 * n_rows),
                             sharey=True, gridspec_kw={"wspace": 0.05, "hspace": 0.30})
    axes = np.atleast_2d(axes)

    all_S = np.concatenate([p["S"].ravel() for p in panels])
    vmin = float(np.nanpercentile(all_S, 5))
    vmax = float(np.nanpercentile(all_S, 99))

    im = None
    for idx, p in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        im = _draw_spectrogram(ax, p["t"], p["f"], p["S"], t_set_start=0.0,
                               t_set_end=p["set_end_rel"],
                               mnf_t=p["mnf_t"], mnf_y=p["mnf_y"],
                               rep_times_rel=p["rep_rel"],
                               vmin=vmin, vmax=vmax)
        ax.set_title(f"Set {p['set_n']} — {p['exercise']} — RPE {p['rpe']}",
                     fontsize=12, fontweight="bold",
                     color=EXERCISE_COLORS.get(p["exercise"], "#222"))
        if r == n_rows - 1:
            ax.set_xlabel("Tid (s)")
        if c == 0:
            ax.set_ylabel("Frekvens (Hz)")
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.01, fraction=0.015)
    cbar.set_label("Power (dB / Hz)", fontsize=11)

    fig.suptitle(f"Recording {recording_id} — EMG power spectrogram (all sets)",
                 fontsize=20)
    despine(fig=fig)

    out_dir = Path(f"inspections/recording_{recording_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_sets_emg_power_spectrogram.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", dest="set_number", type=int)
    group.add_argument("--exercise",
                       choices=["pullup", "squat", "deadlift", "benchpress"])
    group.add_argument("--all", action="store_true")
    parser.add_argument("--pad", type=float, default=2.0)
    args = parser.parse_args()
    rid = args.recording.zfill(3)
    if args.all:
        plot_all_sets_spectrogram(rid, pad_s=args.pad)
    elif args.exercise:
        plot_exercise_spectrogram(rid, args.exercise, pad_s=args.pad)
    else:
        plot_set_spectrogram(rid, args.set_number, pad_s=args.pad)


if __name__ == "__main__":
    main()
