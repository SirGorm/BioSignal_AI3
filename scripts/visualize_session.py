"""Render a 3D skeleton video of a recording session with per-window labels.

Ported from biosignal_pipe/src/eval/visualize_session.py to BioSignal_AI3 paths
and data layout:
  - raw signals: dataset_aligned/recording_NNN/{emg,ax,ay,az,ppg_green,temperature}.csv
  - joints:      dataset_aligned/recording_NNN/recording_NN_joints.json
  - labels:      data/labeled/recording_NNN/aligned_features.parquet
  - rep peaks:   dataset_aligned/recording_NNN/markers.json (Set:N_Rep:K)

Shows the 4 model-input modalities only (EMG, ACC magnitude, PPG-green, Temp).
ECG/EDA are excluded per CLAUDE.md (signal quality unusable on this dataset).

Run:
    python scripts/visualize_session.py --recording 7 --set 5
    python scripts/visualize_session.py --recording 7 --set 5 --gif-mid 5
    python scripts/visualize_session.py --recording 7  # whole session

Output: cache/viz_rec_<id>[_set_<n>].mp4 (and .gif when --gif-mid given).
Requires ffmpeg on PATH for MP4; falls back to GIF via PillowWriter otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from scipy.signal import butter, iirnotch, lfilter

# Use the ffmpeg shipped with imageio-ffmpeg if there's none on PATH.
try:
    import imageio_ffmpeg as _iioff
    matplotlib.rcParams["animation.ffmpeg_path"] = _iioff.get_ffmpeg_exe()
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset_aligned"
LABELED_DIR = REPO_ROOT / "data" / "labeled"
CACHE_DIR = REPO_ROOT / "cache"

WINDOW_S = 2.0  # 2 s model window (matches CLAUDE.md feature-pipeline default)

# Shift the biosignal slice by SIGNAL_LOOKBACK_S seconds so that the signal
# feature corresponding to the current skeleton pose lands at x=0 (middle of
# panel) instead of at the left edge. With WINDOW_S=2 and SIGNAL_LOOKBACK_S=1
# the panel shows a pure lookback window [t-2, t]: middle = "now" (matches the
# skeleton pose), right edge = newest sample, left edge = 2 s ago.
SIGNAL_LOOKBACK_S = 1.0

# EMG envelope settings (causal, real-time-compatible).
EMG_BP_LOW_HZ = 20.0    # Konrad (2005), De Luca (2002): 20–450 Hz BP for sEMG
EMG_BP_HIGH_HZ = 450.0
EMG_NOTCH_HZ = 50.0     # Norway mains
EMG_NOTCH_Q = 30.0
EMG_RMS_WIN_S = 0.10    # 100 ms RMS window — typical for sEMG envelope display

MODEL_MODALITIES = ["emg", "acc", "ppg", "temp"]
SIGNAL_TITLES = {
    "emg":  "EMG RMS (causal, 100 ms)",
    "acc":  "ACC mag (100 Hz)",
    "ppg":  "PPG-green (100 Hz)",
    "temp": "Temp (1 Hz)",
}
SIGNAL_COLORS = {
    "emg":  "#d62728",
    "acc":  "#2ca02c",
    "ppg":  "#9467bd",
    "temp": "#ff7f0e",
}

EXERCISE_COLORS = {
    "rest":       "#999999",
    "squat":      "#1f77b4",
    "deadlift":   "#fd8d3c",
    "benchpress": "#2ca02c",
    "pullup":     "#9467bd",
}

PHASE_COLORS = {
    "rest":        "#999999",
    "concentric":  "#2ca02c",
    "eccentric":   "#d62728",
    "isometric":   "#1f77b4",
}


# ---------------------------------------------------------------------------
# Data loading (AI3 layout)
# ---------------------------------------------------------------------------

def _rec_dir(recording_id: int) -> Path:
    return DATASET_DIR / f"recording_{recording_id:03d}"


def _emg_rms_envelope(y: np.ndarray, fs: float) -> np.ndarray:
    """Causal sEMG envelope: 20–450 Hz BP + 50 Hz notch + rectify + RMS.

    Pipeline (all causal — uses lfilter, not filtfilt):
      1) Butterworth band-pass (Konrad 2005; De Luca 2002)
      2) Notch at mains (50 Hz in Norway)
      3) Rectify |x|
      4) Rolling RMS over EMG_RMS_WIN_S via sqrt(running_mean(x^2))
    """
    nyq = 0.5 * fs
    bp_b, bp_a = butter(4, [EMG_BP_LOW_HZ / nyq, EMG_BP_HIGH_HZ / nyq], btype="band")
    nb, na = iirnotch(EMG_NOTCH_HZ, EMG_NOTCH_Q, fs=fs)
    x = lfilter(bp_b, bp_a, y)
    x = lfilter(nb, na, x)
    x = x * x  # squared
    win = max(1, int(round(EMG_RMS_WIN_S * fs)))
    kernel = np.ones(win, dtype=np.float64) / win
    ms = lfilter(kernel, [1.0], x)  # causal moving average of x^2
    return np.sqrt(np.maximum(ms, 0.0))


def _read_signal(rec: Path, name: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, values) as float64 numpy arrays."""
    df = pd.read_csv(rec / f"{name}.csv", usecols=[0, 1])
    df.columns = ["timestamp", col]
    return df["timestamp"].to_numpy(np.float64), df[col].to_numpy(np.float64)


def _load_signals(recording_id: int) -> dict[str, dict]:
    """Load model-input modalities into memory as (t, y) numpy arrays.

    `acc` is the per-sample magnitude across ax/ay/az (100 Hz).
    `temp` may be empty — caller handles that.
    """
    rec = _rec_dir(recording_id)
    signals: dict[str, dict] = {}

    t, y = _read_signal(rec, "emg", "emg")
    signals["emg"] = {"t": t, "y": _emg_rms_envelope(y, fs=2000.0), "fs": 2000.0}

    tx, ax = _read_signal(rec, "ax", "ax")
    _,  ay = _read_signal(rec, "ay", "ay")
    _,  az = _read_signal(rec, "az", "az")
    signals["acc"] = {"t": tx, "y": np.sqrt(ax**2 + ay**2 + az**2), "fs": 100.0}

    t, y = _read_signal(rec, "ppg_green", "ppg_green")
    signals["ppg"] = {"t": t, "y": y, "fs": 100.0}

    temp_path = rec / "temperature.csv"
    if temp_path.exists():
        try:
            t, y = _read_signal(rec, "temperature", "temperature")
            if len(t) >= 2:
                signals["temp"] = {"t": t, "y": y, "fs": 1.0}
        except Exception:
            pass
    if "temp" not in signals:
        signals["temp"] = {"t": np.array([]), "y": np.array([]), "fs": 1.0}

    return signals


def _slice_window(sig: dict, t_center: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x_rel, y) for the 2 s window centered on t_center; None if empty."""
    t = sig["t"]
    if t.size == 0:
        return None
    lo = t_center - WINDOW_S / 2
    hi = t_center + WINDOW_S / 2
    i0 = np.searchsorted(t, lo, side="left")
    i1 = np.searchsorted(t, hi, side="right")
    if i1 - i0 <= 0:
        return None
    return t[i0:i1] - t_center, sig["y"][i0:i1]


def _signal_y_limits(sig: dict, t_lo: float, t_hi: float) -> tuple[float, float]:
    """1st/99th percentile across the set window, with a small pad."""
    t = sig["t"]
    if t.size == 0:
        return -1.0, 1.0
    i0 = np.searchsorted(t, t_lo, side="left")
    i1 = np.searchsorted(t, t_hi, side="right")
    if i1 - i0 < 4:
        sub = sig["y"][max(0, i0):i1]
    else:
        sub = sig["y"][i0:i1]
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    pad = 0.1 * max(hi - lo, 1e-3)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Labels (from aligned_features.parquet + markers.json)
# ---------------------------------------------------------------------------

def _set_meta(recording_id: int, set_number: int) -> dict:
    rec = _rec_dir(recording_id)
    meta = json.loads((rec / "metadata.json").read_text())
    for ks in meta.get("kinect_sets", []):
        if int(ks["set_number"]) == set_number:
            return ks
    raise SystemExit(
        f"set {set_number} not in metadata.json kinect_sets for rec {recording_id}"
    )


def _set_labels(recording_id: int, set_number: int) -> dict:
    """Load per-set labels (exercise, RPE) and per-sample phase from
    aligned_features.parquet, plus per-rep peak timestamps from markers.json."""
    df = pd.read_parquet(LABELED_DIR / f"recording_{recording_id:03d}"
                         / "aligned_features.parquet")
    s = df[df["set_number"] == set_number].copy()
    if s.empty:
        raise SystemExit(
            f"no rows for set {set_number} in aligned_features.parquet rec {recording_id}"
        )
    exercise = str(s["exercise"].iloc[0])
    rpe = float(s["rpe_for_this_set"].iloc[0])

    phase_t = s["t_unix"].to_numpy(np.float64)
    phase_l = s["phase_label"].astype(str).to_numpy()

    markers = json.loads((_rec_dir(recording_id) / "markers.json").read_text())
    entries = markers if isinstance(markers, list) else markers.get("markers", [])
    rep_t = np.array(
        [float(e["unix_time"]) for e in entries
         if str(e.get("label", "")).startswith(f"Set:{set_number}_Rep:")],
        dtype=np.float64,
    )
    rep_t.sort()

    return {
        "exercise": exercise,
        "rpe": rpe,
        "phase_t": phase_t,
        "phase_l": phase_l,
        "rep_t": rep_t,
    }


def _phase_at(phase_t: np.ndarray, phase_l: np.ndarray, t: float) -> str:
    if phase_t.size == 0:
        return "rest"
    idx = int(np.clip(np.searchsorted(phase_t, t, side="right") - 1, 0, phase_t.size - 1))
    return str(phase_l[idx])


def _reps_in_window(rep_t: np.ndarray, t: float) -> int:
    if rep_t.size == 0:
        return 0
    return int(((rep_t >= t - WINDOW_S / 2) & (rep_t < t + WINDOW_S / 2)).sum())


def _reps_done(rep_t: np.ndarray, t: float) -> int:
    if rep_t.size == 0:
        return 0
    return int((rep_t <= t).sum())


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

def _load_joints(recording_id: int, set_number: int) -> dict:
    path = _rec_dir(recording_id) / f"recording_{set_number:02d}_joints.json"
    if not path.exists():
        raise SystemExit(f"joints file missing: {path}")
    with path.open() as fh:
        return json.load(fh)


def _frame_timestamps(n_frames: int, t_start: float, t_end: float) -> np.ndarray:
    """timestamp_usec is always 0 in this dataset's joints files — spread frames
    linearly across the kinect_set start/end Unix bounds (CLAUDE.md convention)."""
    if n_frames <= 1:
        return np.array([t_start], dtype=np.float64)
    return t_start + np.arange(n_frames, dtype=np.float64) * (
        (t_end - t_start) / (n_frames - 1)
    )


def _dominant_body_id(frames: list[dict]) -> int | None:
    counts: dict[int, int] = {}
    for fr in frames:
        if fr.get("num_bodies"):
            for b in fr["bodies"]:
                counts[b["body_id"]] = counts.get(b["body_id"], 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _skeleton_points(frame: dict, bid: int | None, n_joints: int) -> np.ndarray | None:
    if not frame.get("num_bodies") or bid is None:
        return None
    body = next((b for b in frame["bodies"] if b["body_id"] == bid), None)
    if body is None:
        return None
    jp = np.asarray(body["joint_positions"], dtype=np.float32)
    if jp.shape != (n_joints, 3):
        return None
    return jp


def _kinect_to_plot(jp: np.ndarray) -> np.ndarray:
    """Azure Kinect (x right, y down, z forward) -> matplotlib (z up)."""
    out = np.empty_like(jp)
    out[..., 0] = jp[..., 0]
    out[..., 1] = jp[..., 2]
    out[..., 2] = -jp[..., 1]
    return out


def _auto_limits(skeletons: list[np.ndarray | None]) -> np.ndarray:
    pts = [_kinect_to_plot(s) for s in skeletons if s is not None]
    if not pts:
        return np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
    stacked = np.concatenate(pts, axis=0)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    pad = 0.15 * np.maximum(hi - lo, 1e-3)
    return np.stack([lo - pad, hi + pad], axis=1)


# ---------------------------------------------------------------------------
# Plan + render
# ---------------------------------------------------------------------------

def _build_set_plan(recording_id: int, set_number: int, fps_cap: int) -> dict:
    sm = _set_meta(recording_id, set_number)
    t_start = float(sm["start_unix_time"])
    t_end = float(sm["end_unix_time"])
    labels = _set_labels(recording_id, set_number)
    jd = _load_joints(recording_id, set_number)

    n = len(jd["frames"])
    ts = _frame_timestamps(n, t_start, t_end)
    bid = _dominant_body_id(jd["frames"])
    n_joints = len(jd["joint_names"])
    sk = [_skeleton_points(fr, bid, n_joints) for fr in jd["frames"]]

    native_fps = n / max(1e-6, t_end - t_start)
    stride = max(1, int(round(native_fps / fps_cap)))
    ts = ts[::stride]
    sk = sk[::stride]
    print(f"[viz]   set {set_number}: {len(ts)} frames "
          f"({native_fps:.1f} fps native, stride={stride}), exercise={labels['exercise']}, "
          f"rpe={labels['rpe']:.0f}")

    return {
        "set_number": set_number,
        "t_start": t_start,
        "t_end": t_end,
        "exercise": labels["exercise"],
        "rpe": labels["rpe"],
        "phase_t": labels["phase_t"],
        "phase_l": labels["phase_l"],
        "rep_t": labels["rep_t"],
        "rep_total": int(labels["rep_t"].size),
        "bone_list": jd["bone_list"],
        "joint_names": jd["joint_names"],
        "timestamps": ts,
        "skeletons": sk,
        "limits": _auto_limits(sk),
    }


def _render(
    recording_id: int,
    plan: list[dict],
    signals: dict[str, dict],
    fps: int,
    output: Path,
    frame_range: tuple[int, int] | None = None,
) -> Path:
    """plan = list of per-set dicts. frame_range slices the flat frame list."""
    flat: list[dict] = []
    for sp in plan:
        for i in range(len(sp["timestamps"])):
            flat.append({"sp": sp, "i": i})
    if not flat:
        raise RuntimeError("no frames to render")
    if frame_range is not None:
        a, b = frame_range
        flat = flat[a:b]
        if not flat:
            raise RuntimeError("empty frame_range slice")

    # Per-modality y-limits across the full plan window.
    t_lo = min(sp["t_start"] for sp in plan) - WINDOW_S
    t_hi = max(sp["t_end"]   for sp in plan) + WINDOW_S
    scales = {m: _signal_y_limits(signals[m], t_lo, t_hi) for m in MODEL_MODALITIES}

    fig = plt.figure(figsize=(14, 8), dpi=110)
    gs = fig.add_gridspec(nrows=2, ncols=4,
                          height_ratios=[2.2, 1.0],
                          width_ratios=[1, 1, 1, 1],
                          hspace=0.35, wspace=0.35)
    ax3d = fig.add_subplot(gs[0, 0:3], projection="3d")
    ax_txt = fig.add_subplot(gs[0, 3]); ax_txt.axis("off")

    sig_axes: dict[str, plt.Axes] = {
        m: fig.add_subplot(gs[1, i]) for i, m in enumerate(MODEL_MODALITIES)
    }
    sig_lines: dict[str, plt.Line2D] = {}
    sig_now_dots: dict[str, plt.Line2D] = {}
    for m, ax in sig_axes.items():
        ax.set_title(SIGNAL_TITLES[m], fontsize=10)
        ax.set_xlim(-WINDOW_S / 2, WINDOW_S / 2)
        ax.set_ylim(*scales[m])
        # "now" line — middle of the rolling window = the person's current pose.
        ax.axvline(0.0, color="#d62728", lw=1.4, alpha=0.85, zorder=4)
        ax.text(0.0, 1.02, "now", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#d62728",
                fontweight="bold")
        ax.set_xticks([-WINDOW_S / 2, 0.0, WINDOW_S / 2])
        ax.set_xticklabels([f"-{WINDOW_S/2:.0f}s", "0", f"+{WINDOW_S/2:.0f}s"])
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, lw=0.5)
        marker = "o" if m == "temp" else None
        (ln,) = ax.plot([], [], lw=0.9, color=SIGNAL_COLORS[m], marker=marker, ms=3)
        sig_lines[m] = ln
        # Big dot at x=0 highlights the current sample (matches skeleton frame).
        (dot,) = ax.plot([], [], "o", color="#d62728", mec="black", mew=0.6,
                         ms=9, zorder=6)
        sig_now_dots[m] = dot

    bone_lines: list[plt.Line2D] = []
    joint_scatter = ax3d.scatter([], [], [], s=15, c="#d62728")
    title = ax3d.set_title("")

    def _apply_limits(limits: np.ndarray) -> None:
        ax3d.set_xlim(limits[0])
        ax3d.set_ylim(limits[1])
        ax3d.set_zlim(limits[2])
        ax3d.set_box_aspect((limits[0, 1] - limits[0, 0],
                             limits[1, 1] - limits[1, 0],
                             limits[2, 1] - limits[2, 0]))
        ax3d.set_xlabel("x (right)"); ax3d.set_ylabel("y (depth)"); ax3d.set_zlabel("z (up)")
        ax3d.view_init(elev=10, azim=-80)

    def init():
        _apply_limits(flat[0]["sp"]["limits"])
        return []

    def update(idx: int):
        f = flat[idx]
        sp = f["sp"]
        i = f["i"]

        nonlocal bone_lines
        name_to_idx = {n: k for k, n in enumerate(sp["joint_names"])}
        need = len(sp["bone_list"])
        while len(bone_lines) < need:
            (ln,) = ax3d.plot([], [], [], lw=2.0, color="#444444")
            bone_lines.append(ln)
        for ln in bone_lines[need:]:
            ln.set_data([], []); ln.set_3d_properties([])
        _apply_limits(sp["limits"])

        sk = sp["skeletons"][i]
        if sk is not None:
            skp = _kinect_to_plot(sk)
            joint_scatter._offsets3d = (skp[:, 0], skp[:, 1], skp[:, 2])
            for ln, bone in zip(bone_lines[:need], sp["bone_list"]):
                a, b = bone
                if a in name_to_idx and b in name_to_idx:
                    pa = skp[name_to_idx[a]]
                    pb = skp[name_to_idx[b]]
                    ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    ln.set_3d_properties([pa[2], pb[2]])
                else:
                    ln.set_data([], []); ln.set_3d_properties([])
        else:
            joint_scatter._offsets3d = ([], [], [])
            for ln in bone_lines[:need]:
                ln.set_data([], []); ln.set_3d_properties([])

        t = float(sp["timestamps"][i])
        phase = _phase_at(sp["phase_t"], sp["phase_l"], t)
        reps_in = _reps_in_window(sp["rep_t"], t)
        reps_done = _reps_done(sp["rep_t"], t)
        t_set = t - sp["t_start"]

        title.set_text(
            f"rec {recording_id}  |  set {sp['set_number']}  |  "
            f"t={t_set:+.2f}s into set  ({t:.1f} unix)"
        )

        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        ex = sp["exercise"]
        color = EXERCISE_COLORS.get(ex, "#444444")
        ax_txt.add_patch(plt.Rectangle((0.04, 0.80), 0.92, 0.14,
                                       facecolor=color, alpha=0.8, edgecolor="black"))
        ax_txt.text(0.5, 0.87, ex.upper(), ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white")
        ph_color = PHASE_COLORS.get(phase, "#444444")
        ax_txt.add_patch(plt.Rectangle((0.04, 0.64), 0.92, 0.12,
                                       facecolor=ph_color, alpha=0.6, edgecolor="black"))
        ax_txt.text(0.5, 0.70, f"phase: {phase}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")
        rpe = sp["rpe"]
        lines = [
            f"set number     :  {sp['set_number']}",
            f"rpe (set)      :  {rpe:.1f}" if rpe == rpe else "rpe (set)      :  n/a",
            f"reps done      :  {reps_done} / {sp['rep_total']}",
            f"reps in 2 s win:  {reps_in}",
            f"t in set       :  {t_set:.2f} s",
        ]
        for k, line in enumerate(lines):
            ax_txt.text(0.06, 0.55 - k * 0.08, line, ha="left", va="center",
                        fontsize=13, family="monospace")
        ax_txt.text(0.06, 0.04,
                    f"window = {WINDOW_S:.0f} s  (model input)",
                    fontsize=10, style="italic", color="#555555")

        for m in MODEL_MODALITIES:
            # Slice centered on (t - SIGNAL_LOOKBACK_S) so that the signal
            # sample at the skeleton's current time t lands at the right edge
            # of the panel and the panel "middle" represents the biosignal
            # feature physically associated with the current movement.
            w = _slice_window(signals[m], t - SIGNAL_LOOKBACK_S)
            if w is None:
                sig_lines[m].set_data([], [])
                sig_now_dots[m].set_data([], [])
            else:
                x_rel, y = w
                sig_lines[m].set_data(x_rel, y)
                # "now" sample = whichever point is closest to x=0.
                k = int(np.argmin(np.abs(x_rel)))
                sig_now_dots[m].set_data([0.0], [y[k]])
        return []

    anim = FuncAnimation(fig, update, frames=len(flat), init_func=init,
                         interval=1000 / fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    elif FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, bitrate=2500)
    else:
        print("[viz] ffmpeg not on PATH — falling back to .gif via PillowWriter")
        output = output.with_suffix(".gif")
        writer = PillowWriter(fps=fps)
    print(f"[viz] writing {len(flat)} frames -> {output}")
    anim.save(str(output), writer=writer)
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=int, required=True)
    ap.add_argument("--set", type=int, default=None, dest="only_set",
                    help="render only this set (default: every set with a joints file)")
    ap.add_argument("--fps", type=int, default=15,
                    help="target playback fps (Kinect native ~30)")
    ap.add_argument("--gif-mid", type=float, default=None,
                    help="also write a GIF of N seconds centered in the (single) set")
    ap.add_argument("--output", type=Path, default=None,
                    help="override MP4 output path (default cache/viz_rec_<id>[_set_<n>].mp4)")
    args = ap.parse_args()

    rec = _rec_dir(args.recording)
    if not rec.exists():
        raise SystemExit(f"recording dir missing: {rec}")
    meta = json.loads((rec / "metadata.json").read_text())
    kinect_sets = meta.get("kinect_sets", [])
    if not kinect_sets:
        raise SystemExit(f"no kinect_sets in metadata.json for rec {args.recording}")

    if args.only_set is not None:
        set_numbers = [args.only_set]
    else:
        set_numbers = [int(ks["set_number"]) for ks in kinect_sets]

    print("[viz] loading raw signals into memory ...")
    signals = _load_signals(args.recording)
    print("[viz] modalities loaded:",
          {m: f"{signals[m]['t'].size} samples" for m in MODEL_MODALITIES})

    plan: list[dict] = []
    for sn in set_numbers:
        joints = rec / f"recording_{sn:02d}_joints.json"
        if not joints.exists():
            print(f"[viz]   set {sn}: {joints.name} missing — skipping")
            continue
        plan.append(_build_set_plan(args.recording, sn, args.fps))
    if not plan:
        raise SystemExit("nothing to render")

    if args.output is None:
        stem = f"viz_rec_{args.recording:03d}"
        if args.only_set is not None:
            stem += f"_set_{args.only_set:02d}"
        out = CACHE_DIR / f"{stem}.mp4"
    else:
        out = args.output

    written = _render(args.recording, plan, signals, args.fps, out)
    print(f"[viz] done -> {written}")

    if args.gif_mid is not None:
        if len(plan) != 1:
            print("[viz] --gif-mid requires --set; skipping GIF")
            return
        sp = plan[0]
        ts = sp["timestamps"]
        if len(ts) < 2:
            print("[viz] not enough frames for GIF")
            return
        mid = len(ts) // 2
        half = max(1, int(round(args.gif_mid * args.fps / 2)))
        a = max(0, mid - half)
        b = min(len(ts), mid + half)
        gif_out = out.with_name(out.stem + "_clip.gif")
        gif_written = _render(
            args.recording, plan, signals, args.fps, gif_out, frame_range=(a, b)
        )
        actual_s = (b - a) / args.fps
        print(f"[viz] gif done -> {gif_written}  ({actual_s:.1f} s, frames {a}..{b})")


if __name__ == "__main__":
    main()
