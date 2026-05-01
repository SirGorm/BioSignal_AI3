"""
Segmentation QC plots for dataset_clean/.

Reads native CSVs + markers.json + metadata.json + set_quality.json from
dataset_clean/recording_NNN/ and dumps visualisations to:

  inspections/segmentation_qc_clean/recording_NNN/
    overview.png                      — full-session stack of 6 modalities
                                         with set bands shaded by exercise +
                                         per-set rep markers as vertical lines
    per_set/set_NN_<exercise>.png     — zoomed view ±5 s of each set
    report.md                         — set-quality summary table

No labels (parquet) needed — reads raw signals + Participants.xlsx for the
exercise/RPE per set. Filtering is OFFLINE (filtfilt) for visual clarity.

Usage:
    python scripts/segmentation_qc_clean.py                # all recordings
    python scripts/segmentation_qc_clean.py --recording 008
    python scripts/segmentation_qc_clean.py --no-per-set   # skip zoom plots
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

ROOT = Path(__file__).resolve().parent.parent
SRC_CLEAN = ROOT / "dataset_clean"
OUT_ROOT = ROOT / "inspections" / "segmentation_qc_clean"
XLSX = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset\Participants\Participants.xlsx")

# Native sample rates per modality.
NATIVE_FS = {
    "ecg": 500.0, "emg": 2000.0, "eda": 50.0,
    "ppg_green": 100.0, "temperature": 1.0,
    "ax": 100.0, "ay": 100.0, "az": 100.0,
}

EXERCISE_PALETTE = {
    "squat": "#3498db", "deadlift": "#9b59b6",
    "benchpress": "#1abc9c", "pullup": "#e67e22",
    "rest": "#ecf0f1", "unknown": "#bdc3c7",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(rec_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = rec_dir / f"{name}.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    if "timestamp" not in df.columns:
        return None
    return df


def load_acc_mag(rec_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, |a|) computed from ax/ay/az with shared timestamps."""
    ax = load_csv(rec_dir, "ax")
    ay = load_csv(rec_dir, "ay")
    az = load_csv(rec_dir, "az")
    if ax is None or ay is None or az is None:
        return np.array([]), np.array([])
    n = min(len(ax), len(ay), len(az))
    t = ax["timestamp"].to_numpy()[:n]
    mag = np.sqrt(ax["ax"].to_numpy()[:n] ** 2 +
                  ay["ay"].to_numpy()[:n] ** 2 +
                  az["az"].to_numpy()[:n] ** 2)
    return t, mag


def load_xlsx_per_recording() -> dict[int, dict]:
    """Parse Participants.xlsx into {recording_id: {subject, exercises[12], rpes[12]}}."""
    if not XLSX.exists():
        return {}
    df = pd.read_excel(XLSX, header=None)
    out: dict[int, dict] = {}
    i = 0
    while i < len(df):
        v = df.iloc[i, 0]
        try:
            rid = int(v)
        except (ValueError, TypeError):
            i += 1; continue
        name = str(df.iloc[i, 1])
        exs = [str(df.iloc[i, c]).strip() for c in range(2, 14)]
        rpes: list = []
        if i + 1 < len(df) and str(df.iloc[i + 1, 1]).strip().lower() == "fatigue":
            rpes = [df.iloc[i + 1, c] for c in range(2, 14)]
        out[rid] = {"subject": name, "exercises": exs, "rpes": rpes}
        i += 2
    return out


# ---------------------------------------------------------------------------
# Filters (offline; for visual clarity only)
# ---------------------------------------------------------------------------

def _filter_bp(x: np.ndarray, low: float, high: float, fs: float,
                notch: Optional[float] = None) -> np.ndarray:
    valid = np.isfinite(x)
    if valid.sum() < 50:
        return x.copy()
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    if low >= high:
        return x.copy()
    b, a = butter(4, [low, high], btype="band", fs=fs)
    y = x.copy()
    yv = filtfilt(b, a, x[valid].astype(float))
    if notch and notch < nyq:
        b_n, a_n = iirnotch(notch, 30.0, fs=fs)
        yv = filtfilt(b_n, a_n, yv)
    y[valid] = yv
    return y


def _filter_lp(x: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    valid = np.isfinite(x)
    if valid.sum() < 20:
        return x.copy()
    nyq = fs / 2.0
    cutoff = min(cutoff, 0.95 * nyq)
    if cutoff <= 0:
        return x.copy()
    b, a = butter(order, cutoff, btype="low", fs=fs)
    y = x.copy()
    y[valid] = filtfilt(b, a, x[valid].astype(float))
    return y


def emg_envelope(x: np.ndarray, fs: float, win_s: float = 0.1) -> np.ndarray:
    """Band-pass 20-450 Hz + 50 Hz notch + RMS envelope (De Luca 1997)."""
    bp = _filter_bp(x, 20.0, 450.0, fs, notch=50.0)
    win = max(1, int(round(win_s * fs)))
    sq = bp ** 2
    sq = np.where(np.isfinite(sq), sq, 0.0)
    rms = np.sqrt(np.convolve(sq, np.ones(win) / win, mode="same"))
    return rms


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _mod_panel(ax, t, v, label, color="#1f4e79"):
    if len(v) == 0:
        ax.text(0.5, 0.5, f"{label}: NO DATA", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_yticks([])
        return
    ax.plot(t, v, color=color, linewidth=0.5)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _shade_sets(ax, sets_md: list[dict], xlsx_info: Optional[dict]):
    for i, s in enumerate(sets_md):
        sn = s["set_number"]
        t0, t1 = s["start_unix_time"], s["end_unix_time"]
        ex = "unknown"
        if xlsx_info and xlsx_info.get("exercises"):
            # xlsx maps slot 1..12 to set_numbers in order of appearance
            # Conservative: use position in the metadata list, not set_number
            if i < len(xlsx_info["exercises"]):
                ex = xlsx_info["exercises"][i]
        col = EXERCISE_PALETTE.get(ex.lower(), EXERCISE_PALETTE["unknown"])
        ax.axvspan(t0, t1, color=col, alpha=0.20, zorder=0)


def _draw_reps(ax, markers: list[dict]):
    for e in markers:
        lbl = e.get("label", "")
        if "Rep:" in lbl:
            ax.axvline(e["unix_time"], color="#1f4e79", alpha=0.35,
                       linewidth=0.4, zorder=1)


def _annotate_set_labels(ax, sets_md: list[dict], xlsx_info: Optional[dict],
                         set_quality: dict, y_frac: float = 1.02):
    sq_by_n = {s["set_number"]: s for s in set_quality.get("sets", [])}
    for i, s in enumerate(sets_md):
        sn = s["set_number"]
        t_mid = 0.5 * (s["start_unix_time"] + s["end_unix_time"])
        ex = ""
        rpe = ""
        if xlsx_info:
            if i < len(xlsx_info.get("exercises", [])):
                ex = xlsx_info["exercises"][i][:4]
            rpes = xlsx_info.get("rpes", [])
            if i < len(rpes) and pd.notna(rpes[i]):
                rpe = f"R{int(rpes[i])}"
        sq = sq_by_n.get(sn, {})
        flag_str = ""
        if sq and not sq.get("ok_for_training", True):
            flag_str = "!"
        ax.text(t_mid, y_frac, f"{sn}{flag_str}\n{ex}{rpe}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="black")


def render_overview(rec: str, rec_dir: Path, out_dir: Path,
                    xlsx_info: Optional[dict]) -> Path:
    md = json.loads((rec_dir / "metadata.json").read_text()) if (rec_dir / "metadata.json").exists() else {}
    sets_md = md.get("kinect_sets", [])
    mk = json.loads((rec_dir / "markers.json").read_text()) if (rec_dir / "markers.json").exists() else []
    if isinstance(mk, dict):
        mk = mk.get("markers", [])
    sq = json.loads((rec_dir / "set_quality.json").read_text()) if (rec_dir / "set_quality.json").exists() else {"sets": []}

    # Load + filter each modality.
    panels = []
    for mod in ("ecg", "emg", "eda", "acc_mag", "ppg_green", "temperature"):
        if mod == "acc_mag":
            t, v = load_acc_mag(rec_dir)
            if len(v) > 50:
                v = _filter_bp(v, 0.5, 20.0, NATIVE_FS["ax"])
            panels.append(("acc_mag", t, v))
        else:
            df = load_csv(rec_dir, mod)
            if df is None or len(df) == 0:
                panels.append((mod, np.array([]), np.array([])))
                continue
            col = mod
            t = df["timestamp"].to_numpy()
            v = df[col].to_numpy()
            fs = NATIVE_FS[mod]
            try:
                if mod == "ecg":
                    v = _filter_bp(v, 0.5, 40.0, fs, notch=50.0)
                elif mod == "emg":
                    v = emg_envelope(v, fs)
                elif mod == "eda":
                    v = _filter_lp(v, 5.0, fs)
                elif mod == "ppg_green":
                    v = _filter_bp(v, 0.5, 8.0, fs)
                elif mod == "temperature":
                    v = _filter_lp(v, 0.1, fs, order=2)
            except Exception:
                pass
            panels.append((mod, t, v))

    # Determine x-range = ECG range if available, else widest.
    t_lo, t_hi = None, None
    for _, t, _ in panels:
        if len(t):
            t_lo = t[0] if t_lo is None else min(t_lo, t[0])
            t_hi = t[-1] if t_hi is None else max(t_hi, t[-1])

    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 11), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (mod, t, v) in zip(axes, panels):
        _shade_sets(ax, sets_md, xlsx_info)
        _draw_reps(ax, mk)
        nice = {"ecg": "ECG\n(BP 0.5-40 Hz)", "emg": "EMG envelope\n(20-450 Hz RMS)",
                "eda": "EDA\n(LP 5 Hz)", "acc_mag": "|acc|\n(BP 0.5-20 Hz)",
                "ppg_green": "PPG-green\n(BP 0.5-8 Hz)", "temperature": "Temp\n(LP 0.1 Hz)"}.get(mod, mod)
        _mod_panel(ax, t, v, nice)

    _annotate_set_labels(axes[0], sets_md, xlsx_info, sq)

    if t_lo is not None:
        axes[-1].set_xlim(t_lo, t_hi)
    axes[-1].set_xlabel("Unix time (s)")
    subj = xlsx_info.get("subject", "") if xlsx_info else ""
    fig.suptitle(f"recording_{rec} — {subj}  |  {len(sets_md)} sets, "
                 f"{sum(1 for s in sq.get('sets', []) if s.get('ok_for_training'))} ok",
                 fontsize=11)

    legend_items = [Patch(facecolor=EXERCISE_PALETTE[ex], alpha=0.4, label=ex)
                    for ex in ("squat", "deadlift", "benchpress", "pullup")]
    legend_items.append(Patch(facecolor="none", edgecolor="#1f4e79", label="rep marker"))
    fig.legend(handles=legend_items, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path = out_dir / "overview.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_per_set(rec: str, rec_dir: Path, out_dir: Path,
                   xlsx_info: Optional[dict], pad_s: float = 5.0) -> list[Path]:
    md = json.loads((rec_dir / "metadata.json").read_text()) if (rec_dir / "metadata.json").exists() else {}
    sets_md = md.get("kinect_sets", [])
    mk = json.loads((rec_dir / "markers.json").read_text()) if (rec_dir / "markers.json").exists() else []
    if isinstance(mk, dict):
        mk = mk.get("markers", [])
    sq = json.loads((rec_dir / "set_quality.json").read_text()) if (rec_dir / "set_quality.json").exists() else {"sets": []}
    sq_by_n = {s["set_number"]: s for s in sq.get("sets", [])}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    # Pre-load + filter each modality once for the full recording.
    panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mod in ("ecg", "emg", "eda", "acc_mag", "ppg_green"):
        if mod == "acc_mag":
            t, v = load_acc_mag(rec_dir)
            if len(v) > 50:
                v = _filter_bp(v, 0.5, 20.0, NATIVE_FS["ax"])
            panels[mod] = (t, v)
        else:
            df = load_csv(rec_dir, mod)
            if df is None or len(df) == 0:
                panels[mod] = (np.array([]), np.array([])); continue
            t = df["timestamp"].to_numpy()
            v = df[mod].to_numpy()
            fs = NATIVE_FS[mod]
            try:
                if mod == "ecg":
                    v = _filter_bp(v, 0.5, 40.0, fs, notch=50.0)
                elif mod == "emg":
                    v = emg_envelope(v, fs)
                elif mod == "eda":
                    v = _filter_lp(v, 5.0, fs)
                elif mod == "ppg_green":
                    v = _filter_bp(v, 0.5, 8.0, fs)
            except Exception:
                pass
            panels[mod] = (t, v)

    for i, s in enumerate(sets_md):
        sn = s["set_number"]
        t0, t1 = s["start_unix_time"] - pad_s, s["end_unix_time"] + pad_s
        ex = ""
        rpe = ""
        if xlsx_info:
            if i < len(xlsx_info.get("exercises", [])):
                ex = xlsx_info["exercises"][i]
            rpes = xlsx_info.get("rpes", [])
            if i < len(rpes) and pd.notna(rpes[i]):
                rpe = str(int(rpes[i]))

        fig, axes = plt.subplots(len(panels), 1, figsize=(13, 9), sharex=True)
        if len(panels) == 1:
            axes = [axes]
        for ax, (mod, (t, v)) in zip(axes, panels.items()):
            mask = (t >= t0) & (t <= t1) if len(t) else np.array([], dtype=bool)
            tx = t[mask] if mask.any() else np.array([])
            vx = v[mask] if mask.any() else np.array([])
            ax.axvspan(s["start_unix_time"], s["end_unix_time"],
                       color=EXERCISE_PALETTE.get(ex.lower(), "#bdc3c7"),
                       alpha=0.25, zorder=0)
            for e in mk:
                if "Rep:" in e.get("label", "") and t0 <= e["unix_time"] <= t1:
                    ax.axvline(e["unix_time"], color="#1f4e79", alpha=0.5,
                               linewidth=0.7, zorder=1)
            nice = {"ecg": "ECG", "emg": "EMG env", "eda": "EDA",
                    "acc_mag": "|acc|", "ppg_green": "PPG-g"}[mod]
            _mod_panel(ax, tx, vx, nice)
        axes[-1].set_xlabel("Unix time (s)")
        sq_s = sq_by_n.get(sn, {})
        flag_str = ", ".join(sq_s.get("flags", [])) or "ok"
        title = (f"recording_{rec} — set {sn} ({ex or '?'})  RPE={rpe or '-'}  "
                 f"dur={s['end_unix_time']-s['start_unix_time']:.1f}s  reps={sq_s.get('rep_count_markers', '?')}"
                 f"  | flags: {flag_str}")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        ex_short = (ex or "unknown").replace(" ", "")
        out_p = out_dir / f"set_{sn:02d}_{ex_short}.png"
        fig.savefig(out_p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_p)
    return out_paths


def write_report(rec: str, rec_dir: Path, out_dir: Path,
                 xlsx_info: Optional[dict], per_set_paths: list[Path]) -> Path:
    md = json.loads((rec_dir / "metadata.json").read_text()) if (rec_dir / "metadata.json").exists() else {}
    sets_md = md.get("kinect_sets", [])
    sq = json.loads((rec_dir / "set_quality.json").read_text()) if (rec_dir / "set_quality.json").exists() else {"sets": []}
    qr = json.loads((rec_dir / "quality_report.json").read_text()) if (rec_dir / "quality_report.json").exists() else {}
    ca = json.loads((rec_dir / "clock_alignment.json").read_text()) if (rec_dir / "clock_alignment.json").exists() else {}

    sq_by_n = {s["set_number"]: s for s in sq.get("sets", [])}

    lines = [
        f"# Segmentation QC — recording_{rec}",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Subject: {xlsx_info.get('subject', 'unknown') if xlsx_info else 'unknown'}",
        f"Source dataset: dataset_clean/",
        f"",
        f"## Plots",
        f"",
        f"- Overview: [overview.png](overview.png)",
        f"- Per-set zooms: `per_set/`",
    ]
    for p in per_set_paths:
        lines.append(f"  - [{p.name}](per_set/{p.name})")

    # Modality-source decisions
    lines += [
        "",
        "## Modality source choice (from clock_alignment.json)",
        "",
        "| modality | source | ds_score | mem_score |",
        "|----------|--------|----------|-----------|",
    ]
    for mod, info in qr.items():
        d = info.get("decision", {})
        ds_s = d.get("ds_score", "?")
        mem_s = d.get("mem_score", "?")
        ds_s_str = "inf" if ds_s == float("inf") else (f"{ds_s:.0f}" if isinstance(ds_s, (int, float)) else str(ds_s))
        mem_s_str = "inf" if mem_s == float("inf") else (f"{mem_s:.0f}" if isinstance(mem_s, (int, float)) else str(mem_s))
        lines.append(f"| {mod} | {d.get('choice', '-')} | {ds_s_str} | {mem_s_str} |")

    # Per-set table
    lines += [
        "",
        "## Per-set summary",
        "",
        "| set | exercise | dur (s) | reps | rpe | flags | ok |",
        "|-----|----------|---------|------|-----|-------|----|",
    ]
    for i, s in enumerate(sets_md):
        sn = s["set_number"]
        dur = s["end_unix_time"] - s["start_unix_time"]
        ex = ""
        rpe = ""
        if xlsx_info:
            if i < len(xlsx_info.get("exercises", [])):
                ex = xlsx_info["exercises"][i]
            rpes = xlsx_info.get("rpes", [])
            if i < len(rpes) and pd.notna(rpes[i]):
                rpe = str(int(rpes[i]))
        sq_s = sq_by_n.get(sn, {})
        nr = sq_s.get("rep_count_markers", "?")
        flags = ", ".join(sq_s.get("flags", [])) or "—"
        ok = "✓" if sq_s.get("ok_for_training") else "✗"
        lines.append(f"| {sn} | {ex} | {dur:.1f} | {nr} | {rpe} | {flags} | {ok} |")

    out_p = out_dir / "report.md"
    out_p.write_text("\n".join(lines), encoding="utf-8")
    return out_p


# ---------------------------------------------------------------------------
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", "-r", default=None,
                    help="recording id (e.g. 008). Default: all in dataset_clean/")
    ap.add_argument("--no-per-set", action="store_true",
                    help="skip per-set zoom plots (faster)")
    ap.add_argument("--output-root", default=None)
    args = ap.parse_args()

    out_root = Path(args.output_root) if args.output_root else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    xlsx = load_xlsx_per_recording()

    if args.recording:
        recs = [args.recording.zfill(3)]
    else:
        recs = sorted(d.name.replace("recording_", "")
                      for d in SRC_CLEAN.iterdir()
                      if d.is_dir() and d.name.startswith("recording_"))

    print(f"Rendering {len(recs)} recordings to {out_root}\n")
    t_start = time.time()
    for rec in recs:
        rec_dir = SRC_CLEAN / f"recording_{rec}"
        if not rec_dir.exists():
            print(f"  skip {rec}: not in dataset_clean/")
            continue
        out_dir = out_root / f"recording_{rec}"
        out_dir.mkdir(parents=True, exist_ok=True)
        per_set_dir = out_dir / "per_set"
        per_set_dir.mkdir(exist_ok=True)

        info = xlsx.get(int(rec))
        t0 = time.time()
        ov = render_overview(rec, rec_dir, out_dir, info)
        per_set_paths = []
        if not args.no_per_set:
            per_set_paths = render_per_set(rec, rec_dir, per_set_dir, info)
        write_report(rec, rec_dir, out_dir, info, per_set_paths)
        print(f"  rec_{rec}: overview + {len(per_set_paths)} per-set in {time.time()-t0:.1f}s")

    print(f"\nDone in {time.time()-t_start:.1f}s. Output: {out_root}")


if __name__ == "__main__":
    main()
