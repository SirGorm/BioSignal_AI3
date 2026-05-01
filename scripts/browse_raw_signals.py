"""
Interactive raw-signal browser for the original (un-aligned) recordings.

Loads all 6 modalities directly from `dataset/` or `dataset_memory/` and
renders a stacked Plotly figure with synchronised zoom/pan across rows.
Output is a self-contained HTML opened in the default browser.

Sources
-------
- ``dataset``        — PC-clock recordings (raw biosignals, but
                         ``temperature.csv`` is empty in all 9 recordings).
- ``dataset_memory`` — sensor-onboard recordings on a different clock
                         (offset documented in
                         ``dataset_aligned/alignment_offsets.json``).
                         All 9 memory dumps have populated temperature.
- ``both``           — render two HTMLs side-by-side, one per source.

Layout
------
Stacked subplots with ``shared_xaxes=True`` so zoom/pan is synchronised:
  Row 1: ECG (500 Hz)
  Row 2: EMG (2000 Hz, decimated to ~100k)
  Row 3: EDA (50 Hz)
  Row 4: Temperature (1 Hz)
  Row 5: Acc magnitude (sqrt(ax²+ay²+az²), 100 Hz)
  Row 6: PPG (4 wavelengths — toggleable via legend, 100 Hz)

X-axis shows wall-clock UTC; hover shows Unix epoch + value.

Usage
-----
    python scripts/browse_raw_signals.py --recording 012
    python scripts/browse_raw_signals.py --recording 012 --source dataset_memory
    python scripts/browse_raw_signals.py --recording 012 --source both
    python scripts/browse_raw_signals.py --all --no-open

Output: ``inspections/raw_browser/recording_NNN/<source>.html``
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATASET_ROOT = Path(r"C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset")
MEMORY_ROOT = Path(r"C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset_memory")
OUT_ROOT = Path(__file__).resolve().parents[1] / "inspections" / "raw_browser"

MAX_POINTS_DEFAULT = 100_000

ROW_SPEC = [
    ("ecg",        "ECG (500 Hz)",        "#c0392b"),
    ("emg",        "EMG (2000 Hz)",       "#8e44ad"),
    ("eda",        "EDA (50 Hz)",         "#2980b9"),
    ("temp",       "Temperature (1 Hz)",  "#e67e22"),
    ("acc_mag",    "Acc-mag (100 Hz)",    "#16a085"),
    ("ppg",        "PPG (100 Hz)",        "#27ae60"),
]
PPG_CHANNELS = [
    ("ppg_green", "#27ae60"),
    ("ppg_red",   "#e74c3c"),
    ("ppg_blue",  "#2980b9"),
    ("ppg_ir",    "#7f8c8d"),
]


def _decimate(t: np.ndarray, v: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Stride-based decimation. For browser performance only — exact
    waveform shape preserved at zoom levels visible at the chosen stride."""
    n = len(v)
    if n <= max_points:
        return t, v
    stride = max(1, n // max_points)
    return t[::stride], v[::stride]


def _read_csv(path: Path, col: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "timestamp" not in df.columns or col not in df.columns:
        return None
    df = df.dropna(subset=["timestamp", col])
    if df.empty:
        return None
    return df


def load_acc_mag(rec_dir: Path) -> Optional[pd.DataFrame]:
    """Load ax/ay/az and compute magnitude on the ax timestamp grid."""
    parts = {}
    for axis in ("ax", "ay", "az"):
        df = _read_csv(rec_dir / f"{axis}.csv", axis)
        if df is None:
            return None
        parts[axis] = df
    n = min(len(parts["ax"]), len(parts["ay"]), len(parts["az"]))
    t = parts["ax"]["timestamp"].to_numpy(dtype=float)[:n]
    mag = np.sqrt(
        parts["ax"]["ax"].to_numpy(dtype=float)[:n] ** 2
        + parts["ay"]["ay"].to_numpy(dtype=float)[:n] ** 2
        + parts["az"]["az"].to_numpy(dtype=float)[:n] ** 2
    )
    return pd.DataFrame({"timestamp": t, "acc_mag": mag})


def load_modality(rec_dir: Path, modality: str) -> Optional[pd.DataFrame]:
    """Generic loader for ECG/EMG/EDA/temp/PPG; acc handled separately."""
    if modality == "temp":
        return _read_csv(rec_dir / "temperature.csv", "temperature")
    if modality == "acc_mag":
        return load_acc_mag(rec_dir)
    return _read_csv(rec_dir / f"{modality}.csv", modality)


def load_markers(rec_dir: Path) -> list[dict]:
    p = rec_dir / "markers.json"
    if not p.exists():
        return []
    try:
        with p.open() as f:
            data = json.load(f)
        return data.get("markers", [])
    except Exception:
        return []


def build_figure(
    rec_dir: Path,
    title: str,
    max_points: int,
) -> go.Figure:
    fig = make_subplots(
        rows=len(ROW_SPEC),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        subplot_titles=[label for _, label, _ in ROW_SPEC],
    )

    for i, (mod, _, color) in enumerate(ROW_SPEC, start=1):
        if mod == "ppg":
            for ch_name, ch_color in PPG_CHANNELS:
                print(f"[browse]   loading {ch_name}...", flush=True)
                df = _read_csv(rec_dir / f"{ch_name}.csv", ch_name)
                if df is None:
                    continue
                t = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                t_arr, v_arr = _decimate(t.values, df[ch_name].to_numpy(dtype=float), max_points)
                fig.add_trace(
                    go.Scattergl(
                        x=t_arr, y=v_arr, name=ch_name,
                        line=dict(color=ch_color, width=1),
                        mode="lines",
                        legendgroup="ppg",
                        showlegend=True,
                        hovertemplate=(
                            f"<b>{ch_name}</b><br>"
                            "time=%{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                            "value=%{y:.6g}<extra></extra>"
                        ),
                    ),
                    row=i, col=1,
                )
            continue

        print(f"[browse]   loading {mod}...", flush=True)
        df = load_modality(rec_dir, mod)
        if df is None or df.empty:
            fig.add_annotation(
                text=f"(no {mod} data)", xref=f"x{i} domain", yref=f"y{i} domain",
                x=0.5, y=0.5, showarrow=False, font=dict(color="#888", size=12),
                row=i, col=1,
            )
            continue

        col_name = "acc_mag" if mod == "acc_mag" else (
            "temperature" if mod == "temp" else mod
        )
        t = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        t_arr, v_arr = _decimate(t.values, df[col_name].to_numpy(dtype=float), max_points)
        fig.add_trace(
            go.Scattergl(
                x=t_arr, y=v_arr, name=mod,
                line=dict(color=color, width=1),
                mode="lines",
                showlegend=False,
                hovertemplate=(
                    f"<b>{mod}</b><br>"
                    "time=%{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                    "value=%{y:.6g}<extra></extra>"
                ),
            ),
            row=i, col=1,
        )

    markers = load_markers(rec_dir)
    if markers:
        for entry in markers:
            t_unix = entry.get("unix_time")
            label = entry.get("label", "")
            if t_unix is None:
                continue
            t_dt = pd.to_datetime(float(t_unix), unit="s", utc=True)
            color = "#27ae60" if "Start" in label else (
                "#c0392b" if "End" in label else "#7f8c8d"
            )
            for row in range(1, len(ROW_SPEC) + 1):
                fig.add_vline(
                    x=t_dt, line=dict(color=color, width=0.6, dash="dot"),
                    opacity=0.45, row=row, col=1,
                )

    fig.update_layout(
        title=title,
        height=200 * len(ROW_SPEC),
        margin=dict(l=60, r=20, t=70, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        dragmode="zoom",
    )
    fig.update_xaxes(title_text="Time (UTC)", row=len(ROW_SPEC), col=1)
    return fig


def render_one(
    rec_id: str,
    source: str,
    max_points: int,
    open_browser: bool,
) -> Optional[Path]:
    rec_id = rec_id.lstrip("0").zfill(3)

    if source == "dataset":
        rec_dir = DATASET_ROOT / f"recording_{rec_id}"
    elif source == "dataset_memory":
        rec_dir = MEMORY_ROOT / f"recording_{rec_id}_memory"
    else:
        raise ValueError(f"unknown source: {source}")

    if not rec_dir.exists():
        print(f"[browse]   skip {source}: {rec_dir} not found")
        return None

    out_dir = OUT_ROOT / f"recording_{rec_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source}.html"

    title = f"recording_{rec_id} — raw signals from {source}"
    print(f"[browse] {title}", flush=True)
    print(f"[browse]   dir: {rec_dir}", flush=True)
    fig = build_figure(rec_dir, title, max_points)
    print(f"[browse]   writing HTML (this can take 30–60 s)...", flush=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[browse]   wrote {out_path}", flush=True)
    if open_browser:
        uri = out_path.resolve().as_uri()
        print(f"[browse]   opening {uri}", flush=True)
        webbrowser.open(uri)
    return out_path


def discover_recordings(source: str) -> list[str]:
    if source == "dataset":
        root = DATASET_ROOT
        prefix = "recording_"
        suffix = ""
    else:
        root = MEMORY_ROOT
        prefix = "recording_"
        suffix = "_memory"
    if not root.exists():
        return []
    ids = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith(prefix) and name.endswith(suffix):
            mid = name[len(prefix): len(name) - len(suffix) if suffix else len(name)]
            if mid.isdigit():
                ids.append(mid)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--recording", "-r", default=None,
                    help="Recording id, e.g. 012. Omit with --all to render all.")
    ap.add_argument("--source", "-s", default="both",
                    choices=["dataset", "dataset_memory", "both"],
                    help="Which raw source to browse (default: both).")
    ap.add_argument("--all", action="store_true",
                    help="Render every recording present in the chosen source(s).")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS_DEFAULT,
                    help=f"Per-trace decimation target (default {MAX_POINTS_DEFAULT}).")
    ap.add_argument("--no-open", action="store_true",
                    help="Skip auto-launching the browser.")
    args = ap.parse_args()

    if not args.all and not args.recording:
        ap.error("Provide --recording <id> or --all.")

    sources = ["dataset", "dataset_memory"] if args.source == "both" else [args.source]

    if args.all:
        rec_ids: list[str] = []
        for s in sources:
            for rid in discover_recordings(s):
                if rid not in rec_ids:
                    rec_ids.append(rid)
        rec_ids.sort()
    else:
        rec_ids = [args.recording]

    open_browser = not args.no_open
    # If rendering many files, do not flood the browser.
    if args.all and len(rec_ids) * len(sources) > 2:
        open_browser = False

    written = 0
    for rid in rec_ids:
        for s in sources:
            out = render_one(rid, s, args.max_points, open_browser)
            if out:
                written += 1

    print(f"\n[browse] Done. {written} HTML file(s) under {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
