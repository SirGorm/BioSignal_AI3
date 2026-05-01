"""
Compare raw vs NeuroKit2-cleaned ECG side-by-side from both raw sources.

Loads ECG from ``dataset/recording_NNN/ecg.csv`` and
``dataset_memory/recording_NNN_memory/ecg.csv`` and renders an interactive
4-row Plotly figure with synchronised zoom on a session-relative time axis
(seconds from each source's own first sample, since the two sources are on
different clocks — see CLAUDE.md "Datakilde og synkronisering").

Rows:
  1. dataset/         raw ECG
  2. dataset/         ECG cleaned by ``nk.ecg_clean(method=<method>)``
  3. dataset_memory/  raw ECG
  4. dataset_memory/  ECG cleaned by ``nk.ecg_clean(method=<method>)``

R-peaks detected on the cleaned signal (Pan & Tompkins 1985 via NeuroKit2,
Makowski et al. 2021) are overlaid as red dots on the cleaned rows.

Usage
-----
    python scripts/compare_ecg_filtering.py --recording 012
    python scripts/compare_ecg_filtering.py --recording 012 --method biosppy
    python scripts/compare_ecg_filtering.py --recording 012 --max-points 50000
    python scripts/compare_ecg_filtering.py --recording 012 --no-open

Output: ``inspections/ecg_filter_qc/recording_NNN/<method>.html``

References
----------
- Makowski, D. et al. (2021). NeuroKit2: A Python toolbox for
  neurophysiological signal processing. Behavior Research Methods, 53(4),
  1689-1696.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
  IEEE TBME, BME-32(3), 230-236.
"""

from __future__ import annotations

import argparse
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
OUT_ROOT = Path(__file__).resolve().parents[1] / "inspections" / "ecg_filter_qc"

FS_ECG = 500
MAX_POINTS_DEFAULT = 80_000  # per trace — keeps browser responsive

VALID_METHODS = (
    "neurokit", "biosppy", "pantompkins1985", "hamilton2002",
    "elgendi2010", "engzeemod2012",
)


def _decimate(t: np.ndarray, v: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(v)
    if n <= max_points:
        return t, v
    stride = max(1, n // max_points)
    return t[::stride], v[::stride]


def load_ecg(rec_dir: Path) -> Optional[pd.DataFrame]:
    p = rec_dir / "ecg.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty or "ecg" not in df.columns:
        return None
    df = df.dropna(subset=["timestamp", "ecg"])
    return df if not df.empty else None


def clean_and_detect(
    ecg: np.ndarray, fs: int, method: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cleaned_signal, r_peak_indices)."""
    import neurokit2 as nk
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs, method=method)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="pantompkins1985")
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return np.asarray(cleaned, dtype=float), peaks


def add_ecg_row(
    fig: go.Figure,
    row: int,
    label: str,
    t_rel: np.ndarray,
    values: np.ndarray,
    color: str,
    max_points: int,
    r_peaks_idx: Optional[np.ndarray] = None,
) -> None:
    t_d, v_d = _decimate(t_rel, values, max_points)
    fig.add_trace(
        go.Scattergl(
            x=t_d, y=v_d, name=label,
            mode="lines", line=dict(color=color, width=1),
            showlegend=False,
            hovertemplate=(f"<b>{label}</b><br>t=%{{x:.3f}} s<br>"
                           "value=%{y:.6g}<extra></extra>"),
        ),
        row=row, col=1,
    )
    if r_peaks_idx is not None and len(r_peaks_idx):
        valid = r_peaks_idx[(r_peaks_idx >= 0) & (r_peaks_idx < len(t_rel))]
        if len(valid):
            fig.add_trace(
                go.Scattergl(
                    x=t_rel[valid], y=values[valid],
                    mode="markers",
                    marker=dict(color="#c0392b", size=4, symbol="circle"),
                    name=f"{label} R-peaks", showlegend=False,
                    hovertemplate="R-peak<br>t=%{x:.3f} s<extra></extra>",
                ),
                row=row, col=1,
            )


def render(rec_id: str, method: str, max_points: int, open_browser: bool) -> Path:
    rec_id = rec_id.lstrip("0").zfill(3)

    sources = {
        "dataset":        DATASET_ROOT / f"recording_{rec_id}",
        "dataset_memory": MEMORY_ROOT / f"recording_{rec_id}_memory",
    }

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=[
            "dataset/ — raw ECG",
            f"dataset/ — cleaned (nk.ecg_clean method='{method}') + R-peaks",
            "dataset_memory/ — raw ECG",
            f"dataset_memory/ — cleaned (nk.ecg_clean method='{method}') + R-peaks",
        ],
    )

    rows_for_source = {"dataset": (1, 2), "dataset_memory": (3, 4)}
    summary_lines = []

    for src_name, rec_dir in sources.items():
        raw_row, clean_row = rows_for_source[src_name]
        if not rec_dir.exists():
            print(f"[ecg-cmp] skip {src_name}: {rec_dir} not found", flush=True)
            continue
        print(f"[ecg-cmp] loading ECG from {src_name}...", flush=True)
        df = load_ecg(rec_dir)
        if df is None:
            print(f"[ecg-cmp]   no ECG data in {src_name}", flush=True)
            continue

        t_unix = df["timestamp"].to_numpy(dtype=float)
        ecg = df["ecg"].to_numpy(dtype=float)
        t0 = float(t_unix[0])
        t_rel = t_unix - t0
        duration_s = float(t_rel[-1])

        print(f"[ecg-cmp]   {src_name}: {len(ecg)} samples, "
              f"{duration_s:.1f} s @ ~{len(ecg)/max(duration_s,1e-9):.0f} Hz",
              flush=True)

        add_ecg_row(fig, raw_row, f"{src_name} raw", t_rel, ecg,
                    "#7f8c8d", max_points)

        print(f"[ecg-cmp]   cleaning {src_name} with nk.ecg_clean(method='{method}')...",
              flush=True)
        try:
            cleaned, r_peaks = clean_and_detect(ecg, FS_ECG, method)
        except ImportError:
            print("[ecg-cmp]   ERROR: neurokit2 not installed. "
                  "Run `pip install neurokit2`.", flush=True)
            sys.exit(2)
        except Exception as e:
            print(f"[ecg-cmp]   cleaning failed: {e}", flush=True)
            cleaned = ecg.copy()
            r_peaks = np.array([], dtype=int)

        add_ecg_row(fig, clean_row, f"{src_name} cleaned", t_rel, cleaned,
                    "#2980b9", max_points, r_peaks_idx=r_peaks)

        hr_bpm = float("nan")
        if len(r_peaks) >= 2:
            rr_s = np.diff(t_rel[r_peaks])
            rr_s = rr_s[(rr_s > 0.3) & (rr_s < 2.0)]
            if len(rr_s):
                hr_bpm = float(60.0 / np.mean(rr_s))
        summary_lines.append(
            f"{src_name}: n={len(ecg)}, dur={duration_s:.1f}s, "
            f"R-peaks={len(r_peaks)}, mean HR~{hr_bpm:.1f} bpm, t0_unix={t0:.3f}"
        )

    fig.update_layout(
        title=(f"recording_{rec_id} — raw vs NeuroKit2-cleaned ECG "
               f"(method='{method}')"),
        height=240 * 4,
        margin=dict(l=60, r=20, t=80, b=40),
        hovermode="x unified",
        template="plotly_white",
        dragmode="zoom",
    )
    fig.update_xaxes(title_text="Session-relative time (s)", row=4, col=1)
    for r in (1, 2, 3, 4):
        fig.update_yaxes(title_text="ECG", row=r, col=1)

    out_dir = OUT_ROOT / f"recording_{rec_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method}.html"

    print(f"[ecg-cmp] writing HTML...", flush=True)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    print(f"[ecg-cmp] wrote {out_path}", flush=True)
    print("[ecg-cmp] summary:")
    for line in summary_lines:
        print(f"  - {line}")

    if open_browser:
        uri = out_path.resolve().as_uri()
        print(f"[ecg-cmp] opening {uri}", flush=True)
        webbrowser.open(uri)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--recording", "-r", required=True,
                    help="Recording id, e.g. 012.")
    ap.add_argument("--method", "-m", default="neurokit", choices=VALID_METHODS,
                    help="nk.ecg_clean cleaning method (default: neurokit).")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS_DEFAULT,
                    help=f"Per-trace decimation target (default {MAX_POINTS_DEFAULT}).")
    ap.add_argument("--no-open", action="store_true",
                    help="Skip auto-launching the browser.")
    args = ap.parse_args()

    render(args.recording, args.method, args.max_points, not args.no_open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
