"""
Interactive QC viewer for the 6 biosignal modalities.

Loads a recording from `dataset/recording_NNN/` and shows all 6 modalities
(ECG, EMG, EDA, Temperature, Acc-mag, PPG-green) stacked in a single window
with shared x-axis. Set boundaries (from metadata.json), per-rep markers
(from markers.json), and exercise/RPE labels (from Participants.xlsx) are
overlaid on every subplot.

The user can:
  - Pan/zoom via the matplotlib toolbar (built-in)
  - Move between recordings with Prev/Next buttons or Left/Right arrow keys
  - Toggle per-modality OK / REJECT with buttons or number keys 1-6
  - Save flags to disk with the Save button or 'S' key (also auto-saves on
    recording change)
  - Add a free-text note per recording via the Note button or 'N' key

Flags are persisted to `inspections/manual_qc/qc_flags.json` so multiple
sessions can resume.

Usage:
  python scripts/manual_qc_viewer.py
  python scripts/manual_qc_viewer.py --recording 014
  python scripts/manual_qc_viewer.py --dataset path/to/dataset

Default dataset is the repo's `dataset_aligned/` (CLAUDE.md's authoritative
training input — biosignals copied verbatim from the raw dataset, with
temperature offset-corrected from the memory-logged variant). Pass
--dataset to point at the original raw `dataset/` if needed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import load_all_biosignals, load_metadata  # noqa: E402
from src.data.participants import load_participants  # noqa: E402
from src.eval.plot_style import apply_style, despine  # noqa: E402

apply_style()


DEFAULT_DATASET = REPO_ROOT / "dataset_aligned"
QC_FLAGS_PATH = REPO_ROOT / "inspections" / "manual_qc" / "qc_flags.json"

MODALITIES = [
    ("ecg", "ecg", "ECG (500 Hz)", "tab:red"),
    ("emg", "emg", "EMG (2000 Hz)", "tab:purple"),
    ("eda", "eda", "EDA (50 Hz)", "tab:blue"),
    ("temperature", "temperature", "Temperature (1 Hz)", "tab:orange"),
    ("imu", "acc_mag", "Acc-magnitude (100 Hz)", "tab:green"),
    ("ppg_green", "ppg_green", "PPG-green (100 Hz)", "tab:olive"),
]

# Hard cap on plotted points per modality. Above this, we min/max-decimate so
# matplotlib stays responsive even for the 4.9M-sample EMG trace.
MAX_PLOT_POINTS = 30_000

REP_MARKER_RE = re.compile(r"^Set:(\d+)_Rep:(\d+)$")
SET_START_RE = re.compile(r"^Set:(\d+)_Start$")
SET_END_RE = re.compile(r"^Set_(\d+)_End$")


def list_recordings(dataset_dir: Path) -> list[Path]:
    """Return recording_NNN directories sorted by numeric id."""
    rec_dirs = sorted(
        d for d in dataset_dir.glob("recording_*") if d.is_dir() and d.name[10:].isdigit()
    )
    return rec_dirs


def decimate_for_plot(t: np.ndarray, y: np.ndarray, max_points: int = MAX_PLOT_POINTS):
    """Min/max envelope decimation to keep transient peaks visible.

    Returns interleaved (t_pairs, y_pairs) where consecutive (min, max)
    samples per bin preserve the envelope better than naive stride sampling
    when zoomed out.
    """
    n = len(y)
    if n <= max_points:
        return t, y

    bin_size = int(np.ceil(n / (max_points // 2)))
    n_full_bins = n // bin_size
    if n_full_bins < 2:
        return t, y

    head = y[: n_full_bins * bin_size].reshape(n_full_bins, bin_size)
    t_head = t[: n_full_bins * bin_size].reshape(n_full_bins, bin_size)
    idx_min = head.argmin(axis=1)
    idx_max = head.argmax(axis=1)
    rows = np.arange(n_full_bins)
    t_pairs = np.empty(n_full_bins * 2, dtype=t.dtype)
    y_pairs = np.empty(n_full_bins * 2, dtype=y.dtype)
    t_pairs[0::2] = t_head[rows, idx_min]
    t_pairs[1::2] = t_head[rows, idx_max]
    y_pairs[0::2] = head[rows, idx_min]
    y_pairs[1::2] = head[rows, idx_max]

    if n_full_bins * bin_size < n:
        tail_y = y[n_full_bins * bin_size :]
        tail_t = t[n_full_bins * bin_size :]
        t_pairs = np.concatenate([t_pairs, [tail_t[tail_y.argmin()], tail_t[tail_y.argmax()]]])
        y_pairs = np.concatenate([y_pairs, [tail_y.min(), tail_y.max()]])

    order = np.argsort(t_pairs)
    return t_pairs[order], y_pairs[order]


def load_qc_flags() -> dict:
    if QC_FLAGS_PATH.exists():
        return json.loads(QC_FLAGS_PATH.read_text(encoding="utf-8"))
    return {}


def save_qc_flags(flags: dict) -> None:
    QC_FLAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_FLAGS_PATH.write_text(json.dumps(flags, indent=2), encoding="utf-8")


def parse_markers(markers_path: Path) -> dict:
    """Group markers by set: {set_num: {'start': t, 'end': t, 'reps': [t,...]}}.

    Times are Unix epoch seconds.
    """
    if not markers_path.exists():
        return {}
    raw = json.loads(markers_path.read_text(encoding="utf-8"))
    entries = raw.get("markers", []) if isinstance(raw, dict) else raw
    grouped: dict[int, dict] = {}

    for entry in entries:
        label = entry.get("label", "")
        t_unix = float(entry["unix_time"])

        m = SET_START_RE.match(label)
        if m:
            n = int(m.group(1))
            grouped.setdefault(n, {"reps": []})["start"] = t_unix
            continue

        m = SET_END_RE.match(label)
        if m:
            n = int(m.group(1))
            grouped.setdefault(n, {"reps": []})["end"] = t_unix
            continue

        m = REP_MARKER_RE.match(label)
        if m:
            n = int(m.group(1))
            grouped.setdefault(n, {"reps": []}).setdefault("reps", []).append(t_unix)

    return grouped


class QCViewer:
    def __init__(self, dataset_dir: Path, start_recording: int | None = None):
        self.dataset_dir = dataset_dir
        self.recordings = list_recordings(dataset_dir)
        if not self.recordings:
            raise SystemExit(f"No recordings found under {dataset_dir}")

        # Pick starting index
        if start_recording is not None:
            target = f"recording_{start_recording:03d}"
            for i, p in enumerate(self.recordings):
                if p.name == target:
                    self.idx = i
                    break
            else:
                raise SystemExit(f"--recording {start_recording} not found")
        else:
            self.idx = 0

        # Load participants once. dataset_aligned/ doesn't carry the xlsx,
        # so fall back to the raw dataset path next to it.
        xlsx_candidates = [
            dataset_dir / "Participants" / "Participants.xlsx",
            Path("C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/Participants.xlsx"),
        ]
        self.participants = {}
        for xlsx in xlsx_candidates:
            if xlsx.exists():
                self.participants = load_participants(xlsx)
                break

        # Persistent flags
        self.flags = load_qc_flags()

        self._build_figure()
        self._render_current()

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("Manual Biosignal QC")

        # Layout: 6 plot rows (height 1 each) + button row
        gs = self.fig.add_gridspec(
            nrows=7,
            ncols=8,
            height_ratios=[1, 1, 1, 1, 1, 1, 0.35],
            hspace=0.18,
            wspace=0.4,
            left=0.06,
            right=0.985,
            top=0.93,
            bottom=0.04,
        )

        self.axes = []
        for row, (mod, _col, label, _color) in enumerate(MODALITIES):
            sharex = self.axes[0] if self.axes else None
            ax = self.fig.add_subplot(gs[row, :], sharex=sharex)
            ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelbottom=(row == len(MODALITIES) - 1), labelsize=8)
            ax.grid(alpha=0.25, linewidth=0.5)
            despine(ax=ax)
            self.axes.append(ax)

        self.axes[-1].set_xlabel("Session time (s)")

        # Button row: prev | next | save | note | OK1..OK6
        btn_row = gs[6, :]
        nb = 10  # number of button cells
        button_gs = btn_row.subgridspec(1, nb, wspace=0.3)

        def add_btn(col: int, text: str, cb, color: str = "#dddddd"):
            ax = self.fig.add_subplot(button_gs[0, col])
            b = Button(ax, text, color=color, hovercolor="#cccccc")
            b.on_clicked(cb)
            return ax, b

        add_btn(0, "<< Prev", lambda _e: self._step(-1))
        add_btn(1, "Next >>", lambda _e: self._step(+1))
        add_btn(2, "Save", lambda _e: self._save_and_status("Saved."))
        add_btn(3, "Note", lambda _e: self._prompt_note())

        self.modality_btns = []
        for i, (mod, _col, label, _c) in enumerate(MODALITIES):
            short = label.split(" ")[0]
            ax, btn = add_btn(
                4 + i, f"{i + 1} {short}\nOK", lambda _e, m=mod: self._toggle(m)
            )
            self.modality_btns.append((mod, ax, btn))

        # Top header text
        self.title_text = self.fig.text(
            0.5, 0.97, "", ha="center", fontsize=12, fontweight="bold"
        )
        self.status_text = self.fig.text(
            0.5, 0.945, "", ha="center", fontsize=9, color="#444444"
        )

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------
    # Per-recording rendering
    # ------------------------------------------------------------------

    def _current_dir(self) -> Path:
        return self.recordings[self.idx]

    def _current_id(self) -> str:
        return self._current_dir().name

    def _ensure_flag_entry(self) -> dict:
        rec_id = self._current_id()
        if rec_id not in self.flags:
            self.flags[rec_id] = {
                "ecg": "ok",
                "emg": "ok",
                "eda": "ok",
                "temperature": "ok",
                "imu": "ok",
                "ppg_green": "ok",
                "note": "",
                "reviewed_at": None,
            }
        return self.flags[rec_id]

    def _render_current(self):
        rec_dir = self._current_dir()
        rec_id = rec_dir.name
        rec_num = int(rec_id.split("_")[1])

        # Load all 6 modalities (temperature may be empty)
        sigs = load_all_biosignals(rec_dir)

        # Set boundaries from metadata.json
        try:
            metadata = load_metadata(rec_dir)
            kinect_sets = metadata.get("kinect_sets", [])
            data_start = metadata.get(
                "data_start_unix_time",
                metadata.get("recording_start_unix_time", 0.0),
            )
        except FileNotFoundError:
            kinect_sets = []
            data_start = 0.0

        # Markers
        rep_groups = parse_markers(rec_dir / "markers.json")

        # Participants info
        pinfo = self.participants.get(rec_num, {})
        exercises = pinfo.get("exercises", [None] * 12)
        rpes = pinfo.get("rpe", [None] * 12)
        subject = pinfo.get("subject_id", "(unknown subject)")

        # Reference time = session-relative start
        if data_start <= 1e9:
            # Fall back to earliest sample across modalities
            candidates = [df["timestamp"].iloc[0] for df in sigs.values() if len(df)]
            data_start = min(candidates) if candidates else 0.0

        # Render each modality
        for ax, (mod_key, col, label, color) in zip(self.axes, MODALITIES):
            ax.clear()
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.25, linewidth=0.5)

            df = sigs.get(mod_key, pd.DataFrame())
            if len(df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "(no data)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=11,
                )
            else:
                t = df["timestamp"].to_numpy(dtype=np.float64) - data_start
                y = df[col].to_numpy(dtype=np.float64)
                t_d, y_d = decimate_for_plot(t, y)
                ax.plot(t_d, y_d, color=color, linewidth=0.6)

            ax.tick_params(labelbottom=(ax is self.axes[-1]), labelsize=8)

        self.axes[-1].set_xlabel("Session time (s)")

        # Overlay set spans + rep markers + exercise/rpe text on every axis
        for set_idx, kset in enumerate(kinect_sets):
            t_start = kset["start_unix_time"] - data_start
            t_end = kset["end_unix_time"] - data_start
            set_n = kset.get("set_number", set_idx + 1)
            ex = exercises[set_n - 1] if set_n - 1 < len(exercises) else None
            rpe = rpes[set_n - 1] if set_n - 1 < len(rpes) else None
            band_color = "#a8d5a8" if (set_n % 2 == 0) else "#cce5ff"

            for ax in self.axes:
                ax.axvspan(t_start, t_end, color=band_color, alpha=0.35, zorder=0)

            # Annotate exercise + RPE on top axis only
            label_txt = f"S{set_n}"
            if ex:
                label_txt += f" {ex}"
            if rpe is not None:
                label_txt += f"\nRPE={rpe}"
            self.axes[0].text(
                (t_start + t_end) / 2,
                0.95,
                label_txt,
                transform=self.axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.85),
            )

        # Rep markers — thin lines on every axis
        for set_n, group in rep_groups.items():
            for rep_t in group.get("reps", []):
                rt = rep_t - data_start
                for ax in self.axes:
                    ax.axvline(rt, color="black", linewidth=0.3, alpha=0.45, zorder=1)

        # Set xlim to full session
        if len(self.axes[0].lines):
            xs = []
            for ax in self.axes:
                if ax.lines:
                    xs.append(ax.lines[0].get_xdata()[0])
                    xs.append(ax.lines[0].get_xdata()[-1])
            if xs:
                self.axes[0].set_xlim(min(xs), max(xs))

        # Header
        n_sets = len(kinect_sets)
        n_reps_total = sum(len(g.get("reps", [])) for g in rep_groups.values())
        self.title_text.set_text(
            f"{rec_id} — subject={subject}    [{self.idx + 1}/{len(self.recordings)}]"
        )
        self.status_text.set_text(
            f"sets={n_sets}  reps={n_reps_total}  "
            f"path={rec_dir}"
        )

        # Sync button labels with flag state
        self._refresh_modality_buttons()
        self.fig.canvas.draw_idle()

    def _refresh_modality_buttons(self):
        entry = self._ensure_flag_entry()
        for i, ((mod, _col, label, _c), (_m, ax, btn)) in enumerate(
            zip(MODALITIES, self.modality_btns)
        ):
            short = label.split(" ")[0]
            state = entry.get(mod, "ok")
            text = f"{i + 1} {short}\n{state.upper()}"
            btn.label.set_text(text)
            ax.set_facecolor("#b8e6b8" if state == "ok" else "#f4a8a8")
        # Also reflect note presence in status
        note = self._ensure_flag_entry().get("note", "")
        if note:
            cur = self.status_text.get_text()
            self.status_text.set_text(cur + f"\nnote: {note[:120]}")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle(self, modality: str):
        entry = self._ensure_flag_entry()
        entry[modality] = "reject" if entry.get(modality, "ok") == "ok" else "ok"
        entry["reviewed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self._refresh_modality_buttons()

    def _step(self, delta: int):
        self._save_and_status(None)
        new_idx = (self.idx + delta) % len(self.recordings)
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self._render_current()

    def _save_and_status(self, msg: str | None):
        save_qc_flags(self.flags)
        if msg:
            self.status_text.set_text(self.status_text.get_text() + f"  [{msg}]")
            self.fig.canvas.draw_idle()

    def _prompt_note(self):
        # Simple console prompt — keeps the dependency surface tiny.
        entry = self._ensure_flag_entry()
        cur = entry.get("note", "")
        print(f"\n[note for {self._current_id()}] current: {cur!r}")
        try:
            new = input("enter new note (blank = keep, '-' = clear): ").rstrip()
        except EOFError:
            return
        if new == "-":
            entry["note"] = ""
        elif new:
            entry["note"] = new
        entry["reviewed_at"] = dt.datetime.now().isoformat(timespec="seconds")
        self._save_and_status("note saved.")
        self._refresh_modality_buttons()

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._step(+1)
        elif event.key in ("left", "p"):
            self._step(-1)
        elif event.key == "s":
            self._save_and_status("Saved.")
        elif event.key in {str(i) for i in range(1, 7)}:
            mod = MODALITIES[int(event.key) - 1][0]
            self._toggle(mod)
        elif event.key == "?":
            print(self._help_text())

    @staticmethod
    def _help_text() -> str:
        return (
            "Keyboard shortcuts:\n"
            "  ← / →  or  p / n   prev / next recording\n"
            "  1..6              toggle modality OK/REJECT (ecg, emg, eda, temp, acc, ppg)\n"
            "  s                 save flags\n"
            "  ?                 print this help\n"
            "Note: use the matplotlib toolbar for pan/zoom (magnifier + hand icons)."
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root containing recording_NNN/ subdirs (default: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--recording",
        type=int,
        default=None,
        help="Optional: recording number to start at (e.g. 14 for recording_014).",
    )
    args = p.parse_args(argv)

    print(QCViewer._help_text())
    print(f"\nFlags persisted to {QC_FLAGS_PATH}\n")

    viewer = QCViewer(args.dataset, start_recording=args.recording)
    plt.show()
    # Final save on close
    save_qc_flags(viewer.flags)


if __name__ == "__main__":
    main()
