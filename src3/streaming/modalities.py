"""Per-modality streaming feature extractors.

Single file replaces src/streaming/{ecg,emg,eda,acc,ppg,temp}_streaming.py
because the per-modality logic is mostly: causal filter chain → window
buffer → call the same window_features function used offline.

EMG envelope follows the project's "rå-NN-vei": 20–450 Hz BP + 50 Hz notch
→ square → 50 ms moving-average → sqrt → linear interp to 100 Hz grid.
The feature-vei (MNF/MDF/Dimitrov) needs full bandwidth and runs on the
raw 2 kHz CSV — NOT on the envelope (see CLAUDE.md).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src3.features import acc as acc_feats
from src3.features import emg as emg_feats
from src3.features import ppg as ppg_feats
from src3.features import temp as temp_feats
from src3.streaming.filters import (
    FilterChain, causal_bandpass, causal_lowpass, causal_notch,
)


@dataclass
class _RingWindow:
    win_size: int
    _buf: deque

    def __init__(self, win_size: int):
        self.win_size = win_size
        self._buf = deque(maxlen=win_size)

    def push(self, x: np.ndarray) -> None:
        for v in np.atleast_1d(x):
            self._buf.append(float(v))

    def ready(self) -> bool:
        return len(self._buf) >= self.win_size

    def view(self) -> np.ndarray:
        return np.fromiter(self._buf, dtype=float, count=len(self._buf))


# --- EMG --------------------------------------------------------------------


class EMGStream:
    """Streaming envelope generator. Outputs RMS-envelope at 100 Hz from
    2 kHz raw EMG (matches src/labeling/align.py:emg_envelope)."""

    def __init__(self, fs_native: int = 2000, fs_target: int = 100,
                 envelope_ms: float = 50.0):
        self.chain = FilterChain(
            causal_bandpass(20.0, 450.0, fs_native),
            causal_notch(50.0, fs_native),
        )
        self.win = _RingWindow(int(envelope_ms / 1000 * fs_native))
        self.decim = fs_native // fs_target
        self._counter = 0

    def step(self, samples: np.ndarray) -> np.ndarray:
        filtered = self.chain.step(samples)
        out = []
        for v in filtered:
            self.win.push(np.array([v]))
            self._counter += 1
            if self._counter % self.decim == 0 and self.win.ready():
                buf = self.win.view()
                out.append(float(np.sqrt(np.mean(buf ** 2))))
        return np.asarray(out, dtype=float)


# --- ACC --------------------------------------------------------------------


class ACCStream:
    """Causal ACC magnitude → 0.5–20 Hz BP → 2 s sliding feature window."""

    def __init__(self, fs: int = 100, window_s: float = 2.0):
        self.bp = causal_bandpass(0.5, 20.0, fs)
        self.win = _RingWindow(int(window_s * fs))
        self.fs = fs

    def step_axes(self, ax, ay, az) -> np.ndarray:
        mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        return self.bp.step(mag)

    def push(self, ax, ay, az) -> None:
        filtered = self.step_axes(ax, ay, az)
        self.win.push(filtered)

    def features(self) -> dict[str, float] | None:
        if not self.win.ready():
            return None
        return acc_feats.window_features(self.win.view(), fs=self.fs)


# --- PPG --------------------------------------------------------------------


class PPGStream:
    def __init__(self, fs: int = 100, window_s: float = 10.0):
        self.bp = causal_bandpass(0.5, 8.0, fs)
        self.win = _RingWindow(int(window_s * fs))
        self.fs = fs

    def push(self, samples: np.ndarray) -> None:
        self.win.push(self.bp.step(samples))

    def features(self) -> dict[str, float] | None:
        if not self.win.ready():
            return None
        return ppg_feats.window_features(self.win.view(), fs=self.fs)


# --- TEMP -------------------------------------------------------------------


class TempStream:
    def __init__(self, fs: int = 100, window_s: float = 60.0):
        self.lp = causal_lowpass(0.1, fs, order=2)
        self.win = _RingWindow(int(window_s * fs))

    def push(self, samples: np.ndarray) -> None:
        self.win.push(self.lp.step(samples))

    def features(self) -> dict[str, float] | None:
        if not self.win.ready():
            return None
        return temp_feats.window_features(self.win.view())


# --- combined feature window -----------------------------------------------


def emg_envelope_window_features(envelope_buf: np.ndarray) -> dict[str, float]:
    """Run the offline EMG feature extractor on a streaming envelope window.

    Spectral features (MNF, MDF, Dimitrov) are NOT meaningful on the 100 Hz
    envelope (Nyquist = 50 Hz, full physiological band 20–450 Hz collapsed).
    Returns NaN for those keys; amplitude-only features are computed.
    """
    keys = ("emg_rms", "emg_iemg", "emg_mnf", "emg_mdf", "emg_dimitrov",
            "emg_lscore", "emg_mfl", "emg_msr", "emg_wamp")
    if envelope_buf.size < 16:
        return {k: float("nan") for k in keys}
    feats = emg_feats.window_features(envelope_buf, fs=100,
                                       wamp_threshold=emg_feats.WAMP_THRESHOLD_V)
    feats["emg_mnf"] = float("nan")
    feats["emg_mdf"] = float("nan")
    feats["emg_dimitrov"] = float("nan")
    return feats
