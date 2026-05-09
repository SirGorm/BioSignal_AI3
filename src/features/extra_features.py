"""
Time-domain EMG/IMU features: WAMP, MFL, MSR, LS, LS4.

Pure-NumPy, single-window functions. Generic over signal type — can be
applied to filtered EMG (2000 Hz) or filtered acc-magnitude (100 Hz).

Definitions
-----------
WAMP (Willison Amplitude):
    sum_i 1{|x[i+1] - x[i]| > thr}
    Counts rapid amplitude changes; threshold gates noise.
    (Willison 1964; Phinyomark et al. 2012)

MFL (Maximum Fractal Length):
    log10(sqrt(sum_i (x[i+1] - x[i])^2))
    Self-similarity / signal complexity, sensitive at low activation.
    (Phinyomark et al. 2012)

MSR (Mean Square Root):
    (1/N) * sum_i sqrt(|x[i]|)
    Compresses amplitude dynamic range.
    (Phinyomark et al. 2018)

LS (L-Score):
    (1/N) * sum_i log(|x[i]| + 1)
    Logarithmic amplitude statistic, robust to outliers.
    (Phinyomark et al. 2018)

LS4 (Low Sampling 4):
    LS computed on x[::4] (decimated by factor 4).
    Robust against high-frequency noise via coarser sampling.
    (Phinyomark et al. 2018)

References
----------
- Willison, R. G. (1964). Analysis of electrical activity in healthy and
  dystrophic muscle in man. J. Neurol. Neurosurg. Psychiatry, 27, 386-394.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature
  reduction and selection for EMG signal classification. Expert Systems
  with Applications, 39(8), 7420-7431.
- Phinyomark, A., N Khushaba, R., & Scheme, E. (2018). Feature extraction
  and selection for myoelectric control based on wearable EMG sensors.
  Sensors, 18(5), 1615.
"""

from __future__ import annotations

import numpy as np


def wamp(x: np.ndarray, threshold: float) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(x)) > threshold))


def mfl(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    s = float(np.sum(np.diff(x) ** 2))
    if s <= 0.0:
        return float("nan")
    return float(np.log10(np.sqrt(s)))


def msr(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.sqrt(np.abs(x))))


def ls(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.log(np.abs(x) + 1.0)))


def ls4(x: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    return ls(x[::4])


def extras_window(x: np.ndarray, threshold: float, prefix: str) -> dict:
    """Compute the 5 features for one window with given prefix and WAMP threshold."""
    return {
        f"{prefix}_wamp": wamp(x, threshold),
        f"{prefix}_mfl":  mfl(x),
        f"{prefix}_msr":  msr(x),
        f"{prefix}_ls":   ls(x),
        f"{prefix}_ls4":  ls4(x),
    }


def baseline_threshold(baseline_signal: np.ndarray, k: float = 0.1) -> float:
    """WAMP threshold = k * std of baseline (first 60 s rest) — per Phinyomark 2012.

    Falls back to 1e-6 if baseline is degenerate (all-NaN / zero std) so
    WAMP still produces a finite count rather than NaN.
    """
    s = float(np.nanstd(baseline_signal))
    if not np.isfinite(s) or s <= 0.0:
        return 1e-6
    return k * s
