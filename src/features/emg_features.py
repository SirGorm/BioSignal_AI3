"""
EMG feature extraction — offline (full-session access, filtfilt allowed).

Features per window (500 ms, hop 100 ms):
  emg_rms      : root-mean-square amplitude (De Luca 1997)
  emg_iemg     : integrated EMG = sum of |x| (De Luca 1997)
  emg_mnf      : mean power frequency / spectral centroid (De Luca 1997)
  emg_mdf      : median power frequency (De Luca 1997)
  emg_dimitrov : FInsm5 = M(-1)/M(5), most fatigue-sensitive index in
                 dynamic contractions (Dimitrov et al. 2006)

Per-set slope features (offline only, needs full set):
  emg_mnf_slope      : linear slope of MNF over set (more negative = more fatigue)
  emg_mdf_slope      : linear slope of MDF over set
  emg_dimitrov_slope : linear slope of FInsm5 over set (positive = fatigue)

Baseline-normalised versions: <feature>_rel = value / baseline_median
  (EmgBaselineNormalizer handles this — first 60 s of session, rest-only)

References
----------
- De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
  Journal of Applied Biomechanics, 13(2), 135-163.
- Cifrek, M., Medved, V., Tonkovic, S., & Ostojic, S. (2009). Surface EMG based
  muscle fatigue evaluation in biomechanics. Clinical Biomechanics, 24(4), 327-340.
- Dimitrov, G. V., Arabadzhiev, T. I., Mileva, K. N., Bowtell, J. L., Crichton, N.,
  & Dimitrova, N. A. (2006). Muscle fatigue during dynamic contractions assessed by
  new spectral indices. Medicine and Science in Sports and Exercise, 38(11), 1971-1979.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and
  selection for EMG signal classification. Expert Systems with Applications,
  39(8), 7420-7431.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power
  spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental
  algorithms for scientific computing in Python. Nature Methods, 17, 261-272.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.signal import welch


FS_EMG = 2000  # Hz — verified against dataset/recording_012/metadata.json
WINDOW_MS = 500  # ms — minimum for stable MNF/MDF (Cifrek et al. 2009)
HOP_MS = 100


def _nanfill_signal(signal: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill NaN values in a signal.

    Required before IIR filtering because NaN propagates through sosfiltfilt
    and sosfilt. This is a standard signal pre-processing step for biosignals
    with occasional dropouts (De Luca 1997).
    """
    out = signal.copy().astype(float)
    mask = np.isnan(out)
    if not mask.any():
        return out
    idx = np.arange(len(out))
    # Forward fill
    good = np.where(~mask)[0]
    if len(good) == 0:
        return out
    fp = np.maximum.accumulate(np.where(~mask, idx, 0))
    out[mask] = out[fp[mask]]
    # Backward fill any remaining leading NaN
    mask2 = np.isnan(out)
    if mask2.any():
        bp = np.minimum.accumulate(np.where(~mask2, idx, len(out) - 1)[::-1])[::-1]
        out[mask2] = out[bp[mask2]]
    return out


def _filter_emg_offline(signal: np.ndarray, fs: int = FS_EMG) -> np.ndarray:
    """Zero-phase Butterworth 20-450 Hz bandpass + 50 Hz notch (offline).

    Using filtfilt for zero-phase offline processing (Virtanen et al. 2020).
    Notch at 50 Hz removes power-line interference (Merletti & Parker 2004).
    NaN values in the signal are interpolated before filtering (De Luca 1997).
    """
    signal = _nanfill_signal(signal)
    sos = butter(4, [20.0, 450.0], btype="band", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    b_notch, a_notch = iirnotch(50.0, Q=30, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def _filter_emg_causal(signal: np.ndarray, fs: int = FS_EMG) -> np.ndarray:
    """Causal Butterworth 20-450 Hz bandpass + 50 Hz notch — for parity testing.

    Uses sosfilt (same as streaming) so offline and streaming agree exactly
    on filter output. Used only in parity tests to validate state management,
    not in production offline feature extraction (Oppenheim & Schafer 2010).
    """
    from scipy.signal import sosfilt, sosfilt_zi, tf2sos
    signal = _nanfill_signal(signal)
    sos_bp = butter(4, [20.0, 450.0], btype="band", fs=fs, output="sos")
    zi_bp = sosfilt_zi(sos_bp) * signal[0]
    filtered, _ = sosfilt(sos_bp, signal, zi=zi_bp)
    b_notch, a_notch = iirnotch(50.0, Q=30, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    zi_notch = sosfilt_zi(sos_notch) * filtered[0]
    filtered, _ = sosfilt(sos_notch, filtered, zi=zi_notch)
    return filtered


def emg_window_features(emg_window: np.ndarray, fs: int = FS_EMG) -> dict:
    """Compute EMG features for one window (already filtered).

    Uses Welch's PSD method (Welch 1967) with nperseg = min(256, len(window)).
    Spectral features restricted to physiological range 20-450 Hz.

    MNF = spectral centroid = sum(f * P(f)) / sum(P(f))  (De Luca 1997)
    MDF = frequency at which cumulative PSD reaches 50%   (De Luca 1997)
    FInsm5 = M(-1) / M(5)  where M_k = sum(f^k * P(f))   (Dimitrov et al. 2006)

    Parameters
    ----------
    emg_window : 1D array of filtered EMG samples within the window.
    fs         : Sample rate (default 2000 Hz).

    Returns
    -------
    dict with keys: emg_rms, emg_iemg, emg_mnf, emg_mdf, emg_dimitrov
    """
    nan_result = {k: np.nan for k in
                  ["emg_rms", "emg_iemg", "emg_mnf", "emg_mdf", "emg_dimitrov"]}

    if len(emg_window) < 16:
        return nan_result

    # Time-domain features (De Luca 1997)
    rms = float(np.sqrt(np.mean(emg_window ** 2)))
    iemg = float(np.sum(np.abs(emg_window)))

    # Welch PSD (Welch 1967)
    nperseg = min(256, len(emg_window))
    f, pxx = welch(emg_window, fs=fs, nperseg=nperseg)

    # Restrict to physiological EMG range 20-450 Hz
    mask = (f >= 20.0) & (f <= 450.0)
    f_phys = f[mask]
    pxx_phys = pxx[mask]

    total_power = pxx_phys.sum()
    if total_power < 1e-20:
        return nan_result

    # Mean Power Frequency — spectral centroid (De Luca 1997)
    mnf = float(np.sum(f_phys * pxx_phys) / total_power)

    # Median Power Frequency (De Luca 1997)
    cum_psd = np.cumsum(pxx_phys)
    half = cum_psd[-1] / 2.0
    mdf_idx = np.searchsorted(cum_psd, half)
    mdf_idx = np.clip(mdf_idx, 0, len(f_phys) - 1)
    mdf = float(f_phys[mdf_idx])

    # Dimitrov FInsm5 = M(-1) / M(5) (Dimitrov et al. 2006)
    # Avoid division by zero at f=0 by using np.maximum
    M_neg1 = float(np.sum(pxx_phys / np.maximum(f_phys, 1.0)))
    M_5 = float(np.sum((f_phys ** 5) * pxx_phys))
    dimitrov = M_neg1 / (M_5 + 1e-30)

    return {
        "emg_rms": rms,
        "emg_iemg": iemg,
        "emg_mnf": mnf,
        "emg_mdf": mdf,
        "emg_dimitrov": dimitrov,
    }


class EmgBaselineNormalizer:
    """Per-subject EMG baseline normalization.

    Captures median feature values during the first 60 s of rest at session
    start. All subsequent features are divided by the baseline median to give
    subject-invariant relative values (Cifrek et al. 2009).

    Baseline uses median (not mean) to be robust to electrode-adjustment
    transients at session start (Phinyomark et al. 2012).
    """

    _KEYS = ("emg_mnf", "emg_mdf", "emg_rms", "emg_dimitrov", "emg_iemg")

    def __init__(self) -> None:
        self._accum: dict[str, list[float]] = {k: [] for k in self._KEYS}
        self.locked: bool = False
        self.baseline_medians: dict[str, float] = {}

    def observe(self, feats: dict) -> None:
        """Feed one window's features during baseline capture. Ignores NaN."""
        if self.locked:
            return
        for k in self._KEYS:
            v = feats.get(k, np.nan)
            if not np.isnan(v):
                self._accum[k].append(float(v))

    def lock(self) -> None:
        """Compute and store baseline medians. Call after baseline window ends."""
        self.baseline_medians = {
            k: float(np.median(vals)) if vals else np.nan
            for k, vals in self._accum.items()
        }
        self.locked = True

    def normalize(self, feats: dict) -> dict:
        """Return feats extended with <key>_rel = value / baseline_median."""
        if not self.locked:
            raise RuntimeError("Call lock() before normalize().")
        out = dict(feats)
        for k in self._KEYS:
            bmed = self.baseline_medians.get(k, np.nan)
            v = feats.get(k, np.nan)
            if not np.isnan(bmed) and bmed > 0:
                out[f"{k}_rel"] = v / bmed
            else:
                out[f"{k}_rel"] = np.nan
        return out


def extract_emg_features(
    emg_raw: np.ndarray,
    t_unix: np.ndarray,
    fs: int = FS_EMG,
    window_ms: float = WINDOW_MS,
    hop_ms: float = HOP_MS,
    normalizer: EmgBaselineNormalizer | None = None,
    baseline_end_unix: float | None = None,
) -> pd.DataFrame:
    """Extract EMG features over sliding windows (offline — filtfilt allowed).

    Parameters
    ----------
    emg_raw          : Raw EMG signal at native fs (2000 Hz).
    t_unix           : Unix timestamps at native rate.
    fs               : Sample rate.
    window_ms        : Window length in ms.
    hop_ms           : Hop size in ms.
    normalizer       : Optional EmgBaselineNormalizer; if None one is created.
    baseline_end_unix: Unix time marking end of baseline period.

    Returns
    -------
    DataFrame with one row per window, columns: t_unix, emg_*, emg_*_rel.
    """
    filtered = _filter_emg_offline(emg_raw, fs)

    win_samp = int(window_ms * fs / 1000)
    hop_samp = max(1, int(hop_ms * fs / 1000))
    n = len(filtered)

    if normalizer is None:
        normalizer = EmgBaselineNormalizer()

    rows = []
    pos = 0
    while pos + win_samp <= n:
        t_center = t_unix[pos + win_samp // 2]
        feats = emg_window_features(filtered[pos : pos + win_samp], fs)
        feats["t_unix"] = t_center

        # Feed baseline normalizer during baseline period
        if baseline_end_unix is not None and t_center <= baseline_end_unix:
            normalizer.observe(feats)
        elif baseline_end_unix is not None and not normalizer.locked:
            normalizer.lock()

        rows.append(feats)
        pos += hop_samp

    if not normalizer.locked:
        normalizer.lock()

    # Apply normalization to all rows
    out_rows = []
    for row in rows:
        out_rows.append(normalizer.normalize(row))

    return pd.DataFrame(out_rows)


def within_set_slope(values: np.ndarray, times: np.ndarray) -> float:
    """Linear slope of a feature within a set.

    More negative MNF slope = faster fatigue accumulation (Dimitrov et al. 2006,
    Cifrek et al. 2009).

    Parameters
    ----------
    values : Feature values over the set (e.g. emg_mnf series).
    times  : Corresponding timestamps in seconds.

    Returns
    -------
    Slope coefficient (units per second). Returns 0.0 if fewer than 3 points.
    """
    valid = ~(np.isnan(values) | np.isnan(times))
    if valid.sum() < 3:
        return 0.0
    return float(np.polyfit(times[valid] - times[valid][0], values[valid], 1)[0])
