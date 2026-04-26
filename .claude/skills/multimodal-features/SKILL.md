---
name: multimodal-features
description: Use when extracting features from any of the 6 modalities in this project (ECG, EMG, EDA, temp, acc-magnitude, PPG-green). Provides per-modality feature catalogs, recommended windows, and code templates. Causal versions for streaming, offline versions for training.
---

# Multimodal Feature Extraction

Per-modality reference. For EMG fatigue specifics (MNF/MDF/Dimitrov), use the dedicated `emg-fatigue-features` skill — this skill covers the other 5 modalities.

## ECG → cardiac features

Bandpass 0.5–40 Hz (causal, IIR, persisted state). Detect R-peaks online with Pan-Tompkins or NeuroKit2's online variant.

```python
import numpy as np

def ecg_window_features(rr_intervals_ms_in_window, ecg_window):
    """rr_intervals_ms: detected RR intervals (ms) within the window.
    Use a 30-60s window for stable HRV."""
    if len(rr_intervals_ms_in_window) < 5:
        return {'hr': np.nan, 'rmssd': np.nan, 'sdnn': np.nan, 'pnn50': np.nan}
    rr = np.asarray(rr_intervals_ms_in_window, dtype=float)
    hr = 60_000 / np.mean(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    sdnn = np.std(rr, ddof=1)
    pnn50 = np.mean(np.abs(np.diff(rr)) > 50) * 100
    return {'hr': hr, 'rmssd': rmssd, 'sdnn': sdnn, 'pnn50': pnn50}
```

For real-time streaming: maintain a rolling buffer of RR intervals from the past 30 s, recompute on each new beat.

## EDA → arousal features

Sample rate 4–32 Hz. Bandpass 0.05–5 Hz LP. Decompose into tonic (SCL) and phasic (SCR) — use NeuroKit2's `eda_phasic` offline; for streaming, a simple high-pass at 0.05 Hz separates phasic from tonic.

```python
def eda_window_features(eda_window, fs, scr_thresh=0.05):
    """SCL (tonic level) and SCR (phasic peaks)."""
    scl = np.median(eda_window)  # robust tonic estimate
    # Crude phasic: high-pass via subtracting moving median
    if len(eda_window) > int(2 * fs):
        from scipy.ndimage import median_filter
        baseline = median_filter(eda_window, size=int(2 * fs))
        phasic = eda_window - baseline
    else:
        phasic = eda_window - scl

    scr_amp = np.max(phasic) - np.min(phasic)
    n_scr = np.sum(np.diff((phasic > scr_thresh).astype(int)) == 1)
    return {'eda_scl': scl, 'eda_scr_amp': scr_amp, 'eda_scr_count': n_scr,
            'eda_phasic_mean': np.mean(np.abs(phasic))}
```

EDA is slow — use 10–30 s windows. SCR rate (per minute) is a good effort/arousal proxy.

## Temperature → fatigue/recovery features

Very slow signal (1–4 Hz typical). Skin temp typically rises during exercise then plateaus or drops.

```python
def temp_window_features(temp_window, t_window):
    """Slope and trend over the window (use 60+ s windows)."""
    if len(temp_window) < 5:
        return {'temp_mean': np.nan, 'temp_slope': np.nan}
    slope = np.polyfit(t_window - t_window[0], temp_window, 1)[0]  # deg/s
    return {'temp_mean': np.mean(temp_window),
            'temp_slope': slope,
            'temp_range': np.max(temp_window) - np.min(temp_window)}
```

For streaming, track slope using exponentially-weighted regression to keep state O(1).

## Acc-magnitude → motion features

Sample rate 50–100 Hz. Compute magnitude FIRST, then bandpass 0.5–20 Hz (causal, persisted state).

```python
def acc_mag_window_features(acc_mag_window, fs):
    """Motion intensity, dominant frequency, jerk."""
    rms = np.sqrt(np.mean(acc_mag_window ** 2))
    centered = acc_mag_window - np.mean(acc_mag_window)
    jerk = np.diff(centered) * fs  # m/s^3
    jerk_rms = np.sqrt(np.mean(jerk ** 2)) if len(jerk) else np.nan

    # Spectral
    nperseg = min(256, len(acc_mag_window))
    if nperseg >= 16:
        from scipy.signal import welch
        f, pxx = welch(acc_mag_window, fs=fs, nperseg=nperseg)
        dom_freq = f[np.argmax(pxx)]
        # Power in typical rep frequency band (0.3-1.5 Hz for strength training)
        rep_band_mask = (f >= 0.3) & (f <= 1.5)
        rep_band_power = np.sum(pxx[rep_band_mask])
        total_power = np.sum(pxx)
        rep_band_ratio = rep_band_power / (total_power + 1e-9)
    else:
        dom_freq = np.nan; rep_band_power = np.nan; rep_band_ratio = np.nan

    return {'acc_rms': rms, 'acc_jerk_rms': jerk_rms,
            'acc_dom_freq': dom_freq, 'acc_rep_band_power': rep_band_power,
            'acc_rep_band_ratio': rep_band_ratio}
```

Acc features are dominant for **exercise classification** and **rep counting**.

## PPG (green only) → cardiovascular features

Sample rate 64–128 Hz typical. Bandpass 0.5–8 Hz (causal). Detect peaks for HR; the other 3 wavelengths are loaded but unused per project requirements.

```python
def ppg_green_window_features(ppg_window, fs, peak_indices_in_window):
    """peak_indices: indices of detected systolic peaks within this window.
    Use an online peak detector (e.g., NeuroKit2 ppg_clean + ppg_findpeaks)."""
    if len(peak_indices_in_window) < 2:
        return {'ppg_hr': np.nan, 'ppg_pulse_amp': np.nan, 'ppg_pulse_amp_var': np.nan}

    pp_intervals_s = np.diff(peak_indices_in_window) / fs
    hr = 60.0 / np.mean(pp_intervals_s)

    # Pulse amplitude (peak-to-trough around each peak)
    amps = []
    for p in peak_indices_in_window:
        a = max(0, p - int(0.3 * fs))
        b = min(len(ppg_window), p + int(0.3 * fs))
        amps.append(np.max(ppg_window[a:b]) - np.min(ppg_window[a:b]))

    return {'ppg_hr': hr,
            'ppg_pulse_amp': np.mean(amps),
            'ppg_pulse_amp_var': np.std(amps)}
```

Use PPG HR to **cross-validate ECG HR**. They should agree within 5 bpm; persistent disagreement = motion artifact in one or the other.

## Common windowing scheme

| Modality | Recommended window | Hop | Reason |
|----------|-------------------|-----|--------|
| ECG | 30 s | 1 s | HRV stability |
| EMG | 250 ms | 100 ms | Fast spectral changes |
| EDA | 10 s | 1 s | Slow signal |
| Temp | 60 s | 5 s | Very slow |
| Acc | 1 s | 100 ms | Rep-rate captured |
| PPG | 10 s | 1 s | HR stability |

In practice, you build one feature row per "primary" hop (e.g., 100 ms) and the slower features carry the most recent value forward (forward-fill).

## Joining features across modalities

```python
def join_modality_features(features_per_modality, primary_hop_ms=100):
    """Each modality emits features at its own rate. Forward-fill onto a
    primary hop grid keyed by t_window_start."""
    # Use the fastest modality as the primary hop schedule
    base = features_per_modality['acc']  # has finest hop
    out = base.copy()
    for mod, df in features_per_modality.items():
        if mod == 'acc': continue
        out = pd.merge_asof(out.sort_values('t'), df.sort_values('t'),
                             on='t', direction='backward', tolerance=pd.Timedelta('30s'))
    return out
```

## Quality flags

For every window, also emit:
- `ecg_quality` (NeuroKit2's signal quality index, or your own SNR)
- `emg_saturation` (% samples at ±max range)
- `acc_clipping` (% samples at sensor max)
- `ppg_motion_artifact` (correlation between PPG and acc — high = artifact)

The ML expert filters or weights by these.

## References

For per-modality methodological decisions, cite from these (full entries in `literature-references` skill):

- **ECG/HRV**: Task Force 1996, Shaffer & Ginsberg 2017, Pan & Tompkins 1985
- **EDA**: Greco et al. 2016 (cvxEDA), Boucsein 2012, Posada-Quintero & Chon 2020
- **PPG (green wavelength)**: Allen 2007, Tamura et al. 2014, Maeda et al. 2011, Castaneda et al. 2018
- **Acc-based features**: Bonomi et al. 2009, Mannini & Sabatini 2010, Khan et al. 2010
- **Welch PSD**: Welch 1967
- **NeuroKit2 use**: Makowski et al. 2021

Use inline citations like `(Greco et al. 2016)` in code comments and deliverables. Never invent.
