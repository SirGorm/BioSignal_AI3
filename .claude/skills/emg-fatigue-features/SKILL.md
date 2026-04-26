---
name: emg-fatigue-features
description: Use when extracting EMG features that indicate localized muscle fatigue. Covers Mean Frequency (MNF), Median Frequency (MDF), Dimitrov spectral fatigue index, RMS amplitude, and proper per-subject baseline normalization. Causal/online variant for streaming use.
---

# EMG Fatigue Features

## Why these specific features

Local muscle fatigue manifests in EMG as a **left-shift of the power spectrum** (slower motor unit firing, larger and slower MUAPs from fatigue-resistant fibers). Time-domain RMS often *increases* with fatigue (compensatory recruitment) before it decreases at exhaustion. The most validated features are:

| Feature | What it captures | Direction during fatigue |
|---------|------------------|--------------------------|
| **MNF** (Mean Frequency) | Spectral centroid | ↓ |
| **MDF** (Median Frequency) | 50th percentile of PSD | ↓ |
| **Dimitrov FInsmk** | Spectral moment ratio | ↑ (most sensitive in literature) |
| **RMS** | Amplitude | ↑ then ↓ near failure |
| **Slope of MNF over time** | Rate of fatigue accumulation | More negative = faster fatigue |

The Dimitrov index `FInsm5 = M_(-1) / M_5` (spectral moment of order -1 over order 5) has been shown to be the most sensitive to fatigue in dynamic contractions.

## Causal feature extraction (per window)

```python
import numpy as np
from scipy.signal import welch

def emg_window_features(emg_window, fs):
    """
    emg_window: 1D array, already bandpass-filtered (20-450 Hz) and notch-filtered.
    Returns dict of features computed only on this window. CAUSAL.
    """
    # Use Welch with nperseg <= len(window) for reliable PSD
    nperseg = min(256, len(emg_window))
    f, pxx = welch(emg_window, fs=fs, nperseg=nperseg)

    # Restrict to physiological range
    mask = (f >= 10) & (f <= 500)
    f, pxx = f[mask], pxx[mask]

    if pxx.sum() < 1e-12:
        return {k: np.nan for k in ['rms', 'mnf', 'mdf', 'dimitrov', 'iemg']}

    # Time-domain
    rms = np.sqrt(np.mean(emg_window ** 2))
    iemg = np.sum(np.abs(emg_window))

    # Frequency-domain
    mnf = np.sum(f * pxx) / np.sum(pxx)
    cumpsd = np.cumsum(pxx)
    mdf = f[np.searchsorted(cumpsd, cumpsd[-1] / 2)]

    # Dimitrov FInsm5 (spectral moments)
    # M_k = integral of f^k * P(f) df
    M_neg1 = np.sum(pxx / np.maximum(f, 1.0))  # avoid div-by-zero at f=0
    M_5 = np.sum((f ** 5) * pxx)
    dimitrov = M_neg1 / (M_5 + 1e-12)

    return {'rms': rms, 'iemg': iemg, 'mnf': mnf, 'mdf': mdf, 'dimitrov': dimitrov}
```

## Per-subject baseline normalization (essential!)

Absolute MNF/MDF values are wildly different between subjects (electrode placement, body composition, muscle architecture). You must normalize:

```python
class EmgBaselineNormalizer:
    """Captures the first N seconds of resting/light activity per session per channel
    and normalizes subsequent features to that baseline."""
    def __init__(self, baseline_seconds=30, fs=1000):
        self.baseline_samples_target = baseline_seconds * fs
        self.baseline_features = {'mnf': [], 'mdf': [], 'rms': [], 'dimitrov': []}
        self.locked = False
        self.baseline_means = {}

    def observe(self, feats, n_samples_in_window):
        """Call during baseline capture only (first 30 s of session, low activity)."""
        if self.locked:
            return
        for k in self.baseline_features:
            if not np.isnan(feats[k]):
                self.baseline_features[k].append(feats[k])

    def lock(self):
        self.baseline_means = {k: np.median(v) if v else np.nan
                               for k, v in self.baseline_features.items()}
        self.locked = True

    def normalize(self, feats):
        if not self.locked:
            raise RuntimeError("Lock the baseline before normalizing")
        out = {}
        for k, v in feats.items():
            if k in self.baseline_means and not np.isnan(self.baseline_means[k]):
                out[f'{k}_rel'] = v / self.baseline_means[k]
            out[k] = v
        return out
```

In streaming: enter session, ask user to stand still for 30 s, capture baseline, lock, then start prediction.

## Within-set fatigue trajectory

For per-set fatigue regression, don't just use instantaneous features — use the *slope* of MNF/MDF over the set:

```python
def within_set_slope(feature_history, time_history):
    """Linear slope of feature over time within a set.
    More negative slope on MNF = more fatigue accumulation."""
    if len(feature_history) < 3:
        return 0.0
    return np.polyfit(time_history, feature_history, 1)[0]
```

Key features for the fatigue regressor (per set):
- `mnf_slope`: slope of MNF over the set
- `mdf_slope`: same for MDF
- `dimitrov_slope`: most sensitive in literature
- `mnf_endset_rel`: MNF in last rep / baseline
- `velocity_loss`: from VBT skill (often the strongest predictor)
- `hrv_rmssd_drop`: post-set vs pre-set RMSSD if you have it

## Common pitfalls

1. **Crosstalk from neighboring muscles.** If MNF doesn't drop during what should be a fatiguing set on the target muscle, suspect bad electrode placement or crosstalk.
2. **Movement artifacts.** Big spikes in RMS without spectral shift are usually motion artifacts. High-pass at 20 Hz aggressively. If problems persist, raise to 30 Hz.
3. **Using filtfilt for notch.** Don't. Use IIR notch with `sosfilt` and persisted state in streaming code.
4. **Window too short for spectral features.** Need at least 250 ms (250 samples at 1 kHz) for stable MNF/MDF. If your hop is shorter, that's fine, but the *window* must be long enough.
5. **Normalizing with mean instead of median.** Baseline often has a few outlier windows from the user adjusting; use median.

## Sanity checks before training

Before throwing EMG features into a fatigue regressor, plot:
1. MNF over time within a set known to be fatiguing — must trend down.
2. MNF over time within a set known to be light — should be flat.
3. Dimitrov index — should trend up during fatigue.

If these don't look right in your data, the fatigue regressor will not work no matter what model you throw at it. Fix data quality first.

## References

When documenting EMG fatigue features in code or deliverables, cite from these (full entries in `literature-references` skill):

- **Dimitrov et al. 2006** — for the FInsm5 = M(-1)/M(5) spectral fatigue index
- **De Luca 1997** — for foundational MNF/MDF interpretation as fatigue indicators
- **Cifrek et al. 2009** — for EMG fatigue evaluation in dynamic contractions
- **Phinyomark et al. 2012** — for feature selection rationale
- **Welch 1967** — when computing PSD via Welch's method
- **Makowski et al. 2021** — when using NeuroKit2

Never invent citations. If a reference is needed but absent from `literature-references`, write `[REF NEEDED: <topic>]`.
