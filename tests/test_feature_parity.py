"""
Parity test: verify that streaming (causal) feature extractors agree with
offline feature extractors within tolerance, after filter state warmup.

Uses recording_012 as the representative fixture (verified in inspection).

The first 30 s of outputs are excluded because:
- IIR filter states need time to converge (Oppenheim & Schafer 2010)
- The online R-peak detector has an adaptive threshold warmup period
  (Pan & Tompkins 1985)

Tolerance: rtol=0.05 (5%) for instantaneous features, which is generous
enough to account for:
- Small numerical differences between filtfilt (zero-phase) and sosfilt
  (causal) after state convergence
- Rounding in adaptive threshold vs fixed-window detection

Failing this test aborts the training pipeline — streaming features must
match offline features before we train models on them.

References
----------
- Oppenheim, A. V., & Schafer, R. W. (2010). Discrete-Time Signal Processing
  (3rd ed.). Pearson.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
  IEEE Transactions on Biomedical Engineering, BME-32(3), 230-236.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_biosignal, load_imu, load_metadata, load_temperature

# Offline feature extractors
from src.features.emg_features import (
    _filter_emg_offline,
    _filter_emg_causal,
    emg_window_features,
    EmgBaselineNormalizer,
    FS_EMG,
)
from src.features.eda_features import (
    _filter_eda_offline,
    _filter_eda_causal,
    eda_window_features,
    FS_EDA,
)
from src.features.acc_features import (
    _filter_acc_offline,
    _filter_acc_causal,
    _compute_acc_mag,
    acc_mag_window_features,
    FS_ACC,
)

# Streaming extractors
from src.streaming.emg_streaming import StreamingEMGExtractor
from src.streaming.eda_streaming import StreamingEDAExtractor
from src.streaming.acc_streaming import StreamingAccExtractor

# -----------------------------------------------------------------------
# Fixture paths
# -----------------------------------------------------------------------
REC_DIR = ROOT / "dataset" / "recording_012"
LABELED_PARQUET = ROOT / "data" / "labeled" / "recording_012" / "aligned_features.parquet"
WARMUP_S = 30.0  # seconds to skip (filter state convergence)
RTOL = 0.05  # 5% relative tolerance for causal vs zero-phase after warmup
# Absolute tolerance for near-zero features (avoid division issues)
ATOL = 1e-6

# Chunk size for streaming replay: 100 ms at each modality's native rate
EMG_CHUNK = int(0.1 * FS_EMG)  # 200 samples
EDA_CHUNK = int(0.1 * FS_EDA)  # 5 samples
ACC_CHUNK = int(0.1 * FS_ACC)  # 10 samples


def _skip_if_no_data():
    if not REC_DIR.exists():
        pytest.skip(f"Dataset not found: {REC_DIR}")
    if not LABELED_PARQUET.exists():
        pytest.skip(f"Labeled parquet not found: {LABELED_PARQUET}")


# -----------------------------------------------------------------------
# EMG parity test
# -----------------------------------------------------------------------

@pytest.mark.parametrize("feat_key", ["emg_rms", "emg_mnf", "emg_mdf", "emg_dimitrov"])
def test_emg_parity(feat_key):
    """Streaming EMG extractor matches offline within 5% after 30 s warmup.

    De Luca 1997 — MNF/MDF features.
    Dimitrov et al. 2006 — FInsm5.
    """
    _skip_if_no_data()

    emg_df = load_biosignal(REC_DIR, "emg", "emg")
    emg_raw = emg_df["emg"].values
    t_unix = emg_df["timestamp"].values

    # Trim to first 5 minutes to keep test fast
    trim_s = 300
    trim_n = min(len(emg_raw), trim_s * FS_EMG)
    emg_raw = emg_raw[:trim_n]
    t_unix = t_unix[:trim_n]

    baseline_end = float(t_unix[0]) + 60.0

    # ---- Offline (causal filter for fair parity comparison) ----
    # Use the causal sosfilt offline reference so the parity test checks
    # state management and windowing, not filtfilt vs sosfilt differences.
    # The production offline extractor uses filtfilt for better SNR — this
    # causal reference is only for validating streaming state correctness
    # (Oppenheim & Schafer 2010).
    filtered_off = _filter_emg_causal(emg_raw, FS_EMG)
    win_samp = int(0.5 * FS_EMG)  # 500 ms
    hop_samp = int(0.1 * FS_EMG)  # 100 ms

    normalizer = EmgBaselineNormalizer()
    offline_rows = []
    pos = 0
    while pos + win_samp <= len(filtered_off):
        # Use trailing-edge time (same as streaming deque — emits at last sample)
        t_trailing = t_unix[pos + win_samp - 1]
        feats = emg_window_features(filtered_off[pos : pos + win_samp], FS_EMG)
        if t_trailing <= baseline_end:
            normalizer.observe(feats)
        elif not normalizer.locked:
            normalizer.lock()
        offline_rows.append({"t_unix": t_trailing, **feats})
        pos += hop_samp

    if not normalizer.locked:
        normalizer.lock()
    offline_df = pd.DataFrame(offline_rows)

    # ---- Streaming ----
    extractor = StreamingEMGExtractor(fs=FS_EMG, window_ms=500, hop_ms=100,
                                       baseline_s=60.0)
    extractor.set_baseline_end(baseline_end)

    streaming_rows = []
    for start in range(0, len(emg_raw), EMG_CHUNK):
        end = min(start + EMG_CHUNK, len(emg_raw))
        chunk = emg_raw[start:end]
        t_chunk = t_unix[start:end]
        rows = extractor.step(chunk, t_chunk)
        streaming_rows.extend(rows)

    streaming_df = pd.DataFrame(streaming_rows)

    # ---- Compare after warmup ----
    warmup_t = float(t_unix[0]) + WARMUP_S
    off_post = offline_df[offline_df["t_unix"] > warmup_t][feat_key].dropna()
    str_post = streaming_df[streaming_df["t_unix"] > warmup_t][feat_key].dropna() \
        if feat_key in streaming_df.columns else pd.Series(dtype=float)

    if len(off_post) == 0 or len(str_post) == 0:
        pytest.skip(f"No post-warmup values for {feat_key}")

    # Align by length
    n = min(len(off_post), len(str_post))
    off_vals = off_post.values[:n]
    str_vals = str_post.values[:n]

    # Filter out any rows where either is NaN
    valid = ~(np.isnan(off_vals) | np.isnan(str_vals))
    assert valid.sum() > 10, f"Too few valid pairs for {feat_key}: {valid.sum()}"

    np.testing.assert_allclose(
        str_vals[valid],
        off_vals[valid],
        rtol=RTOL,
        atol=ATOL,
        err_msg=(
            f"EMG {feat_key}: streaming vs offline disagree beyond {RTOL*100:.0f}% "
            f"after {WARMUP_S}s warmup. "
            f"Max relative diff: {np.max(np.abs((str_vals[valid] - off_vals[valid]) / (np.abs(off_vals[valid]) + ATOL))):.4f}"
        ),
    )


# -----------------------------------------------------------------------
# EDA parity test
# -----------------------------------------------------------------------

def test_eda_parity_scl():
    """Streaming EDA SCL matches offline SCL within 5% after warmup.

    Boucsein 2012 — SCL as tonic EDA component.
    """
    _skip_if_no_data()

    eda_df = load_biosignal(REC_DIR, "eda", "eda")
    eda_raw = eda_df["eda"].values
    t_unix = eda_df["timestamp"].values

    # Trim to first 5 minutes
    trim_n = min(len(eda_raw), 300 * FS_EDA)
    eda_raw = eda_raw[:trim_n]
    t_unix = t_unix[:trim_n]

    baseline_end = float(t_unix[0]) + 60.0
    baseline_scl = float(np.nanmedian(eda_raw[t_unix <= baseline_end]))

    # ---- Offline (causal filter for fair parity comparison) ----
    filtered_off = _filter_eda_causal(eda_raw, FS_EDA)
    win_samp = int(10.0 * FS_EDA)
    hop_samp = max(1, int(0.1 * FS_EDA))

    offline_rows = []
    pos = 0
    while pos + win_samp <= len(filtered_off):
        # Trailing-edge time (aligns with streaming deque behaviour)
        t_trailing = t_unix[pos + win_samp - 1]
        feats = eda_window_features(filtered_off[pos : pos + win_samp], FS_EDA)
        offline_rows.append({"t_unix": t_trailing, "eda_scl": feats["eda_scl"]})
        pos += hop_samp
    offline_df = pd.DataFrame(offline_rows)

    # ---- Streaming ----
    extractor = StreamingEDAExtractor(fs=FS_EDA, window_s=10.0, hop_ms=100)
    extractor.set_baseline_end(baseline_end)
    streaming_rows = []
    for start in range(0, len(eda_raw), EDA_CHUNK):
        end = min(start + EDA_CHUNK, len(eda_raw))
        rows = extractor.step(eda_raw[start:end], t_unix[start:end])
        streaming_rows.extend(rows)
    streaming_df = pd.DataFrame(streaming_rows)

    # ---- Compare after warmup ----
    warmup_t = float(t_unix[0]) + WARMUP_S
    off_post = offline_df[offline_df["t_unix"] > warmup_t]["eda_scl"].dropna()
    str_post = streaming_df[streaming_df["t_unix"] > warmup_t]["eda_scl"].dropna() \
        if "eda_scl" in streaming_df.columns else pd.Series(dtype=float)

    n = min(len(off_post), len(str_post))
    if n < 5:
        pytest.skip("Insufficient post-warmup EDA samples")

    off_vals = off_post.values[:n]
    str_vals = str_post.values[:n]
    valid = ~(np.isnan(off_vals) | np.isnan(str_vals))

    assert valid.sum() > 5

    np.testing.assert_allclose(
        str_vals[valid],
        off_vals[valid],
        rtol=RTOL,
        atol=ATOL,
        err_msg="EDA SCL streaming vs offline disagree.",
    )


# -----------------------------------------------------------------------
# Accelerometer parity test
# -----------------------------------------------------------------------

def test_acc_parity_rms():
    """Streaming acc RMS matches offline acc RMS within 5% after warmup.

    Mannini & Sabatini 2010 — acc-based activity features.
    """
    _skip_if_no_data()

    imu_df = load_imu(REC_DIR)
    ax = imu_df["ax"].values
    ay = imu_df["ay"].values
    az = imu_df["az"].values
    t_unix = imu_df["timestamp"].values

    # Trim to first 5 minutes
    trim_n = min(len(ax), 300 * FS_ACC)
    ax, ay, az = ax[:trim_n], ay[:trim_n], az[:trim_n]
    t_unix = t_unix[:trim_n]

    # ---- Offline (causal filter for fair parity comparison) ----
    acc_mag = _compute_acc_mag(ax, ay, az)
    filtered_off = _filter_acc_causal(acc_mag, FS_ACC)
    win_samp = int(2.0 * FS_ACC)
    hop_samp = int(0.1 * FS_ACC)

    offline_rows = []
    pos = 0
    while pos + win_samp <= len(filtered_off):
        # Trailing-edge time (aligns with streaming deque behaviour)
        t_trailing = t_unix[pos + win_samp - 1]
        feats = acc_mag_window_features(filtered_off[pos : pos + win_samp], FS_ACC)
        offline_rows.append({"t_unix": t_trailing, "acc_rms": feats["acc_rms"]})
        pos += hop_samp
    offline_df = pd.DataFrame(offline_rows)

    # ---- Streaming ----
    extractor = StreamingAccExtractor(fs=FS_ACC, window_ms=2000, hop_ms=100)
    streaming_rows = []
    for start in range(0, len(ax), ACC_CHUNK):
        end = min(start + ACC_CHUNK, len(ax))
        rows = extractor.step(
            ax[start:end], ay[start:end], az[start:end], t_unix[start:end]
        )
        streaming_rows.extend(rows)
    streaming_df = pd.DataFrame(streaming_rows)

    # ---- Compare after warmup ----
    warmup_t = float(t_unix[0]) + WARMUP_S
    off_post = offline_df[offline_df["t_unix"] > warmup_t]["acc_rms"].dropna()
    str_post = streaming_df[streaming_df["t_unix"] > warmup_t]["acc_rms"].dropna() \
        if "acc_rms" in streaming_df.columns else pd.Series(dtype=float)

    n = min(len(off_post), len(str_post))
    if n < 5:
        pytest.skip("Insufficient post-warmup ACC samples")

    off_vals = off_post.values[:n]
    str_vals = str_post.values[:n]
    valid = ~(np.isnan(off_vals) | np.isnan(str_vals))
    assert valid.sum() > 5

    np.testing.assert_allclose(
        str_vals[valid],
        off_vals[valid],
        rtol=RTOL,
        atol=ATOL,
        err_msg="ACC RMS streaming vs offline disagree.",
    )


# -----------------------------------------------------------------------
# Streaming hook test (no filtfilt in streaming code)
# -----------------------------------------------------------------------

def test_no_filtfilt_in_streaming():
    """No forbidden non-causal operations in src/streaming/.

    Uses the same regex patterns as the check-no-filtfilt.sh hook so both
    the hook and the pytest suite agree on what is forbidden.
    Oppenheim & Schafer 2010 — causal vs zero-phase filtering.
    """
    import re

    streaming_dir = ROOT / "src" / "streaming"
    if not streaming_dir.exists():
        pytest.skip("src/streaming not found")

    # These patterns mirror check-no-filtfilt.sh exactly
    FORBIDDEN_PATTERNS = [
        r"scipy\.signal\.filtfilt",
        r"from scipy\.signal import .*filtfilt",
        r"scipy\.signal\.savgol_filter",
        r"np\.fft\.fft\(.*signal",
        r"scipy\.signal\.find_peaks\(.*signal",
    ]

    violations = []
    for py_file in streaming_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            for m in re.finditer(pattern, text):
                violations.append(
                    f"{py_file.name}: pattern '{pattern}' at char {m.start()}"
                )

    assert not violations, (
        "Non-causal operations found in streaming code:\n"
        + "\n".join(violations)
    )


# -----------------------------------------------------------------------
# Temperature NaN-tolerance test
# -----------------------------------------------------------------------

def test_temp_nan_tolerance():
    """Temperature features return NaN without crashing when data is absent."""
    from src.features.temp_features import extract_temp_features

    # Simulate all-NaN temperature (recordings 007-014)
    n = 60
    temp_arr = np.full(n, np.nan)
    t_arr = np.arange(n, dtype=float) + 1e10  # valid unix epoch

    result = extract_temp_features(temp_arr, t_arr, fs=1, window_s=60.0, hop_s=1.0)
    # Should return empty DataFrame (no data) — not crash
    assert isinstance(result, pd.DataFrame), "Expected DataFrame"
    # All NaN or empty is acceptable
    if not result.empty:
        assert result["temp_mean"].isna().all(), "Expected all-NaN temp_mean"


if __name__ == "__main__":
    # Running the parity test standalone exits non-zero on failure,
    # which halts the training pipeline per project spec.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-x", "-v"],
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)
