"""
For the 3 problematic recordings (007, 008, 010), do a deeper match check:
  - Try the offset from probes that DID give corr=1.0 (007: 2360.1 s)
  - Run a sliding-window Pearson scan over the FULL signal at that offset
  - Confirm that the bulk of the recording is identical (per-window r > 0.9)
  - Identify any drift or local mismatch
"""
from __future__ import annotations

from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.plot_style import apply_style, despine

apply_style()

DS = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset")
MEM = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory")
OUT = Path(r"c:\Users\skogl\Downloads\eirikgsk\biosignal_2\BioSignal_AI3\inspections\_verify_memory_dataset")

FS = 100.0


def load_acc_mag(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    ax = pd.read_csv(folder / "ax.csv")
    ay = pd.read_csv(folder / "ay.csv")
    az = pd.read_csv(folder / "az.csv")
    n = min(len(ax), len(ay), len(az))
    t = ax["timestamp"].to_numpy()[:n]
    mag = np.sqrt(
        np.nan_to_num(ax["ax"].to_numpy()[:n]) ** 2
        + np.nan_to_num(ay["ay"].to_numpy()[:n]) ** 2
        + np.nan_to_num(az["az"].to_numpy()[:n]) ** 2
    )
    return t, mag


def best_lag_pearson_robust(template: np.ndarray, signal: np.ndarray, top_k: int = 20) -> tuple[int, float]:
    if len(signal) < len(template):
        return 0, 0.0
    t = template - np.nanmean(template)
    s = signal - np.nanmean(signal)
    t = np.nan_to_num(t)
    s = np.nan_to_num(s)
    corr = correlate(s, t, mode="valid", method="fft")
    K = min(top_k, len(corr))
    cand = np.argpartition(-corr, K - 1)[:K]
    best = (0, -np.inf)
    n = len(template)
    for lag in cand:
        win = signal[int(lag) : int(lag) + n]
        if len(win) < n:
            continue
        valid = ~(np.isnan(template) | np.isnan(win))
        if valid.sum() < n // 2:
            continue
        r = float(np.corrcoef(template[valid], win[valid])[0, 1])
        if r > best[1]:
            best = (int(lag), r)
    return best


def scan_pearson_at_offset(ds_t: np.ndarray, ds_x: np.ndarray, mem_t: np.ndarray, mem_x: np.ndarray,
                            offset: float, win_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window Pearson between dataset signal and memory signal aligned by offset."""
    win_n = int(win_s * FS)
    # Resample memory onto dataset grid via offset (round to nearest dataset sample)
    n = len(ds_x)
    n_windows = n // win_n
    rs = []
    centers = []
    for k in range(n_windows):
        a = k * win_n
        b = a + win_n
        ds_win = ds_x[a:b]
        # Find corresponding memory window
        ds_t_a = ds_t[a]
        ds_t_b = ds_t[b - 1]
        # Memory window in memory clock = ds_t + offset
        mem_target_a = ds_t_a + offset
        mem_target_b = ds_t_b + offset
        i0 = int(np.searchsorted(mem_t, mem_target_a))
        i1 = int(np.searchsorted(mem_t, mem_target_b)) + 1
        if i0 < 0 or i1 > len(mem_x) or (i1 - i0) < 0.5 * win_n:
            rs.append(np.nan)
            centers.append(ds_t[a + win_n // 2] - ds_t[0])
            continue
        m_win = mem_x[i0:i1]
        nmin = min(len(ds_win), len(m_win))
        ds_w = ds_win[:nmin]
        m_w = m_win[:nmin]
        valid = ~(np.isnan(ds_w) | np.isnan(m_w))
        if valid.sum() < nmin // 2:
            rs.append(np.nan)
        else:
            v = ds_w[valid]
            w = m_w[valid]
            if v.std() == 0 or w.std() == 0:
                rs.append(np.nan)
            else:
                rs.append(float(np.corrcoef(v, w)[0, 1]))
        centers.append(ds_t[a + win_n // 2] - ds_t[0])
    return np.array(centers), np.array(rs)


fig, axes = plt.subplots(3, 1, figsize=(14, 9))
results = {}

for ax, rec in zip(axes, ["007", "008", "010"]):
    ds_t, ds_mag = load_acc_mag(DS / f"recording_{rec}")
    mem_t, mem_mag = load_acc_mag(MEM / f"recording_{rec}_memory")

    # Find best offset using 10 probes, top-20 candidates each
    n_probes = 10
    win_n = int(60 * FS)
    cand_offsets = []
    cand_corrs = []
    for f in np.linspace(0.05, 0.95, n_probes):
        p = int(len(ds_mag) * f)
        a = max(0, p - win_n // 2)
        b = min(len(ds_mag), a + win_n)
        if b - a < win_n // 2:
            continue
        lag, r = best_lag_pearson_robust(ds_mag[a:b], mem_mag, top_k=30)
        offset = float(mem_t[lag] - ds_t[a])
        cand_offsets.append(offset)
        cand_corrs.append(r)

    # Most-frequent offset (rounded to seconds) wins
    offsets_round = np.round(cand_offsets, 1)
    vals, counts = np.unique(offsets_round, return_counts=True)
    mode_idx = int(np.argmax(counts))
    mode_offset = float(vals[mode_idx])
    mode_count = int(counts[mode_idx])

    print(f"=== rec_{rec} ===")
    print(f"  10-probe offsets: {[round(o,1) for o in cand_offsets]}")
    print(f"  10-probe corrs:   {[round(r,2) for r in cand_corrs]}")
    print(f"  MODE offset: {mode_offset} s (agrees on {mode_count}/{n_probes} probes)")

    # Sliding 30s Pearson at the mode offset
    centers, rs = scan_pearson_at_offset(ds_t, ds_mag, mem_t, mem_mag, mode_offset, win_s=30.0)
    pct_match = float(np.mean(rs > 0.9) * 100)

    print(f"  Sliding-30s Pearson at offset {mode_offset}: pct windows with r>0.9 = {pct_match:.1f}%")

    ax.plot(centers / 60, rs, lw=0.8, color="C0")
    ax.axhline(0.9, color="red", ls="--", lw=0.5, label="r=0.9")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(
        f"rec_{rec}: sliding 30s Pearson at offset = {mode_offset:.1f}s "
        f"(agrees {mode_count}/{n_probes} probes, {pct_match:.0f}% windows r>0.9)"
    )
    ax.set_ylabel("Pearson r")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    results[rec] = {"offset": mode_offset, "agree": mode_count, "pct_match": pct_match,
                    "probe_offsets": cand_offsets, "probe_corrs": cand_corrs}

axes[-1].set_xlabel("Session time [min] (dataset clock)")
plt.tight_layout()
despine()
plt.savefig(OUT / "match_quality_special.png", dpi=110)
plt.close()
print("\nWrote", OUT / "match_quality_special.png")

import json
(OUT / "special_offsets.json").write_text(json.dumps(results, indent=2))
print("Wrote", OUT / "special_offsets.json")
