"""
Verify whether dataset_memory and dataset are the SAME physical recording on
different clocks, by cross-correlating acc-magnitude. For each recording:

  1. Build acc_mag from ax/ay/az (both sources, fs=100 Hz).
  2. Take three 60 s probe-windows from dataset (early, middle, late).
  3. For each probe, find best lag in memory via FFT cross-correlation,
     then compute the EXACT Pearson correlation at that lag.
  4. Estimate per-probe time-offset = mem_t[lag] - ds_t[probe_start].
  5. Report whether the three offsets agree (constant clock skew, no drift)
     and whether peak Pearson > 0.9 (signals truly match).
  6. Plot dataset vs memory aligned on dataset's clock.

Outputs verification PNGs to inspections/_verify_memory_dataset/
and a summary table.
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

DATASET = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset")
MEMORY = Path(r"C:\Users\skogl\Downloads\eirikgsk\BioSignal_AI\dataset_memory")
OUT = Path(r"c:\Users\skogl\Downloads\eirikgsk\biosignal_2\BioSignal_AI3\inspections\_verify_memory_dataset")
OUT.mkdir(parents=True, exist_ok=True)

RECS = ["006", "007", "008", "009", "010", "011", "012", "013", "014"]

FS = 100.0
WIN_S = 60.0
WIN_N = int(WIN_S * FS)


def load_acc_mag(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    ax = pd.read_csv(folder / "ax.csv")
    ay = pd.read_csv(folder / "ay.csv")
    az = pd.read_csv(folder / "az.csv")
    n = min(len(ax), len(ay), len(az))
    t = ax["timestamp"].to_numpy()[:n]
    mag = np.sqrt(
        ax["ax"].to_numpy()[:n] ** 2
        + ay["ay"].to_numpy()[:n] ** 2
        + az["az"].to_numpy()[:n] ** 2
    )
    return t, mag


def best_lag_pearson(template: np.ndarray, signal: np.ndarray) -> tuple[int, float]:
    """FFT cross-correlation to find candidate lag, then exact Pearson at top-K lags."""
    if len(signal) < len(template):
        return 0, 0.0
    t = template - template.mean()
    s = signal - signal.mean()
    # FFT correlate to rank candidate lags fast
    corr = correlate(s, t, mode="valid", method="fft")
    # Take top-5 candidate lags, score each with exact Pearson r
    K = min(5, len(corr))
    cand_lags = np.argpartition(-corr, K - 1)[:K]
    best = (0, -np.inf)
    n = len(template)
    for lag in cand_lags:
        win = signal[lag : lag + n]
        if len(win) < n:
            continue
        r = float(np.corrcoef(template, win)[0, 1])
        if r > best[1]:
            best = (int(lag), r)
    return best


print(f"\n{'rec':>4} {'probe':>5} {'lag_s':>10} {'corr':>5} {'offset_s':>10}")
print("-" * 50)

results: dict[str, dict] = {}
for rec in RECS:
    ds_dir = DATASET / f"recording_{rec}"
    mem_dir = MEMORY / f"recording_{rec}_memory"
    if not (ds_dir.exists() and mem_dir.exists()):
        continue
    ds_t, ds_mag = load_acc_mag(ds_dir)
    mem_t, mem_mag = load_acc_mag(mem_dir)

    # Three probes: 25 %, 50 %, 75 % of dataset
    n_ds = len(ds_mag)
    probe_centers = [int(n_ds * f) for f in (0.25, 0.50, 0.75)]
    offsets = []
    corrs = []
    lags = []
    for p in probe_centers:
        t0 = max(0, p - WIN_N // 2)
        t1 = min(n_ds, t0 + WIN_N)
        if t1 - t0 < WIN_N // 2:
            continue
        template = ds_mag[t0:t1]
        lag, r = best_lag_pearson(template, mem_mag)
        offset = float(mem_t[lag] - ds_t[t0])  # mem_clock - ds_clock
        offsets.append(offset)
        corrs.append(r)
        lags.append(lag)
        print(f"{rec:>4} {p/n_ds*100:4.0f}% {lag/FS:10.2f} {r:5.2f} {offset:10.3f}")

    if offsets:
        med_offset = float(np.median(offsets))
        med_corr = float(np.median(corrs))
        offset_spread = float(max(offsets) - min(offsets))
        print(f"      median offset = {med_offset:10.3f} s  (spread {offset_spread:.3f}s, median corr {med_corr:.2f})")
        results[rec] = {
            "ds_t0": float(ds_t[0]),
            "ds_t1": float(ds_t[-1]),
            "mem_t0": float(mem_t[0]),
            "mem_t1": float(mem_t[-1]),
            "offsets": offsets,
            "corrs": corrs,
            "med_offset": med_offset,
            "med_corr": med_corr,
            "offset_spread": offset_spread,
        }

print("\n\n=== Summary ===")
print(
    f"{'rec':>4}  {'med_corr':>8}  {'med_offset_s':>12}  {'spread_s':>9}  "
    f"{'aligned_pre_s':>13}  {'aligned_post_s':>14}  {'aligned_overlap_s':>17}  status"
)
print("-" * 110)
for rec, r in results.items():
    aligned_first = r["mem_t0"] - r["med_offset"]
    aligned_last = r["mem_t1"] - r["med_offset"]
    pre = r["ds_t0"] - aligned_first  # how many seconds memory starts before dataset
    post = aligned_last - r["ds_t1"]  # seconds memory extends after dataset
    overlap = min(r["ds_t1"], aligned_last) - max(r["ds_t0"], aligned_first)
    status = (
        "OK"
        if r["med_corr"] > 0.9 and r["offset_spread"] < 0.5
        else ("DRIFT?" if r["offset_spread"] > 0.5 else "LOW_CORR")
    )
    print(
        f"{rec:>4}  {r['med_corr']:8.3f}  {r['med_offset']:12.3f}  {r['offset_spread']:9.3f}  "
        f"{pre:13.1f}  {post:14.1f}  {overlap:17.1f}  {status}"
    )

# ---------- Plot: dataset vs memory acc-mag aligned, per recording ----------
print("\nWriting alignment plots ...")
fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.0 * len(results)))
if len(results) == 1:
    axes = [axes]
for ax, (rec, r) in zip(axes, results.items()):
    ds_dir = DATASET / f"recording_{rec}"
    mem_dir = MEMORY / f"recording_{rec}_memory"
    ds_t, ds_mag = load_acc_mag(ds_dir)
    mem_t, mem_mag = load_acc_mag(mem_dir)
    mem_t_aligned = mem_t - r["med_offset"]

    # Use ds_t[0] as zero
    ax.plot((ds_t - ds_t[0]) / 60, ds_mag, lw=0.4, color="C0", alpha=0.7, label="dataset")
    ax.plot((mem_t_aligned - ds_t[0]) / 60, mem_mag, lw=0.4, color="C1", alpha=0.7, label="memory (aligned)")
    ax.set_title(
        f"rec_{rec}  offset={r['med_offset']:.2f}s  Pearson={r['med_corr']:.2f}  "
        f"(memory pre/post = {r['ds_t0'] - (r['mem_t0']-r['med_offset']):.1f}s / "
        f"{(r['mem_t1']-r['med_offset']) - r['ds_t1']:.1f}s)",
        fontsize=10,
    )
    ax.set_ylabel("|acc| m/s²")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Session time [min] (dataset clock)")
plt.tight_layout()
despine()
plt.savefig(OUT / "memory_vs_dataset_acc_aligned.png", dpi=110)
plt.close()
print(f"  {OUT / 'memory_vs_dataset_acc_aligned.png'}")

# Save results json
import json

(OUT / "offsets.json").write_text(json.dumps(results, indent=2))
print(f"  {OUT / 'offsets.json'}")
