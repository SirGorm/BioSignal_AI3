---
name: data-inspection
description: Use FIRST when working with this dataset, before labeling, feature extraction, or training. Loads one or more recordings and produces a comprehensive inspection report with signal statistics, time-series plots, spectral plots, timestamp alignment verification (Unix time), and detected anomalies. The output is the source of truth that all subsequent agents reference. Always run this on a new dataset or when sample rates/file formats are uncertain.
---

# Data Inspection (run before anything else)

Before labeling, feature extraction, or modeling, you read one real recording and produce a reproducible inspection report. The downstream pipeline reads its assumptions FROM your output rather than from defaults — so this step grounds the entire project in actual data.

## Inputs

A `(subject_id, session_id)` pair from `data/raw/<subject_id>/<session_id>/`. Pick a session that's likely representative (not the very first or very last; mid-protocol).

## Outputs

```
inspections/<subject_id>_<session_id>/
├── report.md                        # Human-readable findings
├── signal_overview.png              # All channels, full session, time-axis
├── signal_zoomed_set1.png           # First detected active set, all channels
├── ppg_channel_check.png            # All 4 PPG wavelengths overlaid
├── psd_<channel>.png                # Per-channel power spectrum (one figure each)
├── timestamp_alignment.png          # Biosignal vs joint-angle Unix time
├── sets_detected.png                # acc-magnitude with set boundaries marked
├── joint_coverage.png               # Joint-angle availability vs detected sets
├── stats.json                       # Machine-readable signal statistics
└── findings.md                      # Action items / data-quality flags
```

The findings.md drives updates to `CLAUDE.md` (sample rates, channel names, gotchas).

## Step-by-step procedure

### 1. List files

```python
from pathlib import Path
sess = Path('data/raw') / subject_id / session_id
print('Files in session directory:')
for f in sorted(sess.iterdir()):
    print(f'  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)')
```

### 2. Load biosignals at native rates

Don't resample yet. Keep each modality at its original rate so you can verify what it actually is.

```python
import pandas as pd, numpy as np

# CSV with Unix timestamps (most likely format for this project)
bio_df = pd.read_csv(sess / 'biosignals.csv')
print('Columns:', bio_df.columns.tolist())
print('First 3 rows:')
print(bio_df.head(3))
print('Last 3 rows:')
print(bio_df.tail(3))

# Detect timestamp column
ts_col = next((c for c in ['t_unix', 'timestamp', 'unix_time', 't', 'time']
               if c in bio_df.columns), None)
assert ts_col, f"No timestamp column found in {bio_df.columns.tolist()}"

# Verify it looks like Unix time (>1e9 = post-2001)
ts = bio_df[ts_col].values
if ts[0] < 1e9:
    print(f"WARNING: timestamps look like seconds-since-start, not Unix epoch.")
    print(f"  First: {ts[0]}, Last: {ts[-1]}")
else:
    from datetime import datetime, timezone
    print(f"Unix timestamps confirmed.")
    print(f"  Recording start: {datetime.fromtimestamp(ts[0], tz=timezone.utc)}")
    print(f"  Recording end:   {datetime.fromtimestamp(ts[-1], tz=timezone.utc)}")
    print(f"  Duration:        {(ts[-1] - ts[0]) / 60:.1f} minutes")
```

### 3. Compute actual sample rates

```python
def estimate_sample_rate(timestamps):
    """Robust sample rate from median inter-sample interval."""
    dt = np.median(np.diff(timestamps))
    return 1.0 / dt

# If all channels share one timestamp column, fs is the same for all
# If channels have separate timestamps (mixed-rate hardware), compute per channel
fs_global = estimate_sample_rate(ts)
print(f'Global fs (from timestamp column): {fs_global:.2f} Hz')

# Some hardware records all channels at the highest rate with NaN-padding
# for slower channels. Detect by checking NaN density per channel.
for c in [c for c in bio_df.columns if c != ts_col]:
    n_valid = bio_df[c].notna().sum()
    duration_s = ts[-1] - ts[0]
    eff_fs = n_valid / duration_s
    nan_pct = bio_df[c].isna().mean() * 100
    print(f'  {c}: effective fs = {eff_fs:.1f} Hz, {nan_pct:.1f}% NaN')
```

### 4. Per-channel statistics

```python
import json

def channel_stats(name, x, fs):
    x = x[~np.isnan(x)] if hasattr(x, '__len__') else x
    if len(x) == 0:
        return {'name': name, 'fs_hz': fs, 'status': 'empty'}
    return {
        'name': name,
        'fs_hz': float(fs),
        'n_samples': int(len(x)),
        'duration_s': float(len(x) / fs),
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
        'p1': float(np.percentile(x, 1)),
        'p99': float(np.percentile(x, 99)),
        'n_nan': int(np.sum(np.isnan(x) if hasattr(x, '__len__') else [])),
        'pct_nan': float(np.mean(np.isnan(x)) * 100) if hasattr(x, '__len__') else 0,
        'pct_at_max': float(np.mean(np.abs(x) >= np.max(np.abs(x)) * 0.99) * 100),
    }

stats = {c: channel_stats(c, bio_df[c].values, fs_global)
         for c in bio_df.columns if c != ts_col}
with open(out_dir / 'stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

### 5. Plot full-session overview

All plots in this project use the seaborn `white` theme + `sns.despine` to
strip top/right spines. Use the project helper `src/eval/plot_style.py` so
every figure is consistent — call `apply_style()` once at module import,
and `despine(fig=fig)` (or no args for the current figure) right before
each `savefig`.

```python
import matplotlib.pyplot as plt
from src.eval.plot_style import apply_style, despine

apply_style()

channels = [c for c in bio_df.columns if c != ts_col]
fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2 * len(channels)),
                          sharex=True)
session_t = ts - ts[0]  # convert to session-relative seconds for plotting

# Downsample for plot if huge
step = max(1, len(session_t) // 50_000)
for ax, c in zip(axes, channels):
    ax.plot(session_t[::step] / 60, bio_df[c].values[::step], lw=0.5)
    ax.set_ylabel(c)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Session time [minutes]')
plt.suptitle(f'{subject_id} / {session_id} — full session overview')
plt.tight_layout()
despine(fig=fig)
plt.savefig(out_dir / 'signal_overview.png', dpi=110)
plt.close()
```

### 6. PPG channel sanity check (4 wavelengths)

The project uses only the green channel but loads all 4. Verify which is which:

```python
ppg_cols = [c for c in bio_df.columns if 'ppg' in c.lower()]
print(f'PPG columns found: {ppg_cols}')

if len(ppg_cols) >= 2:
    fig, ax = plt.subplots(figsize=(14, 4))
    # Plot a 30s window from a quiet period
    t0_idx = int(30 * fs_global)
    t1_idx = int(60 * fs_global)
    for c in ppg_cols:
        ax.plot(session_t[t0_idx:t1_idx], bio_df[c].values[t0_idx:t1_idx],
                label=c, lw=0.7)
    ax.set_xlabel('Session time [s]')
    ax.set_title('PPG channels — verify which is green')
    ax.legend()
    plt.savefig(out_dir / 'ppg_channel_check.png', dpi=110)
    plt.close()

# Print amplitude ratio: green typically has best SNR
for c in ppg_cols:
    x = bio_df[c].dropna().values
    if len(x):
        print(f'  {c}: range = {x.max() - x.min():.0f}, std = {x.std():.0f}')
```

If the column "ppg_green" doesn't exist but you find e.g. `ppg_2` or `ppg_550`, document this in findings.md and propose a renaming for `data-loader`.

### 7. Power spectra per channel

```python
from scipy.signal import welch

for c in channels:
    x = bio_df[c].dropna().values
    if len(x) < 256:
        continue
    fs_c = stats[c]['fs_hz']
    f, pxx = welch(x, fs=fs_c, nperseg=min(4096, len(x) // 4))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(f, pxx)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD')
    ax.set_title(f'{c} — fs = {fs_c:.1f} Hz')
    ax.grid(alpha=0.3, which='both')

    # Mark common artifact bands
    for hz in [50, 60]:
        ax.axvline(hz, color='red', ls='--', alpha=0.4, lw=0.8)

    plt.tight_layout()
    plt.savefig(out_dir / f'psd_{c}.png', dpi=110)
    plt.close()
```

Inspect each PSD and note in findings.md:
- Line noise visible at 50 or 60 Hz?
- ECG: dominant peak around HR (1–2 Hz)?
- EMG: power up to 200–400 Hz?
- PPG-green: dominant peak at HR fundamental + harmonic?

### 8. Load joint-angle data (Unix time sync — no offset needed)

```python
joint_path = sess / 'joint_angles.csv'
joint_df = pd.read_csv(joint_path)
print('Joint columns:', joint_df.columns.tolist())
print('Joint head:')
print(joint_df.head(3))

joint_ts_col = next((c for c in ['t_unix', 'timestamp', 'unix_time', 't', 'time']
                     if c in joint_df.columns), None)
joint_ts = joint_df[joint_ts_col].values

assert joint_ts[0] > 1e9, "Joint timestamps must be Unix epoch for sync to work"

# THE key check: do biosignal and joint Unix times overlap?
print(f'Biosignal time range: [{ts[0]:.1f}, {ts[-1]:.1f}]')
print(f'Joint     time range: [{joint_ts[0]:.1f}, {joint_ts[-1]:.1f}]')
overlap_start = max(ts[0], joint_ts[0])
overlap_end = min(ts[-1], joint_ts[-1])
overlap_s = max(0, overlap_end - overlap_start)
print(f'Overlap: {overlap_s:.1f} s ({overlap_s/60:.1f} min)')

assert overlap_s > 60, "Joint and biosignal recordings barely overlap — wrong files?"
```

If both timestamps are Unix epochs and overlap is reasonable, **synchronization is solved**. No offset estimation, no signal correlation. The Unix time is the authoritative shared clock.

### 9. Timestamp alignment plot

Visual confirmation that the two streams cover the expected joint windows:

```python
fig, ax = plt.subplots(figsize=(14, 3))
# Biosignal availability (every sample)
ax.scatter(ts - ts[0], np.ones_like(ts) * 1, s=0.5, label='biosignal', alpha=0.5)
# Joint availability (will be sparse — only during sets)
ax.scatter(joint_ts - ts[0], np.ones_like(joint_ts) * 0, s=0.5,
           label='joint', alpha=0.5, color='orange')
ax.set_yticks([0, 1]); ax.set_yticklabels(['joint', 'biosignal'])
ax.set_xlabel('Session time [s] (relative to biosignal start)')
ax.set_title('Timestamp coverage (Unix time anchored)')
ax.legend(loc='right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'timestamp_alignment.png', dpi=110)
plt.close()
```

### 10. Detect active sets and verify against participants.xlsx

```python
# Build acc-magnitude
ax_, ay_, az_ = bio_df['ax'].values, bio_df['ay'].values, bio_df['az'].values
acc_mag = np.sqrt(ax_**2 + ay_**2 + az_**2)
fs_acc = stats['ax']['fs_hz']

# Quick segmentation (use the session-segmentation skill in production)
from scipy.ndimage import uniform_filter1d
g_baseline = np.median(acc_mag[:int(30 * fs_acc)])
smooth = uniform_filter1d(np.abs(acc_mag - g_baseline),
                           size=int(1.0 * fs_acc))
is_active = smooth > 0.3
# Find segments and filter min-duration
diff = np.diff(is_active.astype(int))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0]
if len(starts) and len(ends):
    if ends[0] < starts[0]: ends = ends[1:]
    n = min(len(starts), len(ends))
    segments = [(starts[i], ends[i]) for i in range(n)
                if (ends[i] - starts[i]) / fs_acc > 20]
else:
    segments = []
print(f'Detected {len(segments)} active segments (min 20 s)')

# Compare with participants.xlsx
parts_path = sess / 'participants.xlsx'
if not parts_path.exists():
    parts_path = Path('data/raw/participants.xlsx')
if parts_path.exists():
    parts = pd.read_excel(parts_path)
    parts.columns = [c.strip().lower().replace(' ', '_') for c in parts.columns]
    sess_parts = parts[(parts['subject_id'].astype(str) == str(subject_id)) &
                        (parts['session_id'].astype(str) == str(session_id))]
    print(f'Spreadsheet expects {len(sess_parts)} sets:')
    print(sess_parts[['exercise', 'set_number', 'set_start_time',
                       'set_end_time', 'rpe']])

# Plot detected sets over acc-magnitude
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(session_t / 60, acc_mag, lw=0.3, color='gray', alpha=0.6)
ax.plot(session_t[::step] / 60, smooth[::step] + g_baseline, lw=0.8, color='black',
        label='smoothed |acc - g|')
for s, e in segments:
    ax.axvspan(session_t[s] / 60, session_t[e] / 60, color='C2', alpha=0.2)
ax.set_xlabel('Session time [min]')
ax.set_ylabel('|acc| [m/s²]')
ax.set_title(f'Detected sets: {len(segments)}')
ax.legend()
plt.tight_layout()
plt.savefig(out_dir / 'sets_detected.png', dpi=110)
plt.close()
```

### 11. Joint-angle availability over detected sets

The reps/phase ground truth requires joint data DURING active sets. Verify coverage:

```python
fig, ax = plt.subplots(figsize=(14, 3))
# Active sets as background
for s, e in segments:
    ax.axvspan(session_t[s] / 60, session_t[e] / 60,
               color='C2', alpha=0.2, label='detected set' if s == segments[0][0] else None)
# Joint data presence
joint_session_t = (joint_ts - ts[0]) / 60
ax.scatter(joint_session_t, np.ones_like(joint_ts), s=0.5,
           color='orange', label='joint sample available')
ax.set_xlabel('Session time [min]')
ax.set_yticks([])
ax.set_title('Joint-angle coverage vs detected active sets')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(out_dir / 'joint_coverage.png', dpi=110)
plt.close()
```

If joint samples are present outside detected sets — or missing during detected sets — flag in findings.md.

### 12. Write findings.md

```markdown
# Inspection findings: <subject_id> / <session_id>

## Hardware (CONFIRMED)
| Modality | Column | Sample rate | Unit | Range |
|----------|--------|-------------|------|-------|
| ECG      | ecg    | 500 Hz      | mV   | -2.0 .. 1.8 |
| EMG      | emg    | 1000 Hz     | mV   | -1.5 .. 1.5 |
| ...      | ...    | ...         | ...  | ... |

## Synchronization
- Biosignal Unix start: 1714123456.789 (2024-04-26 10:30:56 UTC)
- Joint Unix start:     1714123456.812
- Time offset:          0.023 s — within sample interval, OK
- Overlap:              28.4 min

## Sets detected vs spreadsheet
- Detected: 9 active segments
- Expected (from participants.xlsx): 9 sets (3 squat, 3 bench, 3 deadlift)
- Match: ALL 9 detected segments overlap a spreadsheet entry within 5 s

## Action items for CLAUDE.md
- ECG fs is 500 Hz (default placeholder was 250) → update default.yaml
- PPG-green is column `ppg_2` not `ppg_green` → add rename in data-loader
- 50 Hz line noise visible on ECG and EMG → notch filter MANDATORY (already in default config)
- Subject S001 has temp = 0 for first 90 s → temp sensor warm-up; skip first 90 s baseline

## Quality flags
- None — data looks clean

## Recommendations
- Proceed to /label
- Set acc-segmentation threshold to 0.25 g (default 0.3 g misses warm-up rep)
```

## When to re-run inspection

- New hardware setup or firmware update on the device
- New subject cohort
- Anything in findings.md changes (then update CLAUDE.md too)
- Any agent or hook reports unexpected schema

## How downstream agents use this

The data-labeler reads `inspections/<subj>_<sess>/stats.json` to determine actual sample rates per channel rather than trusting `configs/default.yaml`. If stats.json is absent for a session, the data-labeler refuses to proceed and prompts you to run inspection first.

The biosignal-feature-extractor reads stats.json to set filter parameters per actual fs.

The ml-expert reads stats.json to flag any subject whose data deviates significantly from the cohort norm (e.g., one subject's ECG range is 10× higher → bad amplifier setting).

## Hard rules

- **Always inspect before labeling a new dataset.** Never trust default sample rates.
- **Always verify Unix timestamps** with `>1e9` check. Session-relative timestamps require manual offset estimation; Unix doesn't.
- **Always save plots**, even if they look fine. They're the audit trail.
- **All plots use seaborn + despine** via `src/eval/plot_style.py` (`apply_style()` once, `despine(fig=fig)` before every `savefig`). No bare-matplotlib figures.
- **Write findings.md in human-readable form** so the user can read it and update CLAUDE.md accordingly.

## References

When findings.md makes claims about typical signal characteristics or quality criteria:

- **De Luca 1997** — EMG signal characteristics, expected frequency range
- **Allen 2007** — PPG morphology and HR derivation
- **Task Force 1996** — ECG-based HRV measurement standards
- **Greco et al. 2016** — EDA decomposition assumptions
- **Maeda et al. 2011** — wrist-PPG motion artifacts

Full entries in `literature-references` skill. Never invent.
