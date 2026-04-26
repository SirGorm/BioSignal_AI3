# Findings — recording_013 (Juile)

UTC start: 2026-03-12 19:36:33
Duration: 37.7 min
Total sets: 12 (markers ✓ metadata ✓ Participants.xlsx ✓)
Reps total: 105 (range 7–10 per set)

## CONFIRMED hardware (measured from CSV timestamps)

| Modality | Column file | fs (measured) | Range | NaN% | Status |
|----------|-------------|--------------:|-------|-----:|--------|
| ECG | `ecg.csv` | 499.98 Hz | -5.43e-3 … 5.54e-3 V | 0% | OK |
| EMG | `emg.csv` | **2000.14 Hz** | -4.48e-3 … 4.08e-3 V | 0% | OK |
| EDA | `eda.csv` | 50.00 Hz | 2.33e-7 … 2.57e-7 S | 0% | OK |
| Temperature | `temperature.csv` | declared 1 Hz | — | — | **EMPTY (header only)** |
| PPG-green | `ppg_green.csv` | 100.00 Hz | 5.03e3 … 6.63e3 (raw) | 0% | OK |
| PPG-blue/red/ir | `ppg_{blue,red,ir}.csv` | 100.00 Hz each | — | 0% | logged but unused |
| ax / ay / az | `ax.csv`, `ay.csv`, `az.csv` | 100.00 Hz each | ~ ±65 m/s² | 0% | OK |
| acc_mag (computed) | sqrt(ax²+ay²+az²) | 100.00 Hz | 0.45 … 95 | 0% | OK |
| `a_combined.csv` | aux file w/ "Sampling Rate: 100 Hz" header | 100 Hz | — | — | redundant — ignore |

Declared in `metadata.json`: `{emg: 2000, ecg: 500, temperature: 1, eda: 50, ppg: 100, imu: 100}` — measurements match declared (except temperature file is empty).

## Synchronization

- All CSV files share Unix-epoch timestamps (`>1e9` ✓). bio_t0 = 1773344193.9778 s.
- ECG, EMG, EDA, PPG, IMU all start within microseconds of each other (single shared start across modalities).
- Joint skeleton files (`recording_NN_joints.json`) carry per-frame `timestamp_usec = 0` — JSON internal time is NOT Unix-anchored.
- Sync between joint frames and biosignals derives from `markers.json` (`Set:N_Start unix_time`) plus `metadata.json["kinect_sets"]`.
- All 12 joint files present (400–771 frames per set, ~30 fps × set-duration consistent with Azure Kinect).

## Labels available

- **Per-set** from `Participants.xlsx`:
  - Exercises: `deadlift×3, squat×3, benchpress×3, pullup×3`
  - RPE (fatigue): `[4, 5, 7, 5, 7, 7, 8, 7, 7, 7, 8, 9]`
- **Per-rep** from `markers.json` — explicit Unix-time markers for each rep.
- **Per-frame skeleton** from `recording_NN_joints.json` (32 joints, positions + orientations). Joint angles must be derived downstream.

## Quality flags

1. **Temperature data missing** (empty file) — same as recording_012. Either a sensor disconnect on the recording rig, a logger config bug for this batch, or a deliberate change to drop temp from newer protocols. Flag for the user.
2. **Acc-magnitude segmentation over-counts sets (29 detected vs 12 actual)** — same threshold issue as recording_012, slightly less severe (Juile may move the wrist less between sets). Use `markers.json` for offline ground truth.
3. **Set duration mean 29.9 s** vs 25.3 s on rec 012 — Juile is slower on reps (or has higher RPE pulled by more difficult sets); makes sense given the higher RPE distribution `[4,5,7,5,7,7,8,7,7,7,8,9]`.
4. **Last set RPE = 9** (squat→pullup×3 final set) — strongest fatigue signal in this session; useful test case for the fatigue model.
5. EDA range 2.3e-7 … 2.6e-7 S is much narrower than rec 012 (3.18e-7) — Juile may have lower baseline arousal, or the sensor contact differs. Watch for inter-subject EDA scaling.

## Action items for `CLAUDE.md` and `configs/default.yaml`

Same as `recording_012/findings.md` — see that file for the consolidated list. Specifically reconfirmed here:

- [x] EMG fs = 2000 Hz (not 1000 Hz)
- [x] PPG fs = 100 Hz (not 50 Hz on these newer recordings)
- [x] 12 sets / 4 exercises (squat, deadlift, benchpress, pullup) — pullup must be added to `configs/exercises.yaml`
- [x] Data layout = per-modality CSVs (`ecg.csv`, `emg.csv`, …) under `dataset/recording_NNN/`
- [x] Joint angles must be computed from raw Kinect skeleton; sync via `markers.json`
- [x] `participants.xlsx` at `dataset/Participants/Participants.xlsx`, dual-row schema (exercise + fatigue rows per recording)
- [x] Temperature can be empty — loader must tolerate missing temperature gracefully
- [ ] Cross-subject EDA normalization (per-subject baseline) is necessary; absolute amplitudes vary 30%+ between Tias and Juile.

## Recommendation

Together with recording_012, this confirms the newer-protocol setup (EMG=2000 Hz, PPG=100 Hz, no temperature, 12 sets). Recommend `/inspect 001` or `/inspect 005` once to verify the older-protocol recordings, then update `CLAUDE.md` once for both regimes (and add a config `protocol_version` flag).

## References

- Inter-subject EDA baseline normalization: Greco et al. 2016.
- EMG fs 2000 Hz adequate for fatigue spectral indices (MNF/MDF up to ~500 Hz): De Luca 1997, Dimitrov et al. 2006.
- Per-rep markers as ground truth (preferred over signal-derived peak detection when available): consistent with project methodology — no specific external reference.

(All references in `literature-references` skill.)
