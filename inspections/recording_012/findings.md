# Findings — recording_012 (Tias)

UTC start: 2026-03-12 18:47:29
Duration: 36.2 min
Total sets: 12 (markers ✓ metadata ✓ Participants.xlsx ✓)
Reps total: 105 (range 8–10 per set)

## CONFIRMED hardware (measured from CSV timestamps, not declared)

| Modality | Column file | fs (measured) | Range | NaN% | Status |
|----------|-------------|--------------:|-------|-----:|--------|
| ECG | `ecg.csv` (cols `timestamp, ecg`) | 499.98 Hz | -5.55e-3 … 5.55e-3 V | 0% | OK |
| EMG | `emg.csv` (cols `timestamp, emg`) | **2000.14 Hz** | -4.42e-3 … 4.56e-3 V | 0% | OK |
| EDA | `eda.csv` (cols `timestamp, eda`) | 50.00 Hz | 2.39e-7 … 3.18e-7 S | 0% | OK |
| Temperature | `temperature.csv` | declared 1 Hz | — | — | **EMPTY (header only)** |
| PPG-green | `ppg_green.csv` | 100.00 Hz | 4.74e3 … 6.60e3 (raw) | 0% | OK |
| PPG-blue/red/ir | `ppg_{blue,red,ir}.csv` | 100.00 Hz each | — | 0% | logged but unused |
| ax / ay / az | `ax.csv`, `ay.csv`, `az.csv` | 100.00 Hz each | ~ ±78 m/s² | 0% | OK |
| acc_mag (computed) | sqrt(ax²+ay²+az²) | 100.00 Hz | 0.38 … 122 | 0% | OK |
| `a_combined.csv` | extra file w/ "Sampling Rate: 100 Hz" header | 100 Hz | -0.08 … 0.03 | — | redundant — ignore |

Declared in `metadata.json`: `{emg: 2000, ecg: 500, temperature: 1, eda: 50, ppg: 100, imu: 100}` — measurements match declared (except temperature file is empty).

## Synchronization

- All CSV files share Unix-epoch timestamps (`>1e9` ✓). bio_t0 = 1773341249.6383 s.
- ECG, EMG, EDA, PPG all start within ~5 µs of each other (single shared start across modalities).
- **Joint skeleton files (`recording_NN_joints.json`) carry per-frame `timestamp_usec = 0`** — the JSONs themselves are NOT Unix-anchored.
- Sync between joint skeleton frames and biosignals must come from `markers.json` (`Set:N_Start unix_time`) and from `metadata.json["kinect_sets"]`. Within a set, frame_id × (1/Kinect_fps) is the offset relative to start_unix.
- All 12 joint files present (398–662 frames per set, ~30 fps × ~25 s consistent with Kinect Azure DK).

## Labels available

- **Per-set** from `Participants.xlsx` (sheet has 2 rows per recording: exercise row + fatigue row, columns `set1`..`set12`):
  - Exercises: `pullup×3, benchpress×3, deadlift×3, squat×3` (4 exercises × 3 sets)
  - RPE (fatigue): `[4, 6, 8, 6, 6, 6, 6, 6, 8, 6, 7, 8]`
- **Per-rep** from `markers.json` — every rep has a Unix timestamp. This is GROUND TRUTH for rep timing, far better than acc-magnitude peak detection.
- **Per-frame skeleton** from `recording_NN_joints.json` — 32 joints with `joint_positions` (3D) and `joint_orientations` (quaternion). **Joint angles are NOT pre-computed**; the labeling pipeline must derive `knee_angle`, `hip_angle`, etc. from positions/orientations.

## Quality flags

1. **Temperature data missing** (`temperature.csv` is empty — header only). Spot-check shows this also affects rec 013. Drop temperature features for these recordings or treat session-level `temp` as missing.
2. **Acc-magnitude segmentation over-counts sets (42 detected vs 12 actual)** — the simple `>0.3 g` threshold catches every rest-period hand movement. **Use `markers.json` set boundaries directly; do not rely on acc-segmentation for these recordings.** State-machine for streaming will need higher threshold + longer min-duration + cooldown.
3. ECG/EMG amplifier ranges are tiny (mV-scale signed → likely already in volts). PSDs (see `psd_ecg.png`, `psd_emg.png`) — confirm whether 50 Hz line noise is present before finalizing notch.
4. Recording duration ~36 min covers the full 12-set protocol (≈3 min per set+rest cycle).
5. EDA values are in microsiemens range converted to siemens (~ 1e-7) — sanity check the unit and consider rescaling.

## Action items for `CLAUDE.md` and `configs/default.yaml`

- [ ] **EMG fs is 2000 Hz**, not the 1000 Hz template in CLAUDE.md. Update the hardware table and EMG bandpass parameters (Nyquist now 1000 Hz; current "20–450 Hz BP" still fine).
- [ ] **PPG fs is 100 Hz** for these recordings (was 50 Hz on `recording_001`). Note in CLAUDE.md that PPG fs varies between sessions and must be read from `metadata.json["sampling_rates"]["ppg"]` per recording.
- [ ] **Protocol is 12 sets = 4 exercises × 3 sets**, exercises = {squat, deadlift, benchpress, pullup}. CLAUDE.md says "3 sets per exercise, 9 total" — update to 12 total, list pullup in `configs/exercises.yaml`.
- [ ] **Data layout differs from CLAUDE.md** (`data/raw/<subj>/<sess>/biosignals.csv`). Actual: `dataset/recording_NNN/{ecg,emg,eda,...}.csv` (one file per modality). Update `data-loader` skill or `src/data/` accordingly.
- [ ] **Joint-angle source is raw Kinect skeleton**, not a CSV with `knee_angle` columns. The data-labeler must compute joint angles from `joint_positions` (3D) and select primary joint per exercise (squat→knee, bench→elbow, deadlift→hip, pullup→elbow).
- [ ] **Joint timestamps are anchored via `markers.json`** — joint frames have `timestamp_usec = 0` internally. Use `markers.json["Set:N_Start"]["unix_time"]` + `frame_id / kinect_fps` to map frames to Unix time.
- [ ] **`participants.xlsx` location is `dataset/Participants/Participants.xlsx`**. Schema: `Recording:` (recording number), `Name:` (participant), `set1..set12` columns; alternating rows = exercise / fatigue (RPE). Update CLAUDE.md schema table accordingly.
- [ ] **Temperature missing for newer recordings** — guard the loader against empty `temperature.csv`. Don't fail the run; mark temperature features as NaN.
- [ ] **Use `markers.json` for set/rep ground truth**, not acc-magnitude segmentation. The session-segmentation skill is still useful for *streaming/inference*, where markers don't exist.
- [ ] In `configs/default.yaml` set: `ecg.fs=500, emg.fs=2000, eda.fs=50, ppg.fs=100, imu.fs=100, temp.fs=1` (and document fallback when temperature is missing).

## Recommendation

Proceed to `/inspect` on at least one early recording (e.g. `recording_001`, where PPG fs=50 Hz and temperature data exist) to confirm the hardware-config drift across the cohort before running `/label --all`.

## References

- Sample-rate verification approach (timestamp delta, not declared): standard practice. No specific citation.
- EMG bandpass 20–450 Hz: De Luca 1997.
- ECG-derived HRV at 250–500 Hz minimum: Task Force 1996.
- 50 Hz line-noise notch necessity for EU mains: De Luca 1997.
- Wrist PPG motion artifact sensitivity: Maeda et al. 2011.

(All references in `literature-references` skill.)
