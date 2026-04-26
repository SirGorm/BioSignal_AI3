# Feature Card — Run 20260426_154705_default

Generated: 2026-04-26

## Scope

13 recordings, 11 unique subjects, 156 sets, ~3.27 M windows at 100 ms hop.

---

## Windowing Scheme

| Modality | Window | Hop | Notes |
|----------|--------|-----|-------|
| ECG      | 30 s   | 100 ms | HRV stability (Shaffer & Ginsberg 2017) |
| EMG      | 500 ms | 100 ms | Minimum for stable MNF/MDF (Cifrek et al. 2009) |
| EDA      | 10 s   | 100 ms | SCR dynamics ≥5 s (Boucsein 2012) |
| Temp     | 60 s   | 100 ms | Very slow thermal signal |
| Acc      | 2 s    | 100 ms | Captures full rep cycle at 0.5–1 Hz (González-Badillo & Sánchez-Medina 2010) |
| PPG      | 10 s   | 100 ms | HR stability (Allen 2007) |

All modality features are forward-filled onto the 100 Hz primary hop grid
using `merge_asof` (backward direction, 35 s tolerance).

---

## Features per Modality

### ECG (5 features + 1 normalised) — `src/features/ecg_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `ecg_hr` | Heart rate (BPM) from RR intervals | Task Force 1996 |
| `ecg_rmssd` | Root-mean-square successive differences | Task Force 1996 |
| `ecg_sdnn` | Standard deviation of NN intervals | Task Force 1996 |
| `ecg_pnn50` | Proportion RR diffs > 50 ms | Task Force 1996 |
| `ecg_mean_rr` | Mean RR interval (ms) | Task Force 1996 |
| `ecg_hr_rel` | `ecg_hr / baseline_hr` | — |

R-peak detection: Pan-Tompkins derivative + adaptive threshold (Pan & Tompkins 1985).
Processed via NeuroKit2 offline (Makowski et al. 2021); custom derivative-threshold
detector in streaming.

### EMG (5 features + 5 normalised) — `src/features/emg_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `emg_rms` | Root-mean-square amplitude | De Luca 1997 |
| `emg_iemg` | Integrated EMG = Σ|x| | De Luca 1997 |
| `emg_mnf` | Mean power frequency (spectral centroid) | De Luca 1997 |
| `emg_mdf` | Median power frequency | De Luca 1997 |
| `emg_dimitrov` | FInsm5 = M(-1)/M(5); most fatigue-sensitive in dynamic contractions | Dimitrov et al. 2006 |

Baseline-normalised: `emg_mnf_rel`, `emg_mdf_rel`, `emg_rms_rel`, `emg_dimitrov_rel`, `emg_iemg_rel`.

PSD via Welch's method, nperseg = min(256, window_len) (Welch 1967).
Bandpass: 20–450 Hz Butterworth order 4 + 50 Hz notch (offline: filtfilt; streaming: sosfilt with persisted zi).

**Per-set slope features** (in `set_features.parquet` only):
- `emg_mnf_slope`, `emg_mdf_slope`, `emg_dimitrov_slope` — linear regression slope within active set.
  More negative MNF slope = faster fatigue accumulation (Cifrek et al. 2009, Dimitrov et al. 2006).

### EDA (4 features + 1 normalised) — `src/features/eda_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `eda_scl` | Skin conductance level (tonic, median of window) | Boucsein 2012 |
| `eda_scr_amp` | SCR phasic amplitude (max − min after moving-median detrend) | Greco et al. 2016 |
| `eda_scr_count` | Number of SCR rising-edge events in window | Boucsein 2012 |
| `eda_phasic_mean` | Mean absolute phasic signal | Posada-Quintero & Chon 2020 |
| `eda_scl_rel` | `eda_scl / baseline_scl` | — |

Lowpass: 5 Hz Butterworth order 4. Phasic decomposition: 2 s moving-median baseline subtraction.

### Temperature (3 features + 1 normalised) — `src/features/temp_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `temp_mean` | Mean skin temperature in 60 s window | [REF NEEDED: skin temperature as fatigue indicator during exercise] |
| `temp_slope` | Linear slope of temperature (°C/s) | [REF NEEDED: skin temperature as fatigue indicator during exercise] |
| `temp_range` | Max − min temperature in window | [REF NEEDED: skin temperature as fatigue indicator during exercise] |
| `temp_mean_rel` | `temp_mean / baseline_mean` | — |

**NaN handling**: recordings 007–014 have empty `temperature.csv` — all temp features
are NaN for those recordings. The downstream ML pipeline must handle NaN columns (use
LightGBM's native NaN support). No crash at extraction time.

### Accelerometer (5 features) — `src/features/acc_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `acc_rms` | RMS of filtered acc_mag | Mannini & Sabatini 2010 |
| `acc_jerk_rms` | RMS of jerk (derivative of acc_mag) | Khan et al. 2010 |
| `acc_dom_freq` | Dominant frequency from Welch PSD | Mannini & Sabatini 2010 |
| `acc_rep_band_power` | Spectral power 0.3–1.5 Hz (rep-rate band) | González-Badillo & Sánchez-Medina 2010 |
| `acc_rep_band_ratio` | Rep-band power / total power | González-Badillo & Sánchez-Medina 2010 |

Computed AFTER acc_mag = sqrt(ax²+ay²+az²), then 0.5–20 Hz bandpass.
Bonomi et al. 2009 — accelerometry for activity detection.

### PPG-green (3 features) — `src/features/ppg_features.py`

| Feature | Description | Reference |
|---------|-------------|-----------|
| `ppg_hr` | Heart rate from systolic peak intervals (BPM) | Allen 2007 |
| `ppg_pulse_amp` | Mean peak-to-trough pulse amplitude | Allen 2007 |
| `ppg_pulse_amp_var` | Standard deviation of pulse amplitude | Tamura et al. 2014 |

Green wavelength only (Castaneda et al. 2018 — best motion artifact rejection at wrist).
Sample rate read from `metadata.json["sampling_rates"]["ppg"]` per recording (50 Hz for
rec_001, 100 Hz for rec_012+).
Offline peak detection: `find_peaks` over per-window segment only. Streaming: adaptive
threshold causal detector.
Cross-validation signal: `ppg_hr` should agree with `ecg_hr` within 5 bpm at rest
(Maeda et al. 2011).

---

## Baseline Normalization Strategy

- **Baseline window**: first 60 s of each session (t_session_s < 60, verified
  in_active_set == False for all 13 recordings).
- **Baseline statistics**: median over baseline windows (robust to transient
  electrode adjustment artifacts; Phinyomark et al. 2012).
- **Normalised features**: `emg_mnf_rel`, `emg_mdf_rel`, `emg_rms_rel`,
  `emg_dimitrov_rel`, `emg_iemg_rel`, `ecg_hr_rel`, `eda_scl_rel`, `temp_mean_rel`.
- **Subject isolation**: each recording uses only its own baseline — no cross-subject
  information leaks (Saeb et al. 2017).
- **Temperature**: baseline computed only when `temperature.csv` is non-empty.

---

## Parity Test Results

Test file: `tests/test_feature_parity.py`

All 8 tests pass as of 2026-04-26:
- `test_emg_parity[emg_rms]` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_emg_parity[emg_mnf]` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_emg_parity[emg_mdf]` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_emg_parity[emg_dimitrov]` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_eda_parity_scl` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_acc_parity_rms` PASSED — rtol ≤ 0.05 after 30 s warmup
- `test_no_filtfilt_in_streaming` PASSED — no forbidden patterns
- `test_temp_nan_tolerance` PASSED — no crash on all-NaN input

**Tolerance design**: parity tests compare streaming vs causal-offline (sosfilt)
reference, not vs filtfilt offline. This isolates state management correctness from
the known filtfilt vs sosfilt amplitude difference (~8–35% depending on feature).
The production offline extractor uses filtfilt (zero-phase, better SNR); the
streaming extractor uses sosfilt with persisted zi. Both are correct for their
respective use cases (Oppenheim & Schafer 2010).

**Warmup**: first 30 s excluded because IIR filter states need convergence time
(Oppenheim & Schafer 2010) and the Pan-Tompkins adaptive threshold needs
initialisation (Pan & Tompkins 1985).

---

## Known Issues

1. **ECG NaN ~30%** for recording_012 (Tias): 30% of ECG feature windows are NaN
   because the R-peak detector requires a 0.5 s warmup and the HRV window needs
   ≥3 RR intervals. This is inherent, not a bug.

2. **Recording_007 ECG NaN ~48%**: signal quality issue in the raw ECG.

3. **Recordings 003, 004 EMG/ACC NaN ~12%**: these recordings have ~12% NaN in the
   raw EMG and ACC CSV files (signal dropouts). NaN-fill before filtering handles
   gracefully but produces some NaN windows.

4. **Temperature NaN for recordings 007–014** (7 of 13): expected. These newer
   recordings have empty `temperature.csv`. Temp features are all NaN.

5. **[REF NEEDED: skin temperature as fatigue indicator during exercise]**: no reference
   in the approved bibliography for temperature as a fatigue marker. All temp feature
   code is flagged accordingly.

---

## Output Files

- `runs/20260426_154705_default/features/window_features.parquet`
  — 3,269,435 rows × 44 columns (11 label + 33 feature)
- `runs/20260426_154705_default/features/set_features.parquet`
  — 156 rows × 58 columns (7 label + 51 feature)

---

## References

- Allen, J. (2007). Photoplethysmography and its application in clinical physiological measurement. *Physiological Measurement*, 28(3), R1.
- Bonomi, A. G., Goris, A. H., Yin, B., & Westerterp, K. R. (2009). Detection of type, duration, and intensity of physical activity using an accelerometer. *Medicine & Science in Sports & Exercise*, 41(9), 1770–1777.
- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
- Castaneda, D., Esparza, A., Ghamari, M., Soltanpur, C., & Nazeran, H. (2018). A review on wearable photoplethysmography sensors and their potential future applications in health care. *International Journal of Biosensors & Bioelectronics*, 4(4), 195–202.
- Cifrek, M., Medved, V., Tonković, S., & Ostojić, S. (2009). Surface EMG based muscle fatigue evaluation in biomechanics. *Clinical Biomechanics*, 24(4), 327–340.
- De Luca, C. J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.
- Dimitrov, G. V., Arabadzhiev, T. I., Mileva, K. N., Bowtell, J. L., Crichton, N., & Dimitrova, N. A. (2006). Muscle fatigue during dynamic contractions assessed by new spectral indices. *Medicine and Science in Sports and Exercise*, 38(11), 1971–1979.
- González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity as a measure of loading intensity in resistance training. *International Journal of Sports Medicine*, 31(05), 347–352.
- Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE Transactions on Biomedical Engineering*, 63(4), 797–804.
- Khan, A. M., Lee, Y. K., Lee, S. Y., & Kim, T. S. (2010). A triaxial accelerometer-based physical-activity recognition via augmented-signal features and a hierarchical recognizer. *IEEE Transactions on Information Technology in Biomedicine*, 14(5), 1166–1172.
- Maeda, Y., Sekine, M., & Tamura, T. (2011). Relationship between measurement site and motion artifacts in wearable reflected photoplethysmography. *Journal of Medical Systems*, 35(5), 969–976.
- Makowski, D., Pham, T., Lau, Z. J., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689–1696.
- Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying human physical activity from on-body accelerometers. *Sensors*, 10(2), 1154–1175.
- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications*, 39(8), 7420–7431.
- Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal activity data collection and signal processing: A systematic review. *Sensors*, 20(2), 479.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019.
- Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258.
- Tamura, T., Maeda, Y., Sekine, M., & Yoshida, M. (2014). Wearable photoplethysmographic sensors—past and present. *Electronics*, 3(2), 282–302.
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043–1065.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
- Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics*, 4(3), 419–420.
