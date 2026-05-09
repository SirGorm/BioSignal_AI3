# features/

**Formål:** Offline feature-ekstraksjon (filtfilt-basert, full-session-tilgang). Produserer per-vindu og per-sett feature-tabeller fra labeled parquets. Causal/online-versjoner ligger i [src/streaming/](../streaming/FOLDER.md).

**Plass i pipeline:** Modus 1 (offline). Kjøres via `python -m src.features.window_features --run-dir runs/<slug>` etter labeling. Outputene mater både Random Forest-baseline og feature-veien til NN.

## Filer

### acc_features.py
- **Hva:** Acc-magnitude features. RMS, jerk RMS, dominant frekvens (Welch PSD), rep-band power 0.3–1.5 Hz pluss Phinyomark/Hudgins amplitude-deskriptorer (lscore, mfl, msr, wamp). Bandpass 0.5–20 Hz på `acc_mag = sqrt(ax²+ay²+az²)`.
- **Inn:** `ax, ay, az: np.ndarray`, `t: np.ndarray` (Unix), `fs=100`, `window_ms=2000`, `hop_ms=100`. `extract_acc_features()` ramme.
- **Ut:** DataFrame med kolonner `t_unix, acc_rms, acc_jerk_rms, acc_dom_freq, acc_rep_band_power, acc_rep_band_ratio, acc_lscore, acc_mfl, acc_msr, acc_wamp`.
- **Nøkkelfunksjoner:** [extract_acc_features()](src/features/acc_features.py#L201), [acc_mag_window_features()](src/features/acc_features.py#L128), [_filter_acc_offline()](src/features/acc_features.py#L98), [_filter_acc_causal()](src/features/acc_features.py#L111)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** `_filter_acc_offline` bruker `sosfiltfilt` (zero-phase) — bare offline-bruk. Per CLAUDE.md MÅ acc_mag beregnes FØR bandpass.

### ecg_features.py
- **Hva:** ECG HRV features (HR, RMSSD, SDNN, pNN50, mean RR) over 30 s vindu. R-peak via Pan & Tompkins (NeuroKit2 om tilgjengelig) + Kubios-artefakt-korreksjon → NN-intervaller.
- **Inn:** `ecg, t, fs=500`. Bandpass 0.5–40 Hz + 50 Hz notch.
- **Ut:** DataFrame `t_unix, ecg_hr, ecg_rmssd, ecg_sdnn, ecg_pnn50, ecg_mean_rr`.
- **Nøkkelfunksjoner:** [extract_ecg_features()](src/features/ecg_features.py#L249), [detect_r_peaks()](src/features/ecg_features.py#L102), [correct_to_nn()](src/features/ecg_features.py#L144), [ecg_hrv_features()](src/features/ecg_features.py#L212)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** **ECG ekskluderes fra modell-input** (CLAUDE.md "Feature-strategi"). Modulen er beholdt for diagnostikk; outputene plukkes ikke opp av `WindowFeatureDataset` fordi `EXCLUDED_FEATURE_PREFIXES = ('ecg_', 'eda_')` filtrerer dem bort. `extract_ecg_features` kalles dessuten ikke fra `window_features.py`.

### eda_features.py
- **Hva:** EDA features: SCL, SCR-amplitude, SCR-count, phasic-mean (10 s vindu, 0.1 s hop). LP 5 Hz (offline) eller causal-variant.
- **Inn:** `eda, t, fs=50`.
- **Ut:** DataFrame `t_unix, eda_scl, eda_scr_amp, eda_scr_count, eda_phasic_mean`.
- **Nøkkelfunksjoner:** [extract_eda_features()](src/features/eda_features.py#L120), [eda_window_features()](src/features/eda_features.py#L70), [_filter_eda_offline()](src/features/eda_features.py#L41), [_filter_eda_causal()](src/features/eda_features.py#L54)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** **EDA ekskluderes fra modell-input** (alle 9 opptak feiler dynamic-range-test, Greco et al. 2016). `extract_eda_features` kalles ikke fra `window_features.py`. Streaming-extraktoren beholdes for diagnostikk men outputene entrer ikke modellen.

### emg_features.py
- **Hva:** Spektrale fatigue-features (MNF, MDF, Dimitrov FInsm5) + amplitude (RMS, IEMG, lscore, mfl, msr, wamp) på native 2000 Hz. Per-recording baseline-normalisering (`EmgBaselineNormalizer`) + per-set lineær slope (`within_set_slope`) for offline-fatigue.
- **Inn:** `emg, t, fs=2000, window_ms=500, hop_ms=100, normalizer, baseline_end_unix`.
- **Ut:** DataFrame med kolonner `t_unix, emg_rms, emg_iemg, emg_mnf, emg_mdf, emg_dimitrov, emg_lscore, emg_mfl, emg_msr, emg_wamp` (+ `_rel`-baseline-normaliserte). Bandpass 20–450 Hz + 50 Hz notch.
- **Nøkkelfunksjoner:** [extract_emg_features()](src/features/emg_features.py#L271), [emg_window_features()](src/features/emg_features.py#L127), [EmgBaselineNormalizer](src/features/emg_features.py#L217), [within_set_slope()](src/features/emg_features.py#L332), [_filter_emg_offline()](src/features/emg_features.py#L93), [_filter_emg_causal()](src/features/emg_features.py#L108)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:**
  - Spektral-features krever native 2000 Hz — kan IKKE beregnes fra labeled parquet sin envelope (100 Hz, Nyquist = 50 Hz).
  - WAMP-terskel `5e-5` (50 µV) hardkodet for filtrert EMG.
  - `_filter_emg_offline` bruker `sosfiltfilt`/`filtfilt` (zero-phase) — kun offline.

### extra_features.py
- **Hva:** Pure-NumPy single-window features: WAMP, MFL, MSR, LS, LS4. Generic over signal type — kalles fra `emg_features.py` og `acc_features.py` med signal-spesifikke prefix og terskler.
- **Inn:** `x: np.ndarray` (window), `threshold` (for WAMP), `prefix` (kolonne-navn-prefiks).
- **Ut:** Dict `{f"{prefix}_wamp", f"{prefix}_mfl", f"{prefix}_msr", f"{prefix}_ls", f"{prefix}_ls4"}`.
- **Nøkkelfunksjoner:** [extras_window()](src/features/extra_features.py#L84), [baseline_threshold()](src/features/extra_features.py#L95), [wamp()](src/features/extra_features.py#L51), [mfl()](src/features/extra_features.py#L57), [msr()](src/features/extra_features.py#L66), [ls()](src/features/extra_features.py#L72), [ls4()](src/features/extra_features.py#L78)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** `baseline_threshold(k=0.1)` faller tilbake til `1e-6` ved degenerert baseline.

### ppg_features.py
- **Hva:** PPG-grønn HR + pulse-amplitude (10 s vindu, 0.1 s hop). Bandpass 0.5–8 Hz + `find_peaks` for IBI.
- **Inn:** `ppg, t, fs` (lest fra `metadata.json["sampling_rates"]["ppg"]`).
- **Ut:** DataFrame `t_unix, ppg_hr, ppg_pulse_amp, ppg_pulse_amp_var`.
- **Nøkkelfunksjoner:** [extract_ppg_features()](src/features/ppg_features.py#L131), [ppg_window_features()](src/features/ppg_features.py#L83), [detect_ppg_peaks()](src/features/ppg_features.py#L63), [_filter_ppg_offline()](src/features/ppg_features.py#L50)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Bare grønn wavelength brukes (CLAUDE.md). PPG-fs er per-recording — leses dynamisk fra `metadata.json`.

### temp_features.py
- **Hva:** Hudtemperatur features over 60 s vindu — mean, slope (deg/s), range. LP 0.1 Hz.
- **Inn:** `temp, t, fs=1, baseline_mean`.
- **Ut:** DataFrame `t_unix, temp_mean, temp_slope, temp_range, temp_mean_rel`.
- **Nøkkelfunksjoner:** [extract_temp_features()](src/features/temp_features.py#L97), [temp_window_features()](src/features/temp_features.py#L66), [_filter_temp_offline()](src/features/temp_features.py#L32)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Tom DataFrame returneres når `temperature.csv` er tom — håndteres oppstrøms i `window_features.py` ved å NaN-fylle kolonnene.

### window_features.py
- **Hva:** Driver som binder alt sammen. Per opptak: laster labeled parquet for label-kolonner og 100 Hz grid, leser native CSVer for høy-rate signaler (ECG/EMG/PPG), beregner per-modalitet features, aligner til 100 Hz hop-grid via `merge_asof(direction='backward')`, og produserer per-vindu (`window_features.parquet`) + per-sett (`set_features.parquet`).
- **Inn:** `--run-dir runs/<slug>`, `--labeled-dir data/labeled/`, `--dataset-dir dataset/`. Leser `data/labeled/<rec_id>/aligned_features.parquet`, `dataset/<rec_id>/<modality>.csv` (native rate), `dataset/<rec_id>/metadata.json` (PPG fs).
- **Ut:** `runs/<slug>/features/window_features.parquet` (én rad per 100 ms hop med alle kolonner i `LABEL_COLS` + features + soft targets `reps_in_window_2s`, `phase_frac_*`), `runs/<slug>/features/set_features.parquet` (per-set aggregat med `_mean`, `_std`, `_slope`, `_endset` for hver feature).
- **Nøkkelfunksjoner:** [process_recording()](src/features/window_features.py#L152), [_align_features_to_grid()](src/features/window_features.py#L104), [_add_soft_target_columns()](src/features/window_features.py#L301), [_build_set_features()](src/features/window_features.py#L341), [main()](src/features/window_features.py#L422)
- **Avhengigheter:** [src.data.loaders](src/data/loaders.py), [src.features.emg_features](src/features/emg_features.py), [src.features.temp_features](src/features/temp_features.py), [src.features.acc_features](src/features/acc_features.py), [src.features.ppg_features](src/features/ppg_features.py).
- **Gotchas:**
  - **ECG og EDA er hoppet over** i `process_recording()`: `# --- 1. ECG features: SKIPPED ---` og `# --- 3. EDA features: SKIPPED ---`. Ingen `ecg_*`/`eda_*` kolonner skrives.
  - Baseline-vinduet er første 60 s (`_get_baseline_end_unix`) — verifisert ikke å overlappe med aktiv set.
  - Soft-target kolonner (`reps_in_window_2s`, `phase_frac_<class>`) beregnes som backward 2 s rolling mean over per-sample labels og lagres i parquet, så `WindowFeatureDataset` kan plukke dem opp uten å re-aggregere.
  - `merge_asof(direction='backward', tolerance=35.0)` — opp til 35 s back-fill for å dekke 30 s ECG/PPG-vindu.

## Dataflyt inn/ut av mappen

- **Leser:**
  - `data/labeled/recording_NNN/aligned_features.parquet` (kolonner brukt: `t_unix`, `t_session_s`, `subject_id`, `recording_id`, `in_active_set`, `set_number`, `exercise`, `phase_label`, `rep_count_in_set`, `rep_density_hz`, `rpe_for_this_set`).
  - `dataset/recording_NNN/{ecg,emg,ppg_green}.csv` (native rate, kolonner `timestamp` + signal).
  - `dataset/recording_NNN/temperature.csv`, `dataset/recording_NNN/metadata.json` (`sampling_rates.ppg`).
  - `dataset/recording_NNN/{ax,ay,az}.csv` (via `load_imu`).
- **Skriver:**
  - `runs/<slug>/features/window_features.parquet` (kolonner: `LABEL_COLS` + `emg_*`, `acc_*`, `ppg_*`, `temp_*` features + `reps_in_window_2s` + `phase_frac_<class>`).
  - `runs/<slug>/features/set_features.parquet` (kolonner: `recording_id, subject_id, set_number, exercise, rpe_for_this_set, n_reps, set_duration_s` + per-feature `_mean/_std/_slope/_endset`).

## Relaterte mapper

- **Importerer fra:** [src/data/](../data/FOLDER.md) (`loaders`).
- **Importeres av:** [src/streaming/](../streaming/FOLDER.md) (causal-variantene gjenbruker konstanter), tester (`test_feature_parity.py`), `scripts/*.py` (RF-baselines, plotting). Outputene konsumeres av [src/data/datasets.py](src/data/datasets.py) (WindowFeatureDataset).
