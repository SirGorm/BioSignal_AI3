# labeling/

**Formål:** Offline preprosessering: tar `dataset_aligned/recording_NNN/` rådata + `Participants.xlsx` og produserer den autoritative `data/labeled/recording_NNN/aligned_features.parquet` på 100 Hz grid med per-sample labels (exercise, RPE, phase, rep_count, in_active_set, ...). Skriver også `quality_report.md`.

**Plass i pipeline:** Modus 1 (offline), kjøres én gang per opptak. Driver av `src/pipeline/label.py` (kalt fra `/label`-kommandoen). Brukes ALDRI i sanntid — joints og participants-data slippes ikke inn i streaming-veien.

## Filer

### align.py
- **Hva:** 100 Hz grid-bygger og resampling-utilities. Inkluderer EMG-envelope-pipeline (bandpass + notch + 50 ms RMS) på native 2000 Hz før decimering, samt `build_aligned_dataframe()` som setter sammen alle modaliteter + label-arrays til én parquet-DataFrame.
- **Inn:** Per-modalitet DataFrames fra `src.data.loaders`, `set_info`-DataFrame fra `build_set_info_array()`, valgfri `joint_angle_df` fra `joint_angles.py`. Fysisk grid genereres med `make_100hz_grid(t_start, t_end)`.
- **Ut:** DataFrame med kolonner `t_unix`, `t`, `t_session_s`, `ecg`, `emg` (envelope), `eda`, `ppg_green`, `ax`, `ay`, `az`, `acc_mag`, `temp`, `in_active_set`, `set_number`, `exercise`, `rpe_for_this_set`, `set_phase`, `rep_count_in_set`, `rep_index`, `rep_density_hz`, `primary_joint_angle_deg`, `phase_label`. Outside aktive sett er `phase_label='rest'` og `rep_*`/`exercise`/`rpe`/`set_number` settes til NaN.
- **Nøkkelfunksjoner:** [emg_envelope()](src/labeling/align.py#L155), [build_aligned_dataframe()](src/labeling/align.py#L218), [build_set_info_array()](src/labeling/align.py#L429), [make_100hz_grid()](src/labeling/align.py#L72), [_resample_linear()](src/labeling/align.py#L93), [_resample_ffill()](src/labeling/align.py#L102), [_resample_nearest()](src/labeling/align.py#L115)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:**
  - EMG-envelope bruker `sosfiltfilt`/`filtfilt` (zero-phase) — dette er OFFLINE-vei. Hooks blokkerer disse i `src/streaming/`.
  - 50 ms RMS-vindu fungerer både som amplitude-estimator (Konrad 2005) og anti-aliasing for 20:1 decimering (Oppenheim & Schafer 2010, Konrad 2005).
  - Sett med `n_reps == 0` markeres `in_active_set=False` (verifiseres i `build_set_info_array`) — totalt utelatt fra alle tasks.
  - Når Kinect-utledet rep-count ≠ marker-count, faller koden tilbake til marker-tider for `rep_density_hz` (count bevares, timing svekkes ±0.5–1 s).
  - Temperature returneres som NaN-kolonne hvis `temp_df` er tom/`None`.
  - `phase_label` settes alltid til `'rest'` utenfor aktive sett.

### joint_angles.py
- **Hva:** Beregner primær-leddvinkel (kne for squat, hofte for deadlift, albue for benk/pullup) fra Azure Kinect K4ABT skjelett-frames via cosinus-formel mellom bone-vektorer. Også per-fase klassifisering (concentric/eccentric/isometric/rest) fra angular velocity, og rep-counting fra peak/valley-deteksjon. Acc-basert phase- og rep-fallback for benkpress (Kinect ser ikke albuen) og når Kinect-rangen er for liten.
- **Inn:** Frame-lister fra `recording_NN_joints.json`, set-start/-end Unix-time, `exercise: str`. Filer: `rec_dir / "recording_<NN>_joints.json"` (med fallback-mønstre for rec_001 offset).
- **Ut:** DataFrame med `t_unix, primary_joint_angle_deg, joint_velocity_deg_s, joint_accel_deg_s2, phase_label, rep_count_in_set` (afhengig av om derivat/phase/rep-funksjonene kalles).
- **Nøkkelfunksjoner:** [extract_angles_from_frames()](src/labeling/joint_angles.py#L159), [compute_angle_derivatives()](src/labeling/joint_angles.py#L256), [label_phase()](src/labeling/joint_angles.py#L303), [label_phase_from_acc()](src/labeling/joint_angles.py#L683), [count_reps_from_angles()](src/labeling/joint_angles.py#L1113), [count_reps_from_acc()](src/labeling/joint_angles.py#L917), [load_joint_angles_for_set()](src/labeling/joint_angles.py#L1283), [_angle_from_triplet()](src/labeling/joint_angles.py#L126)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:**
  - K4ABT-skjelettet har 32 ledd; venstre + høyre vinkler midles for å håndtere delvis tracking-feil (Fukuchi et al. 2018).
  - 5 deg/s isometric-terskel (De Luca 1997) — under dette regnes leddet som "ikke beveger seg".
  - Frame-tid anchoring: prefererer å spre frames lineært over `[set_start_unix, set_end_unix]` (Azure Kinect effektiv fps er ~21, ikke nominelle 30, i dette datasettet). Fall tilbake til `t = start + frame_id / kinect_fps` bare når end mangler.
  - Benkpress må bruke acc-basert phase fordi Kinect ikke ser albuen under torsoen.

### markers.py
- **Hva:** Parser `markers.json` til `SetMarker`/`RepMarker`-objekter; filtrerer overflødige sett til "kanoniske" 12.
- **Inn:** `markers.json` (entries med `unix_time`, `time`, `label`, `color`). Labels: `Set:N_Start`, `Set_N_End`, `Set:N_Rep:M`, `Rest:K`.
- **Ut:** `list[SetMarker]` med `set_num`, `start_unix`, `end_unix`, `rep_markers: list[RepMarker]`. `select_canonical_sets()` returnerer den filtrerte listen (≤ `expected_n`).
- **Nøkkelfunksjoner:** [parse_markers()](src/labeling/markers.py#L92), [select_canonical_sets()](src/labeling/markers.py#L164), [SetMarker](src/labeling/markers.py#L53), [RepMarker](src/labeling/markers.py#L41)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:**
  - Subtil syntaks: `Set:N_Start` bruker kolon, `Set_N_End` bruker underscore — to forskjellige regex.
  - `select_canonical_sets()` har 3-stegs filter: først 0-rep-sett bort, så aborterte (få reps + kort varighet), så worst-of-N inntil targetcount. Trygg på alle 9 opptak per docstring.
  - `ValueError` hvis et sett har Start uten End eller omvendt.

### run.py
- **Hva:** Orkestrator for ett opptak: bygger labeled parquet + `quality_report.md` ende-til-ende. Inkluderer per-recording manuelle marker-eksklusjoner og post-labeling blacklists for trustverdige sets.
- **Inn:** `rec_dir`, parsed `participants_data` (fra `data.participants.load_participants`), `out_base = data/labeled/`. Leser CSVer, `markers.json`, `metadata.json`, `recording_NN_joints.json`.
- **Ut:** `data/labeled/<rec_id>/aligned_features.parquet` og `quality_report.md`. Returnerer dict `{recording_id, status, subject_id, n_sets, flags, error, active_minutes, eda_status}`.
- **Nøkkelfunksjoner:** [label_one_recording()](src/labeling/run.py#L366), [_check_eda_quality()](src/labeling/run.py#L183), [_align_to_canonical()](src/labeling/run.py#L146), [_build_session_joint_df()](src/labeling/run.py#L929), [_write_quality_report()](src/labeling/run.py#L207), [_write_partial_quality_report()](src/labeling/run.py#L944)
- **Avhengigheter:** [src.data.loaders](src/data/loaders.py), [src.data.participants](src/data/participants.py), [src.labeling.markers](src/labeling/markers.py), [src.labeling.joint_angles](src/labeling/joint_angles.py), [src.labeling.align](src/labeling/align.py).
- **Gotchas:**
  - EDA-kvalitetscheck: std < 1e-7 S eller range < 5e-8 S → `eda_status='unusable'` og EDA-kolonnen settes all-NaN. Greco et al. 2016 ref.
  - `_PHASE_REPS_BLACKLIST`-set angir kanoniske sett der phase/reps-supervisjon overstyres til NaN/unknown — andre tasks lever fortsatt med disse settene.
  - `_ALL_HEADS_BLACKLIST`-set angir fullstendig utelatte sett (`in_active_set=False`).
  - `_MANUAL_MARKER_EXCLUSIONS` fjerner orig marker-numre FØR `select_canonical_sets()` kjører.
  - Participants.xlsx slot N er indeksert med `sm.set_num - 1` (orig marker-nummer), ikke kanonisk posisjon — kritisk for `_align_to_canonical()` etter manuelle eksklusjoner.

## Dataflyt inn/ut av mappen

- **Leser:**
  - `dataset_aligned/recording_NNN/{ecg,emg,eda,temperature,ppg_green,ax,ay,az}.csv` (kolonner `timestamp` + signal).
  - `dataset_aligned/recording_NNN/markers.json` (entries `{unix_time, time, label, color}`).
  - `dataset_aligned/recording_NNN/metadata.json` (`recording_start_unix_time`, `total_kinect_sets`, `sampling_rates`).
  - `dataset_aligned/recording_NNN/recording_<NN>_joints.json` (Azure Kinect frames-array).
  - `dataset/Participants/Participants.xlsx` (via `src.data.participants`).
- **Skriver:**
  - `data/labeled/recording_NNN/aligned_features.parquet` (kolonner: alle biosignaler + label-arrays beskrevet over).
  - `data/labeled/recording_NNN/quality_report.md` (set-tellinger, rep-mismatch, modalitetsstatus, flagg).

## Relaterte mapper

- **Importerer fra:** [src/data/](../data/FOLDER.md) (`loaders`, `participants`)
- **Importeres av:** [src/pipeline/](../pipeline/FOLDER.md) (`label.py`).
