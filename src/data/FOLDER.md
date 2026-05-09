# data/

**Formål:** Data-laster-laget for prosjektet. Inneholder lavnivå biosignal-CSV-loadere (Unix-time-validerende), `Participants.xlsx`-parser og PyTorch-datasettene som mater inn både feature-veien og rå-veien til NN-treningen.

**Plass i pipeline:** Brukes i alle stadier — labeling, feature-ekstraksjon, trening og evaluering. Datasett-klassene leses bare i Modus 1 (offline trening); rå biosignal-loadere brukes både offline og av streaming-modulen for batch-replays under testing.

## Filer

### loaders.py
- **Hva:** Robuste CSV-lastere som støtter to historiske timestamp-formater: gammelt (`Time (s)` med `Recording Start Unix Time:` i header) og nytt (`timestamp` direkte som Unix epoch). Returnerer alltid Unix-time DataFrames.
- **Inn:** `rec_dir: Path` (én opptaksmappe under `dataset_aligned/`). Filnavn pr. modalitet: `ecg.csv`, `emg.csv`, `eda.csv`, `temperature.csv`, `ppg_green.csv`, `ax.csv`, `ay.csv`, `az.csv`, `metadata.json`.
- **Ut:** `pd.DataFrame` med kolonner `['timestamp', <signal>]`. `load_imu()` returnerer i tillegg `acc_mag = sqrt(ax²+ay²+az²)`. `load_all_biosignals()` returnerer dict per modalitet pluss `'imu'`. `load_metadata()` returnerer dict.
- **Nøkkelfunksjoner:** [_read_csv_auto()](src/data/loaders.py#L36), [load_biosignal()](src/data/loaders.py#L74), [load_temperature()](src/data/loaders.py#L113), [load_imu()](src/data/loaders.py#L147), [load_metadata()](src/data/loaders.py#L167), [load_all_biosignals()](src/data/loaders.py#L176)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** `load_biosignal()` reiser `ValueError` ved `timestamp <= 1e9` (CLAUDE.md sin Unix-validering). `load_temperature()` returnerer en tom DataFrame når filen er header-only — kallere må sjekke `len(df) == 0` og fylle med NaN. `load_imu()` antar at `ax/ay/az` har sammenfallende timestamps.

### participants.py
- **Hva:** Parser den dual-row-Excel-protokollen i `Participants.xlsx` (én øvelse-rad + én fatigue-rad pr. opptak).
- **Inn:** `xlsx_path` (vanligvis `dataset/Participants/Participants.xlsx`).
- **Ut:** `dict[int, dict]` med nøkler `subject_id`, `exercises` (12 strenger eller None), `rpe` (12 ints eller None). Opptak uten Name-kolonne hoppes over. `get_recording_info()` slår opp ett opptak.
- **Nøkkelfunksjoner:** [load_participants()](src/data/participants.py#L31), [get_recording_info()](src/data/participants.py#L100)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Subject-navn beholdes verbatim — kallere bruker `subject_id` som kanonisk subject-nøkkel for LOSO-CV (samme person på flere opptak grupperes på navn, ikke nummer). Recordings 15+ er tomme i regnearket og hoppes over.

### datasets.py
- **Hva:** `WindowFeatureDataset` (per-vindu engineered features) og `LabelEncoder` (deterministisk str→int) for feature-veien til NN. Per-recording z-score med valgbar `norm_mode`. Soft target-modi for reps og phase. Valgfri GPU-resident materialisering for `_GPUBatchIterator` i treningsløkka.
- **Inn:** Liste av `window_features.parquet`-stier. Forventede kolonner: `recording_id, subject_id, set_number, t_session_s, in_active_set, exercise, phase_label, rep_count_in_set, rpe_for_this_set` (eller `rpe`); soft-mode krever `reps_in_window_2s`, `phase_frac_*`, evt. `soft_overlap_reps_<window>s`. Valgfritt `phase_whitelist`-set fra `phase_whitelist.py`.
- **Ut:** `__getitem__` → `{'x': float32-tensor, 'targets': {exercise/phase/fatigue/reps}, 'masks': {...}}`. `materialize_to_device(device)` flytter alt til CUDA og setter `gpu_resident=True` slik at `_GPUBatchIterator` slipper DataLoader.
- **Nøkkelfunksjoner:** [LabelEncoder](src/data/datasets.py#L62), [WindowFeatureDataset](src/data/datasets.py#L90), [WindowFeatureDataset.__init__()](src/data/datasets.py#L99), [materialize_to_device()](src/data/datasets.py#L375)
- **Avhengigheter:** [src.data.phase_whitelist.whitelist_mask](src/data/phase_whitelist.py).
- **Gotchas:**
  - `EXCLUDED_FEATURE_PREFIXES = ('ecg_', 'eda_')` — ECG og EDA droppes på vei inn (CLAUDE.md "Feature-strategi"). Ekskluderingen håndheves selv om `feature_cols` eksplisitt sender inn `ecg_*`/`eda_*`-kolonner.
  - `norm_mode ∈ {'baseline','robust','percentile'}` — fallback til full-recording stats når mindre enn 100 baseline-rader er gyldige.
  - Outliers klippes til ±8σ; NaN → 0 på input.
  - Reps-mode `soft_window`/`soft_overlap` krever spesifikke kolonner — kasterer tydelig feilmelding hvis manglende.
  - Mask for fatigue og reps krever `in_active_set=True` — rest-vinduer får forward-pass men null gradient. NaN i `rep_count_in_set` brukes som "blacklist"-signal fra labeling.
  - `stride` defaulter til `window_s/2 × 100` Hz for 50 % overlap matching rå-veien.

### phase_whitelist.py
- **Hva:** CSV-basert whitelist for `(recording_id, set_number)`-par som har trustverdige Kinect-utledede phase-labels. Når satt brukes andre tasks (exercise/fatigue/reps) på alle data, men phase-loss blir kun talt for whitelistede sett.
- **Inn:** CSV-fil med header `recording_id,set_number` (kommentarlinjer `#` tillatt). Kallere kan også sende inn arrays direkte til `whitelist_mask()`.
- **Ut:** `Set[Tuple[str, int]]` (eller `None` når path er None). `whitelist_mask()` returnerer numpy bool-array pr. rad.
- **Nøkkelfunksjoner:** [load_phase_whitelist()](src/data/phase_whitelist.py#L35), [whitelist_mask()](src/data/phase_whitelist.py#L68)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Empty whitelist (header-only) ≠ `None` — empty betyr "ingen sets inkludert" så phase-head trener ikke; `None` betyr "ingen filtrering". `set_number` normaliseres alltid til int.

### raw_window_dataset.py
- **Hva:** `RawMultimodalWindowDataset` — per-vindu rå biosignaler (4-kanals input) til NN-Phase 2. Lazy `__getitem__` leser fra cached DataFrames; valgfri full materialisering til GPU.
- **Inn:** Liste av `aligned_features.parquet`-stier (per opptak, 100 Hz grid). Forventede kolonner: alle av `RAW_CHANNELS = ['emg', 'ppg_green', 'acc_mag', 'temp']` pluss `recording_id, subject_id, set_number, exercise, phase_label, rpe_for_this_set, rep_count_in_set, in_active_set, has_rep_intervals`; soft-mode krever `rep_density_hz`, `soft_overlap_reps_*`.
- **Ut:** `__getitem__` → `{'x': (C=4,T=200), 'targets': {...}, 'masks': {...}}`. End-of-window labels for klassifisering; per-vindu sannsynlighetsvektor for `target_modes['phase']='soft'`.
- **Nøkkelfunksjoner:** [RawMultimodalWindowDataset](src/data/raw_window_dataset.py#L84), [_compute_baseline_stats()](src/data/raw_window_dataset.py#L244), [materialize_to_device()](src/data/raw_window_dataset.py#L323), [__getitem__()](src/data/raw_window_dataset.py#L363)
- **Avhengigheter:** [src.data.datasets.LabelEncoder](src/data/datasets.py), [src.data.phase_whitelist.WhitelistKey](src/data/phase_whitelist.py).
- **Gotchas:**
  - `RAW_CHANNELS = ['emg', 'ppg_green', 'acc_mag', 'temp']` — EDA og ECG ekskludert (Greco et al. 2016 sensor-floor for EDA, ustabil QRS for ECG; se docstring).
  - Default vindu = 2 s @ 100 Hz; hop = window_s/2. `BASELINE_SAMPLES=9000` (90 s).
  - 3 norm-modi delt med feature-veien (`baseline`/`robust`/`percentile`); MAD × 1.4826 for robust.
  - Phase soft-target ekskluderes når > 50 % av vinduet er `'unknown'` (`PHASE_UNKNOWN_MAX_FRAC`).
  - Kommentaren i koden sier (C=6, T=200), men implementasjonen er C=4 etter ECG/EDA-ekskluderingen.
  - Signal-stempelet på dokumentert `(C, T)` er `(C=4, T=200)` med default `window_s=2.0`.

## Dataflyt inn/ut av mappen

- **Leser:**
  - `dataset_aligned/recording_NNN/{ecg,emg,eda,temperature,ppg_green,ax,ay,az}.csv` (kolonner `timestamp` + signal).
  - `dataset_aligned/recording_NNN/metadata.json`.
  - `dataset/Participants/Participants.xlsx` (alle ark som dual-row schema).
  - `data/labeled/recording_NNN/window_features.parquet` (kolonner: `recording_id`, `subject_id`, `set_number`, `t_session_s`, `t_unix`, `in_active_set`, `exercise`, `phase_label`, `rep_count_in_set`, `rpe_for_this_set`, `reps_in_window_2s`, `phase_frac_*`, `soft_overlap_reps_*`, `has_rep_intervals` + alle feature-kolonner).
  - `data/labeled/recording_NNN/aligned_features.parquet` (kolonner: alle i `RAW_CHANNELS` + label-kolonner over).
  - Phase-whitelist CSVer med `recording_id, set_number`.
- **Skriver:** Ingen filer. Kun returnerte tensorer/dicts.

## Relaterte mapper

- **Importeres av:** [src/labeling/](../labeling/FOLDER.md), [src/features/](../features/FOLDER.md), [src/streaming/](../streaming/FOLDER.md) (`realtime.py`), [src/eval/](../eval/FOLDER.md) (`latency_benchmark.py`), [src/pipeline/](../pipeline/FOLDER.md) (datasett brukes av nesten alle treningsskript).
