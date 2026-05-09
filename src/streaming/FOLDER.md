# streaming/

**Formål:** Causal (online) feature-ekstraksjon. Per-modalitet streaming-extractors med vedvarende filterstate (`sosfilt + zi`), online HR/SCR-detektorer, og en pipeline-driver som spiller av et opptak sample-for-sample for replay-validering eller live deployment.

**Plass i pipeline:** Modus 2 (real-time inference). Mater den deployerte modellen i sanntid på et live biosignal-feed; hooksene blokkerer alt som ville ødelegge sanntidsegenskapene.

> **Streaming-restriksjoner gjelder.** Ingen joint-imports, ingen `Participants.xlsx` / `markers.json` / `metadata.json` / `*_joints.json`, ingen `filtfilt` / `savgol_filter` / full-signal FFT / full-signal `find_peaks`. Håndheves av `check-no-joint-in-streaming.sh` og `check-no-filtfilt.sh`.

## Filer

### acc_streaming.py
- **Hva:** Online acc-magnitude features (RMS, jerk RMS, dominant freq via Welch på vindu, rep-band ratio, lscore/mfl/msr/wamp). Bandpass 0.5–20 Hz på `acc_mag` etter magnitude-beregning.
- **Inn:** `step(ax_chunk, ay_chunk, az_chunk, t_arr)` — 100 Hz biter.
- **Ut:** Liste av feature-dicts per emit (matchende `acc_window_features`-skjema fra offline-veien).
- **Nøkkelfunksjoner:** [StreamingAccExtractor](src/streaming/acc_streaming.py#L113), [_acc_features_from_window()](src/streaming/acc_streaming.py#L64)
- **Avhengigheter:** [src.streaming.filters.CausalBandpass](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** Welch PSD beregnes per vindu (ikke over hele signalet) — ok per hook-regelen. WAMP-terskel = `0.05` (filtrert acc_mag).

### ecg_streaming.py
- **Hva:** Online R-peak-detektor (Pan & Tompkins-aktig) + NN-korreksjon + HRV-features (HR, RMSSD, SDNN, pNN50, mean RR). `OnlineRPeakDetector`/`OnlineNNCorrector` opprettholder rullende RR-historikk uten å se hele signalet.
- **Inn:** `step(ecg_chunk, t_arr)` ved 500 Hz.
- **Ut:** Feature-dicts ved 100 ms hop.
- **Nøkkelfunksjoner:** [StreamingECGExtractor](src/streaming/ecg_streaming.py#L162), [OnlineRPeakDetector](src/streaming/ecg_streaming.py#L49), [OnlineNNCorrector](src/streaming/ecg_streaming.py#L121)
- **Avhengigheter:** [src.streaming.filters](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** **ECG ekskluderes fra modell-input** (CLAUDE.md). Modulen er beholdt for diagnostikk; outputene entrer ikke modellen.

### eda_streaming.py
- **Hva:** Online EDA features (SCL, phasic via causal moving-median, SCR-amplitude, SCR-count). Causal LP 5 Hz; phasic = LP-signal − 2 s moving median.
- **Inn:** `step(eda_chunk, t_arr)` ved 50 Hz.
- **Ut:** Feature-dicts.
- **Nøkkelfunksjoner:** [StreamingEDAExtractor](src/streaming/eda_streaming.py#L42)
- **Avhengigheter:** [src.streaming.filters](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** **EDA ekskluderes fra modell-input** (sensor floor; Greco et al. 2016). Outputene entrer ikke modellen.

### emg_streaming.py
- **Hva:** Online EMG features (MNF, MDF, Dimitrov, RMS, lscore/mfl/msr/wamp) på 500 ms vindu @ 2000 Hz med causal bandpass 20–450 Hz + 50 Hz notch. Per-vindu Welch PSD for spektral-features.
- **Inn:** `step(emg_chunk, t_arr)` ved 2000 Hz.
- **Ut:** Feature-dicts ved 100 ms hop.
- **Nøkkelfunksjoner:** [StreamingEMGExtractor](src/streaming/emg_streaming.py#L85), [_emg_window_features_causal()](src/streaming/emg_streaming.py#L42)
- **Avhengigheter:** [src.streaming.filters](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** Native 2000 Hz kreves for spektral-features (MNF/MDF/Dimitrov) — IKKE den 100 Hz envelopen som ligger i parquet.

### filters.py
- **Hva:** Causal IIR-primitiver — Butterworth bandpass/lowpass og 50 Hz IIR-notch — med vedvarende `_zi` (filterminne) mellom kall. Warm-start ved første sample for å unngå transient.
- **Inn:** Konstruktør-args `(low_hz, high_hz, fs, order)`. `step(x)` tar en chunk.
- **Ut:** Filtrert chunk samme lengde.
- **Nøkkelfunksjoner:** [CausalBandpass](src/streaming/filters.py#L24), [CausalLowpass](src/streaming/filters.py#L70), [CausalNotch](src/streaming/filters.py#L93), [CausalFilterChain](src/streaming/filters.py#L119)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Bare `sosfilt`/`sosfilt_zi` brukes — `filtfilt` ville bryte hooken.

### online_stats.py
- **Hva:** Welford running mean + variance for streaming-normalisering. Brukes i live-pipelinen for z-score.
- **Inn:** `update(x)` med skalar/array. `z(x)` returnerer z-score med løpende stats.
- **Ut:** `n`, `mean`, `var`, `std`-properties.
- **Nøkkelfunksjoner:** [OnlineStats](src/streaming/online_stats.py#L19)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** NaN-prøver hoppes over (ingen oppdatering).

### ppg_streaming.py
- **Hva:** Online HR fra PPG-grønn med causal bandpass 0.5–8 Hz + per-prøve peak-detektor (`OnlinePPGPeakDetector`) som vedlikeholder lokal min/max-historikk uten å kalle `find_peaks` på hele signalet.
- **Inn:** `step(ppg_chunk, t_arr)` ved fs = 50 eller 100 Hz.
- **Ut:** Feature-dicts (HR, pulse_amp, pulse_amp_var).
- **Nøkkelfunksjoner:** [StreamingPPGExtractor](src/streaming/ppg_streaming.py#L92), [OnlinePPGPeakDetector](src/streaming/ppg_streaming.py#L46)
- **Avhengigheter:** [src.streaming.filters](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** Hovedkilde for HR siden ECG er ekskludert. PPG-fs varierer per recording (50 Hz på rec_001, 100 Hz på rec_012+).

### realtime.py
- **Hva:** Top-level `StreamingFeaturePipeline` som pakker alle 6 modalitetsekstraktorer + en `replay_session(rec_dir, chunk_size)`-driver som spiller av en opptaksmappe i 100-sample chunks (synced på 100 Hz acc-grid).
- **Inn:** `--replay <rec_dir>` (default chunk_size=100). Leser native CSVer + `metadata.json` for PPG-fs.
- **Ut:** DataFrame av streaming-features (replay-modus). I produksjon vil `step()` bli kalt med live-sensorbiter.
- **Nøkkelfunksjoner:** [StreamingFeaturePipeline](src/streaming/realtime.py#L54), [StreamingFeaturePipeline.step()](src/streaming/realtime.py#L94), [replay_session()](src/streaming/realtime.py#L175), [main()](src/streaming/realtime.py#L296)
- **Avhengigheter:** Alle `src.streaming.*`-extractors, [src.data.loaders](src/data/loaders.py) (`load_biosignal`, `load_temperature`, `load_imu`, `load_metadata`).
- **Gotchas:**
  - ⚠ **Mulig CLAUDE.md-friksjon:** `replay_session()` kaller `load_metadata(rec_dir)` for å lese `sampling_rates.ppg`. CLAUDE.md "Kritiske regler" forbyr streaming-koden å åpne `metadata.json` ([src/streaming/realtime.py:188](src/streaming/realtime.py#L188)). Dette er et replay-/utviklingsverktøy, ikke produksjons-deployment, men hooksene `check-no-joint-in-streaming.sh` skanner kun etter joint-relaterte ting og slipper denne gjennom. I live-deployment må PPG-fs være en konstant fra sensor-SDK, ikke leses fra fil.
  - ECG/EMG/EDA/Acc/PPG/Temp synces ved å bruke 100 Hz acc-griden som master; andre modaliteter samples i ratio-baserte underbiter.

### temp_streaming.py
- **Hva:** Online temperatur-features (mean, slope, range) med causal LP 0.1 Hz og online lineær regresjon for slope.
- **Inn:** `step(temp_chunk, t_arr)` ved 1 Hz.
- **Ut:** Feature-dicts.
- **Nøkkelfunksjoner:** [StreamingTempExtractor](src/streaming/temp_streaming.py#L35)
- **Avhengigheter:** [src.streaming.filters.CausalLowpass](src/streaming/filters.py), [src.streaming.window_buffer.SlidingWindowBuffer](src/streaming/window_buffer.py).
- **Gotchas:** NaN-tolerant — produserer NaN-features når sensoren mangler.

### window_buffer.py
- **Hva:** Causal `SlidingWindowBuffer` (deque-basert) med konfigurerbar `size_samples`/`hop_samples`. Emitterer komplette vinduer kun når både buffer-fyll og hop-betingelse er møtt.
- **Inn:** `push(x)` med ny chunk.
- **Ut:** Liste av numpy-vinduer (kan være tom).
- **Nøkkelfunksjoner:** [SlidingWindowBuffer](src/streaming/window_buffer.py#L20)
- **Avhengigheter:** Ingen interne `src.*`-imports.

## Dataflyt inn/ut av mappen

- **Leser:** I replay-modus: `dataset/<rec_id>/{ecg,emg,eda,temperature,ppg_green,ax,ay,az}.csv`, `dataset/<rec_id>/metadata.json` (PPG fs — se gotcha over).
- **Skriver:** Replay-driveren skriver ingen filer; returnerer en DataFrame in-memory. Live-bruk emiterer feature-dicts som ML-laget konsumerer direkte.

## Relaterte mapper

- **Importerer fra:** [src/data/](../data/FOLDER.md) (`loaders` — bare `realtime.py`).
- **Importeres av:** Tester som validerer streaming/offline parity ([src/features/](../features/FOLDER.md) deler kun konstanter, ikke kode).
