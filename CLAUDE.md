# Strength-RT: Multi-Task Fatigue/Exercise/Phase/Reps from Wearable Biosignals

Sanntidsestimering under styrketrening fra 6 biosignal-modaliteter. Fase og reps trenes med joint-angle ground truth, men deployes uten — modellen ser bare biosignaler i sanntid. Synkronisering mellom alle datakilder er via Unix epoch timestamps.

## Workflow (rekkefølge er viktig)

```
   build_dataset_aligned.py   →  Bygg dataset_aligned/ (kjøres én gang per ny rådata-batch)
        ↓
   /inspect <recording_id>    →  Bekreft fs, kanaler, sync på ett opptak
        ↓ (oppdater CLAUDE.md med funn)
   /label --all               →  data-labeler kjører på alle opptak i dataset_aligned/
        ↓
   /train <slug>              →  feature extraction + 4 Random Forest modeller (BASELINE)
        ↓
   /train-nn <slug>           →  4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN)
        ↓                        sammenlignes mot Random Forest på samme folds
   src/streaming/realtime.py  →  deploy beste causal arkitektur
```

**Random Forest kjøres ALLTID før neural networks.** Forskningsstandarden er å vise hva en sterk klassisk baseline oppnår, og at NN beats den med margin (eller ikke). Hopp over baseline = ikke-publiserbart.

## Hardware (bekreftet via inspections/recording_012/, recording_013/, recording_014_memory/)

| Modalitet | Fil | Kolonne | Sample rate | Plassering | Causal filter |
|-----------|-----|---------|-------------|------------|---------------|
| ECG | `ecg.csv` | `ecg` | **500 Hz** | brystelektroder | 0.5–40 Hz BP, 50 Hz notch |
| EMG | `emg.csv` | `emg` | **2000 Hz** | underarm/biceps (bekreft per deltaker) | 20–450 Hz BP, 50 Hz notch, **deretter RMS-envelope (50 ms) → 100 Hz i parquet** |
| EDA | `eda.csv` | `eda` | **50 Hz** | håndledd | 0.05–5 Hz LP |
| Temp | `temperature.csv` | `temperature` | **1 Hz** | hud | LP 0.1 Hz |
| Acc | `ax.csv`, `ay.csv`, `az.csv` → `acc_mag` | `ax`, `ay`, `az` | **100 Hz** | håndledd | 0.5–20 Hz BP **etter** magnitude |
| PPG-grønn | `ppg_green.csv` | `ppg_green` | **100 Hz** | håndledd | 0.5–8 Hz BP |

Hver modalitet er én CSV med kolonner `timestamp, <channel>` (Unix epoch i `timestamp`). PPG-loggen har 4 wavelengths (`ppg_blue`, `ppg_green`, `ppg_red`, `ppg_ir`) — vi bruker bare grønn (`ppg_green.csv`). De andre lastes inn men ignoreres for modellinput. Strømnett-frekvens i Norge = **50 Hz** → notch er obligatorisk på ECG og EMG.

**EMG har to løp i pipelinen** (se [src/labeling/align.py:emg_envelope](src/labeling/align.py)):
- **Rå-NN-vei**: native 2000 Hz → 20–450 Hz BP + 50 Hz notch → kvadrer → 50 ms moving-average → √ → lineær interpolasjon til 100 Hz griden. RMS-vinduet er anti-aliasing-LP for 20:1 decimering. Det er denne envelope-en som ligger i `aligned_features.parquet` kolonnen `emg`, og som alle raw-modeller (1D-CNN, LSTM, CNN-LSTM, TCN) leser fra.
- **Feature-vei**: native 2000 Hz CSV lastes på nytt av [src/features/emg_features.py](src/features/emg_features.py) for å beregne MNF, MDF, Dimitrov FInsm5 (krever full spektral båndbredde, kan ikke beregnes fra envelope). 500 ms vindu, 100 ms hop, baseline-norm via median av første 60 s.

`temperature.csv` i `dataset_aligned/` er hentet fra memory-varianten med offset-korrigering (se "Datakilde og synkronisering"). Dataset-variantens `temperature.csv` er tom/manglende i alle 9 opptak.

## Datakilde og synkronisering

**Autoritativ datakilde for trening:** `dataset_aligned/recording_NNN/` (én mappe per opptak).

Originale råopptak finnes i to varianter under `C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/`:
- `dataset/recording_NNN/` — biosignaler streamet til PC + markers + Kinect skeleton (JSON) + metadata. **Hovedkilde for alle modaliteter unntatt temperatur** (temperature.csv er tom eller mangler i alle 9 opptak under dataset/).
- `dataset_memory/recording_NNN_memory/` — biosignaler logget på sensorens onboard-minne. Brukes BARE for `temperature.csv` (alle 9 memory-opptak har populert temp). Sensor-klokken er off med en konstant per-opptak-offset (typisk 2200–5400 s) ift. PC-klokken — verifisert via cross-correlation av acc-magnitude.

`dataset_aligned/` er bygget av [scripts/build_dataset_aligned.py](scripts/build_dataset_aligned.py) ved å:
1. **Kopiere biosignal-CSVer (alle utenom temperature) verbatim fra `dataset/`** — disse er allerede på PC-klokken
2. **Hente `temperature.csv` fra `dataset_memory/`**, subtrahere per-opptak offset (memory_clock − offset = dataset_clock), og trimme til dataset's tidsrekke
3. Kopiere `markers.json`, `metadata.json`, `recording_NN_joints.json` uendret fra `dataset/` (allerede på PC-klokken)

Se [dataset_aligned/alignment_offsets.json](dataset_aligned/alignment_offsets.json) for offset-tabellen og Pearson-konfidens per opptak (offset brukes kun for temperature-shift; biosignaler kopieres verbatim). Originalmapper røres ALDRI.

**Unix epoch er den autoritative klokken** (etter alignment). Alle filer i `dataset_aligned/` bruker Unix epoch sekunder:
- Biosignal-CSVer: kolonne `timestamp` (Unix epoch)
- `markers.json`: hver entry har `unix_time` (Unix) og `time` (session-relativ sekunder fra `metadata.json["data_start_unix_time"]`)
- `metadata.json["kinect_sets"][i]`: `start_unix_time` og `end_unix_time`
- `recording_NN_joints.json`: per-frame `timestamp_usec = 0` (Kinect SDK fyller ikke dette inn) — synkroniseres ved å spre `metadata.kinect_sets[N-1].start_unix_time → end_unix_time` lineært over `frames`-arrayet
- `Participants.xlsx`: dual-row schema (se "Labels og kilder")

Validering: alle loadere sjekker `t_unix > 1e9`. Det filtrerer post-2001-stempel og avviser session-relative tall. `data-loader`-skillet håndterer dette per fil.

## Sesjon-protokoll

Hver opptaks-session = **12 sett over 4 øvelser** (3 sett per øvelse). Øvelser pr. session: `squat, deadlift, benchpress, pullup` (rekkefølge varierer per deltaker — se `Participants.xlsx`).

```
[2 min baseline-hvile, ~120 s før første sett]
[Set 1: ~6–10 reps] [RPE] [hvile] [Set 2] [RPE] [hvile] ... [Set 12] [RPE]
```

RPE 1–10 rapportert etter hvert ferdige sett. Hvile-perioder er lange nok til at acc-magnitude-segmentering er triviell. **Første ~120 s av hvert opptak er hvile-baseline** — bruk denne perioden til per-subject EMG/EDA-baseline (se "Kritiske regler"). Det er ingen pre-roll utenfor dette i `dataset_aligned/`.

## Labels og kilder

| Label | Kilde | Granularitet | Tilgjengelig sanntid? |
|-------|-------|--------------|----------------------|
| RPE/fatigue | `Participants.xlsx` | Per sett | Predikeres |
| Exercise | `Participants.xlsx` | Per sett | Predikeres |
| Set start/end | `metadata.json["kinect_sets"]` | Per sett | Predikeres (state-machine på acc-mag) |
| Per-rep tidspunkt | `markers.json` | Per rep | Predikeres |
| Phase (concentric/eccentric/iso) | `recording_NN_joints.json` (Kinect skeleton) → utledet leddvinkel | Per tidssteg | Nei — bare under trening |
| Rep count i sett | `markers.json` | Per sett (count) | Predikeres fra biosignaler |

**`Participants.xlsx`** (under `dataset/Participants/Participants.xlsx`) har dual-row schema:
```
Recording: 14 | Name: Raghild | set1=pullup | set2=pullup | ... | set12=benchpress
              | Name: fatigue | set1=7      | set2=4      | ... | set12=9
```
Hver `Recording: N`-rad er øvelses-rad (12 kolonner `set1..set12`); raden under (uten Recording-nummer, `Name: "fatigue"`) er RPE per sett. `data-loader` parser dette til en flat tabell med kolonner `recording_id, set_number, exercise, rpe`.

**`markers.json`** har entries med `{unix_time, time, label, color}`. `label` er en av:
- `Set:N_Start` (start på sett N)
- `Set:N_Rep:K` (rep K innenfor sett N)
- `Set_N_End` (slutt på sett N)

**`metadata.json`** har `kinect_sets`-listen: `[{set_number, start_unix_time, end_unix_time, ...}]`. Disse Unix-tidene er offisiell ground-truth for set-grenser.

**`recording_NN_joints.json`** (én fil per kinect_set, NN matcher `set_number`) inneholder:
- `bone_list` (31 bones) og `joint_names` (32 joints; Azure Kinect K4ABT-skjelett)
- `frames`: liste av `{frame_id, timestamp_usec, num_bodies, bodies}` hvor `bodies[i]` har `joint_positions` (32×3) og `joint_orientations` (32×4 quaternion)
- `timestamp_usec` er **alltid 0** (Kinect SDK fyller ikke dette) → synkroniser ved å lineært spre `kinect_sets[N-1].start_unix_time → end_unix_time` over `frames`-arrayet

Leddvinkler (`knee_angle`, `hip_angle`, `elbow_angle`, ...) er IKKE forhåndsberegnet — `data-labeler` må regne dem ut fra `joint_positions` (cosinus-formel mellom bone-vektorer). Vinkler interpoleres deretter på biosignal-timestamps via Unix time.

## To-modus-arkitektur

**Modus 1: Offline labeling + trening**
1. `/inspect <recording_id>` på ett opptak bekrefter setup
2. `data-labeler` kjører på alle opptak i `dataset_aligned/` → `data/labeled/recording_NNN/aligned_features.parquet`
3. `biosignal-feature-extractor` produserer features (offline + online parity)
4. `ml-expert` trener 4 modeller, latency-benchmark, model_card.md

**Modus 2: Real-time deployment**
1. Live biosignaler streames inn
2. Causal filtre + state-machines (set, rep, phase) + Random Forest-inferens
3. Joint data brukes IKKE — modellen står på biosignaler alene

Hooks håndhever skillet:
- `check-no-joint-in-streaming.sh` AST-skanner `src/streaming/` etter joint-referanser
- `check-no-filtfilt.sh` blokkerer non-causale operasjoner i streaming
- `verify-labeled-data.sh` validerer schema etter labeling

## Repo-struktur

```
src/
├── data/             # Loaders (Unix-time-aware)
├── labeling/         # Offline label generation
├── features/         # Causal + offline feature extractors
├── models/           # Random Forest per task
├── streaming/        # Online pipeline (bare biosignaler)
├── eval/             # Metrics, latency, per-subject reports
└── pipeline/
    ├── inspect.py    # Drives /inspect command
    ├── label.py      # Offline labeling
    ├── train.py      # Training
    └── realtime.py   # Real-time inference
inspections/recording_NNN/
├── report.md
├── findings.md       # Action items for CLAUDE.md and configs
├── stats.json        # Machine-readable signal stats
└── *.png             # Time-series, PSDs, set detection, joint coverage
dataset_aligned/                    # AUTORITATIV trenings-input (bygget fra dataset_memory + dataset)
├── alignment_offsets.json          # per-opptak offset, Pearson, kilde-paths
└── recording_NNN/
    ├── ecg.csv, emg.csv, eda.csv, temperature.csv
    ├── ax.csv, ay.csv, az.csv
    ├── ppg_blue.csv, ppg_green.csv, ppg_red.csv, ppg_ir.csv
    ├── markers.json                # per-rep + set-start/end (Unix)
    ├── metadata.json               # kinect_sets med Unix start/end
    ├── recording_NN_joints.json    # 1 per sett — Kinect skeleton frames
    └── clock_alignment.json        # offset, Pearson, kildepaths for dette opptaket
data/labeled/recording_NNN/
└── aligned_features.parquet
runs/<timestamp>_<slug>/
├── model_card.md
├── metrics.json
├── models/, plots/, logs/
└── ...
configs/, tests/
```

## Kritiske regler (håndheves av hooks)

- **Streaming-koden** har ALDRI lov til å:
  - Importere fra `src/labeling/` eller åpne `Participants.xlsx`, `markers.json`, `metadata.json`, `*_joints.json`
  - Referere til joint-utledede kolonner (`knee_angle`, `hip_angle`, `phase_label`, `rep_index`, `rpe_for_this_set`, ...)
  - Bruke `filtfilt`, `savgol_filter`, FFT-over-helt-signal, `find_peaks` over hele signal
- **Train/test-split**: alltid på `recording_id` (≈ subject — én deltaker per recording-nummer per `Participants.xlsx`)
- **EMG-baseline**: per recording, fra første 90–120 s av opptaket (det er ~120 s baseline før første sett — verifisert som hvile via acc-magnitude < terskel)
- **RPE-modellering**: prediksjon per sett, features aggregert over hele settet
- **Inspeksjon kreves**: `data-labeler` refuser å starte hvis `inspections/`-mappen er tom
- **Originaldata røres ALDRI**: `dataset/`, `dataset_memory/`, `Participants.xlsx` er read-only. All preprosessering skriver til `dataset_aligned/` eller `data/labeled/`.
- **Sitering kreves**: hver metodologisk beslutning som dokumenteres (i `model_card.md`, `findings.md`, `quality_report.md`, og non-trivielle kode-kommentarer som forklarer en avgjørelse) MÅ siteres med `(Author Year)` inline + full entry i `## References`-seksjon. Bruk `literature-references` skillet som autoritativ kilde — IKKE finn på siteringer. Hvis en nødvendig referanse mangler, skriv `[REF NEEDED: <topic>]` og spør brukeren.

## Modellvalg

- **Fase 1**: Random Forest først for alt (rask, tolkbar, sterk på low-data)
- **Fase 2**: 4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN) som sammenligning — kjøres på SAMME folds som Random Forest
- State-machine før ML for reps og fase (acc-magnitude er nesten alltid nok for styrketrening)
- ML-fallback bare når state-machine F1 < 0.85 vs joint-angle ground truth
- Deployment: kun causal arkitekturer (TCN, unidirectional LSTM, causal-padded 1D-CNN). BiLSTM er kun for offline analyse.
- Datasettet har **9 opptak (recording_006..014) × 12 sett = 108 RPE-rader** (etter Participants.xlsx og dataset_aligned). Dette er low-data regime for fatigue; tung regularisering, NN slår ofte ikke Random Forest her. NN er mer sannsynlig vinner på per-window oppgaver (exercise, phase) hvor du har ~30–40k vinduer per opptak.
- Subject-skille: ett `recording_NNN` ≈ én deltaker (sjekk `Participants.xlsx["Name:"]` for sann subject-id; samme deltaker kan dukke opp på flere opptak — gruppér på navn ikke nummer for LOSO-CV).

## Feature-strategi per modalitet

**Modell-input består av 4 modaliteter: EMG, Acc, PPG-green, Temp.** ECG og EDA er ekskludert (se under). Eksklusjonen håndheves av `EXCLUDED_FEATURE_PREFIXES = ('ecg_', 'eda_')` i [src/data/datasets.py](src/data/datasets.py) (features-vei) og av `RAW_CHANNELS` i [src/data/raw_window_dataset.py](src/data/raw_window_dataset.py) (rå-vei). Streaming-extractorene for ECG/EDA er beholdt for diagnostikk, men outputene deres entrer ikke modellen.

- **ECG**: ❌ ekskludert — signalkvalitet er utilstrekkelig på dette datasettet (verifisert via [scripts/compare_ecg_filtering.py](scripts/compare_ecg_filtering.py); QRS-morfologi er ustabil selv etter NeuroKit2-rensing)
- **EMG**: feature-vei: MNF, MDF, Dimitrov FInsm5, RMS, slopes innen sett → fatigue (rask, dominerende). Rå-NN-vei: 50 ms RMS-envelope @ 100 Hz (amplitude only — spektrale fatigue-features er ikke tilgjengelig på 100 Hz pga Nyquist; bruk feature-veien for fatigue).
- **EDA**: ❌ ekskludert — alle 9 opptak feiler dynamic-range-terskel (std < 1e-7 S, sensor-floor; Greco et al. 2016)
- **Temp**: slope over session → fatigue (treghet)
- **Acc-magnitude**: motion intensity, dominant frekvens, jerk → exercise + reps
- **PPG-green**: HR, pulse amplitude → primær HR-kilde (siden ECG er ute), effort

## Kommandoer

- `/inspect [recording_id]` — kjør først, alltid (f.eks. `/inspect 014`)
- `/label --all` — offline labeling pipeline
- `/train <slug>` — feature extraction + 4 Random Forest modeller + benchmarks (BASELINE)
- `/train-nn <slug>` — 4 NN-arkitekturer sammenlignet mot Random Forest (krever /train først)
- Test alt: `pytest tests/ -x -v`
- Latency: `python -m eval.latency_benchmark`
- Lint: `ruff check . && ruff format --check .`

## Hva subagentene leverer

- **data-labeler**: `aligned_features.parquet` per opptak med kolonner `recording_id, subject_name, t_unix, t_session_s, in_active_set, exercise, set_number, set_phase, joint_*, phase_label, rep_count_in_set, rep_index, rpe_for_this_set, rep_density_hz` + biosignaler resamplet til 100 Hz grid (ECG/EDA/PPG/IMU lineær interpolasjon, temp forward-fill, **EMG som RMS-envelope** — se hardware-tabellen)
- **biosignal-feature-extractor**: `window_features.parquet` + `set_features.parquet`, causal-versjon i `src/streaming/`, offline i `src/features/`, parity-test
- **ml-expert**: 4 trente Random Forest-modeller (BASELINE), LOSO-CV, per-subject metrikker, latency-benchmark, `model_card.md` med References-seksjon
- **dl-expert**: 4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN), reuser samme folds som Random Forest, 3 seeds, sammenligningsrapport, deployment-anbefaling for sanntid
