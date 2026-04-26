# Strength-RT: Multi-Task Fatigue/Exercise/Phase/Reps from Wearable Biosignals

Sanntidsestimering under styrketrening fra 6 biosignal-modaliteter. Fase og reps trenes med joint-angle ground truth, men deployes uten — modellen ser bare biosignaler i sanntid. Synkronisering mellom alle datakilder er via Unix epoch timestamps.

## Workflow (rekkefølge er viktig)

```
   /inspect <subj> <sess>     →  Bekreft fs, kanaler, sync på én session
        ↓ (oppdater CLAUDE.md med funn)
   /label --all               →  data-labeler kjører på alle sessions
        ↓
   /train <slug>              →  feature extraction + 4 LightGBM modeller (BASELINE)
        ↓
   /train-nn <slug>           →  4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN)
        ↓                        sammenlignes mot LightGBM på samme folds
   src/streaming/realtime.py  →  deploy beste causal arkitektur
```

**LightGBM kjøres ALLTID før neural networks.** Forskningsstandarden er å vise hva en sterk klassisk baseline oppnår, og at NN beats den med margin (eller ikke). Hopp over baseline = ikke-publiserbart.

## Hardware (oppdater fra `inspections/<subj>_<sess>/findings.md`)

| Modalitet | Kolonne | Sample rate | Plassering | Causal filter |
|-----------|---------|-------------|------------|---------------|
| ECG | `ecg` | _____ Hz | brystelektroder | 0.5–40 Hz BP, 50 Hz notch |
| EMG | `emg` | _____ Hz | _____ (muskel) | 20–450 Hz BP, 50 Hz notch |
| EDA | `eda` | _____ Hz | håndledd/finger | 0.05–5 Hz LP |
| Temp | `temp` | _____ Hz | hud | LP 0.1 Hz |
| Acc | `ax`, `ay`, `az` → `acc_mag` | _____ Hz | håndledd | 0.5–20 Hz BP **etter** magnitude |
| PPG-grønn | `ppg_green` (alias _____) | _____ Hz | håndledd | 0.5–8 Hz BP |

PPG-loggen har 4 wavelengths; vi bruker bare grønn. De andre lastes inn men ignoreres. Hvilken kolonne som faktisk er grønn bekreftes i `data-loader`-skillet etter `/inspect`.

## Synkronisering: Unix epoch er den autoritative klokken

Alle filer (biosignals, joint_angles, participants.xlsx) bruker Unix epoch timestamps i sekunder. Det betyr:

- **Ingen offset-estimering**, ingen sync-pulse, ingen DTW
- Konverter til session-relative tid ved å trekke fra `bio_start_unix` når plotting/visning er ønsket
- Joint-angle samples interpoleres direkte onto biosignal-tidsstempler via `align_joint_to_biosignal()` (i `data-loader` skill)
- `participants.xlsx` har `set_start_unix` og `set_end_unix` — direkte sammenlignbart med biosignal-timestampene
- Hvis en fil har timestamps som starter på 0 (session-relativ), STOPP og finn ut hvorfor — det er ikke project-protocol

Validering: alle loadere sjekker `t_unix > 1e9`. Det filtrerer post-2001-stempel og avviser session-relative tall.

## Sesjon-protokoll

```
[2 min hvile] [Set 1: ~8–10 reps] [RPE] [2 min hvile] [Set 2] [RPE] [2 min hvile] [Set 3] [RPE] [2 min hvile]
```

3 sett per øvelse, 8–10 reps per sett, 2 min hvile før/etter/mellom. RPE 1–10 rapportert etter hvert ferdige sett. Lange hvileperioder gjør acc-magnitude-segmentering trivielt.

## Labels og kilder

| Label | Kilde | Granularitet | Tilgjengelig sanntid? |
|-------|-------|--------------|----------------------|
| RPE/fatigue | `participants.xlsx` | Per sett | Predikeres |
| Exercise | `participants.xlsx` | Per sett | Predikeres |
| Phase | Joint-angle data | Per tidssteg | Nei — bare under trening |
| Reps | Joint-angle data | Per sett (count) | Predikeres fra biosignaler |

`participants.xlsx` kolonner: `subject_id, session_id, exercise, set_number, set_start_unix, set_end_unix, rpe`.

Joint-angle data: kun under aktive sett. Format: `t_unix` + leddvinkler (`knee_angle`, `hip_angle`, `elbow_angle`, ...). Interpoleres på biosignal-timestamps via Unix time.

## To-modus-arkitektur

**Modus 1: Offline labeling + trening**
1. `/inspect` på én session bekrefter setup
2. `data-labeler` kjører på alle sessions → `data/labeled/<subj>/<sess>/aligned_features.parquet`
3. `biosignal-feature-extractor` produserer features (offline + online parity)
4. `ml-expert` trener 4 modeller, latency-benchmark, model_card.md

**Modus 2: Real-time deployment**
1. Live biosignaler streames inn
2. Causal filtre + state-machines (set, rep, phase) + LightGBM-inferens
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
├── models/           # LightGBM per task
├── streaming/        # Online pipeline (bare biosignaler)
├── eval/             # Metrics, latency, per-subject reports
└── pipeline/
    ├── inspect.py    # Drives /inspect command
    ├── label.py      # Offline labeling
    ├── train.py      # Training
    └── realtime.py   # Real-time inference
inspections/<subj>_<sess>/
├── report.md
├── findings.md       # Action items for CLAUDE.md and configs
├── stats.json        # Machine-readable signal stats
└── *.png             # Time-series, PSDs, set detection, joint coverage
data/raw/<subj>/<sess>/
├── biosignals.<ext>  # Unix timestamps
├── joint_angles.csv  # Unix timestamps (only during sets)
└── participants.xlsx # set_start_unix, set_end_unix
data/labeled/<subj>/<sess>/
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
  - Importere fra `src/labeling/` eller åpne `participants.xlsx`
  - Referere til joint-angle-kolonner (`knee_angle`, `phase_label`, `rep_index`, `rpe_for_this_set`, ...)
  - Bruke `filtfilt`, `savgol_filter`, FFT-over-helt-signal, `find_peaks` over hele signal
- **Train/test-split**: alltid på `subject_id`
- **EMG-baseline**: per subject per session, fra første 60 s av session (verifisert som hvile av acc-magnitude)
- **RPE-modellering**: prediksjon per sett, features aggregert over hele settet
- **Inspeksjon kreves**: `data-labeler` refuser å starte hvis `inspections/`-mappen er tom
- **Sitering kreves**: hver metodologisk beslutning som dokumenteres (i `model_card.md`, `findings.md`, `quality_report.md`, og non-trivielle kode-kommentarer som forklarer en avgjørelse) MÅ siteres med `(Author Year)` inline + full entry i `## References`-seksjon. Bruk `literature-references` skillet som autoritativ kilde — IKKE finn på siteringer. Hvis en nødvendig referanse mangler, skriv `[REF NEEDED: <topic>]` og spør brukeren.

## Modellvalg

- **Fase 1**: LightGBM først for alt (rask, tolkbar, sterk på low-data)
- **Fase 2**: 4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN) som sammenligning — kjøres på SAMME folds som LightGBM
- State-machine før ML for reps og fase (acc-magnitude er nesten alltid nok for styrketrening)
- ML-fallback bare når state-machine F1 < 0.85 vs joint-angle ground truth
- Deployment: kun causal arkitekturer (TCN, unidirectional LSTM, causal-padded 1D-CNN). BiLSTM er kun for offline analyse.
- Med ~24 subjects × 9 sets = ~216 RPE-rader er du i low-data regime for fatigue; tung regularisering, NN slår ofte ikke LightGBM her. NN er mer sannsynlig vinner på per-window oppgaver (exercise, phase) hvor du har ~84k vinduer.

## Feature-strategi per modalitet

- **ECG**: HR, RMSSD over 30 s, RR-trend → fatigue (langsom)
- **EMG**: MNF, MDF, Dimitrov FInsm5, RMS, slopes innen sett → fatigue (rask, dominerende)
- **EDA**: SCL (tonisk), SCR amplitude/count (fasisk) → arousal/effort
- **Temp**: slope over session → fatigue (treghet)
- **Acc-magnitude**: motion intensity, dominant frekvens, jerk → exercise + reps
- **PPG-green**: HR, pulse amplitude → cross-validate ECG, effort

## Kommandoer

- `/inspect [subj] [sess]` — kjør først, alltid
- `/label --all` — offline labeling pipeline
- `/train <slug>` — feature extraction + 4 LightGBM modeller + benchmarks (BASELINE)
- `/train-nn <slug>` — 4 NN-arkitekturer sammenlignet mot LightGBM (krever /train først)
- Test alt: `pytest tests/ -x -v`
- Latency: `python -m eval.latency_benchmark`
- Lint: `ruff check . && ruff format --check .`

## Hva subagentene leverer

- **data-labeler**: `aligned_features.parquet` per session med kolonner `subject_id, session_id, t_unix, t_session_s, in_active_set, exercise, set_number, set_phase, joint_*, phase_label, rep_count_in_set, rep_index, rpe_for_this_set` + alle 6 modaliteter rå
- **biosignal-feature-extractor**: `window_features.parquet` + `set_features.parquet`, causal-versjon i `src/streaming/`, offline i `src/features/`, parity-test
- **ml-expert**: 4 trente LightGBM-modeller (BASELINE), LOSO-CV, per-subject metrikker, latency-benchmark, `model_card.md` med References-seksjon
- **dl-expert**: 4 NN-arkitekturer (1D-CNN, LSTM, CNN-LSTM, TCN), reuser samme folds som LightGBM, 3 seeds, sammenligningsrapport, deployment-anbefaling for sanntid
