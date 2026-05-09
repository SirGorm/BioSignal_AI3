# models/raw/

**Formål:** Multi-task PyTorch-arkitekturer for rå-signal-veien (Phase 2). Tar `(B, C, T)` rå biosignal-vinduer fra `RawMultimodalWindowDataset` (default `C=4` etter ECG/EDA-eksklusjon, `T=200` = 2 s @ 100 Hz) og forutsier `{exercise, phase, fatigue, reps}`.

**Plass i pipeline:** Modus 1 (offline trening, etter at LightGBM-baseline er etablert). Causal-arkitekturer (TCN, 1D-CNN) er deploymentskandidatene; BiLSTM-baserte er `research_only`.

## Filer

### cnn1d_raw.py
- **Hva:** Causal 1D-CNN. `_CausalConvBlock` left-padder med `kernel_size-1` slik at output i tid `t` kun avhenger av input ≤ `t`. AdaptiveAvgPool over hele tidsaksen + lineær projeksjon.
- **Inn:** `(B, C, T)` float-tensor. Default `n_channels=5, n_timesteps=200`.
- **Ut:** Dict fra `MultiTaskHeads` (`exercise/phase/fatigue/reps`).
- **Nøkkelfunksjoner:** [CNN1DRawMultiTask](src/models/raw/cnn1d_raw.py#L55), [_CausalConvBlock](src/models/raw/cnn1d_raw.py#L35)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** Default-konstruktør har `n_channels=5` i signaturen mens [src/data/raw_window_dataset.py:RAW_CHANNELS](src/data/raw_window_dataset.py#L48) er 4 — kallere må sende `n_channels=dataset.n_channels`.

### cnn_lstm_raw.py
- **Hva:** DeepConvLSTM-stil hybrid (Ordóñez & Roggen 2016). Causal conv front-end (k=5 og k=3 med eksplisitt left-pad) + BiLSTM på den resulterende sekvensen, mean-pool over tid.
- **Inn:** `(B, C, T)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [CNNLSTMRawMultiTask](src/models/raw/cnn_lstm_raw.py#L40)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** `research_only=True` på klassen — BiLSTM-komponenten ser fremover i tid, så modellen kan IKKE deployeres i streaming-pipelinen. Bruk `cnn1d_raw` eller `tcn_raw` for sanntid.

### lstm_raw.py
- **Hva:** BiLSTM (eller unidirectional via `bidirectional=False`-flag) over rå biosignal-vinduer. Final timestep-output → projeksjon → heads.
- **Inn:** `(B, C, T)` → transposes internt til `(B, T, C)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [LSTMRawMultiTask](src/models/raw/lstm_raw.py#L39)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** Modulen eksporterer `RESEARCH_ONLY=True` og klasse-attributtet `research_only=True`. For deployment trenger man å sette `bidirectional=False` ved konstruksjon — dokumentert i docstringen, men `RESEARCH_ONLY`-flagget overkjøres ikke automatisk.

### tcn_raw.py
- **Hva:** Bai et al. 2018 dilated causal TCN. 4 blokker med dilation `[1,2,4,8]` og residual-koblinger. `_trim()` fjerner høyrekantens padding for å håndheve causality. Tar siste timestep som vindurepresentasjon.
- **Inn:** `(B, C, T)`. Default `kernel_size=3`, channels `(4, 8, 16, 32)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [TCNRawMultiTask](src/models/raw/tcn_raw.py#L84), [_TCNRawBlock](src/models/raw/tcn_raw.py#L45)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:**
  - Causality bekreftes i [tests/test_raw_tcn_causal.py](tests/test_raw_tcn_causal.py) per docstring.
  - Receptive field-beregning i header beskriver `kernel_size=5`; default i konstruktøren er `kernel_size=3` — dokumentert RF (121 samples) gjelder bare når `kernel_size=5` brukes.
  - Hovedkandidaten for sanntidsdeployment.

### __init__.py
- **Hva:** Re-eksporterer alle 4 klasser.
- **Eksportert:** `CNN1DRawMultiTask, LSTMRawMultiTask, CNNLSTMRawMultiTask, TCNRawMultiTask`.

## Dataflyt inn/ut av mappen

- **Leser:** Ingen filer.
- **Skriver:** Ingen filer.

## Relaterte mapper

- **Importerer fra:** [src/models/](../FOLDER.md) (`heads.MultiTaskHeads`).
- **Importeres av:** [src/pipeline/](../../pipeline/FOLDER.md) (`run_train_nn_raw_full.py`), tester (`test_raw_model_shapes.py`), `scripts/bench_workers.py`, `scripts/plot_fatigue_predictions.py`.
