# models/

**Formål:** PyTorch-arkitekturer for feature-veien (Phase 1) — fem multi-task arkitekturer som tar per-vindu engineered features og forutsier {exercise, phase, fatigue, reps}. Hard parameter sharing (Caruana 1997): shared encoder + 4 lineære heads.

**Plass i pipeline:** Modus 1 (offline trening). Brukes etter at `WindowFeatureDataset` er bygget. Rå-signal-arkitekturer ligger i undermappen [src/models/raw/](raw/FOLDER.md).

## Filer

### cnn1d.py
- **Hva:** 1D-CNN-encoder over feature-sekvens. `Conv1d`-stack med default channels `(16,32,32)` og kerneler `(5,3,3)` etterfulgt av `AdaptiveAvgPool1d` og lineær projeksjon til `repr_dim`. Kjører konvolusjon LANGS feature-aksen — antar at relaterte features er gruppert sammen i kolonneorden.
- **Inn:** `(B, n_features)` float-tensor. Reshapes til `(B, 1, n_features)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [CNN1DMultiTask](src/models/cnn1d.py#L29)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** Ikke causal — bruker `padding=k//2`. Ikke for sanntid; bruk TCN i stedet.

### cnn_lstm.py
- **Hva:** DeepConvLSTM-stil hybrid. To Conv1d-lag mikser nabofeatures, BiLSTM modellerer den resulterende "sekvensen", mean-pool over feature-aksen og lineær projeksjon.
- **Inn:** `(B, n_features)` → reshape til `(B, 1, n_features)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [CNNLSTMMultiTask](src/models/cnn_lstm.py#L25)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** `bidirectional=True` på LSTM — kun for offline-evaluering. Streaming-deploy må bruke unidirectional-variant.

### heads.py
- **Hva:** `MultiTaskHeads` — delt for alle 5 arkitekturer. Tar `repr_dim`-vektor og produserer 4 task-spesifikke utganger (cross-entropy logits for exercise/phase, scalar regresjon for fatigue/reps).
- **Inn:** `repr_dim`, `n_exercise`, `n_phase`, `dropout`. Forward tar `h: (B, repr_dim)`.
- **Ut:** Dict `{'exercise': (B, n_ex), 'phase': (B, n_ph), 'fatigue': (B,), 'reps': (B,)}`.
- **Nøkkelfunksjoner:** [MultiTaskHeads](src/models/heads.py#L14)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Squeeze på siste dimensjon for fatigue/reps — sørg for at downstream loss-fn forventer `(B,)` ikke `(B, 1)`.

### lstm.py
- **Hva:** Unidirectional LSTM (causal — krav for sanntidsbruk). Behandler `(B, n_features)` som sekvens med 1 kanal. Mean-pool over feature-aksen.
- **Inn:** `(B, n_features)` → `(B, n_features, 1)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [LSTMMultiTask](src/models/lstm.py#L24)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** `bidirectional=False` — eksplisitt valg pga sanntidsdeployment (CLAUDE.md "Modellvalg").

### mlp.py
- **Hva:** Plain MLP (2 skjulte lag) som non-temporal baseline mot conv/recurrent-arkitekturene. BatchNorm + Dropout.
- **Inn:** `(B, n_features)`.
- **Ut:** Dict fra `MultiTaskHeads`.
- **Nøkkelfunksjoner:** [MLPMultiTask](src/models/mlp.py#L19)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** Default `repr_dim=80, hidden_dim=80` — relativt liten, low-data regime (~108 RPE-rader).

### tcn.py
- **Hva:** Temporal Convolutional Network med dilated causal conv-blokker (dilation `1,2,4,8`) og residual-koblinger. Strengt causal — naturlig deploymentskandidat.
- **Inn:** `(B, n_features)` → `(B, 1, n_features)`.
- **Ut:** Dict fra `MultiTaskHeads`. Tar siste timestep (causal).
- **Nøkkelfunksjoner:** [TCNMultiTask](src/models/tcn.py#L55), [_TCNBlock](src/models/tcn.py#L26)
- **Avhengigheter:** [src.models.heads.MultiTaskHeads](src/models/heads.py).
- **Gotchas:** `_trim()` etter konvolusjon fjerner høyrekant-padding for å beholde causality. I Phase 1 brukes causality på feature-aksen; i Phase 2 (rå-vei) på tids-aksen — samme arkitektur.

## Dataflyt inn/ut av mappen

- **Leser:** Ingen filer.
- **Skriver:** Ingen filer.

## Relaterte mapper

- **Importeres av:** [src/pipeline/](../pipeline/FOLDER.md) (`train_nn.py`, `run_train_nn_features_full.py`), tester (`test_model_shapes.py`), `scripts/bench_workers.py`. Heads-modulen gjenbrukes også av [src/models/raw/](raw/FOLDER.md).
- **Undermapper:** [src/models/raw/](raw/FOLDER.md) — rå-signal-varianter for Phase 2.
