# pipeline/

**Formål:** CLI-entrypoints og slash-command-drivere — labeling, NN-trening (feature-vei og rå-vei), Phase 1/2-orkestrering, Optuna-tuning, latency-benchmark, comparison-rapporter og resume/finish-scripts. `train_nn.py` er det store fellesbiblioteket de andre driverne reuserer.

**Plass i pipeline:** Toppnivå-orkestrering. `/label` → `label.py`. `/train-nn` (CPU-budsjett) → `train_nn.py` direkte. Full GPU-økt → `run_train_nn_features_full.py` eller `run_train_nn_raw_full.py`. Etter en avbrutt kjøring brukes `resume_nn.py` / `finish_nn_features_full.py` / `finish_artifacts.py` for å fullføre uten å re-trene fra scratch.

## Filer

### finish_artifacts.py
- **Hva:** Idempotent post-treningsskript som genererer per-subject breakdown, comparison-fil, model_card per arch, soft-sharing-ablation og latency-benchmark fra cached fold-metrikker.
- **Inn:** `RUN_DIR = runs/20260426_160754_nn_comparison`, leser `<run_dir>/<arch>/seed_*/fold_*/metrics.json` + `best_hp.json` + `runs/.../features/window_features.parquet`. Kaller `WindowFeatureDataset` for å re-konstruere folds.
- **Ut:** `comparison.md`, `comparison.png`, `latency_table.md`, `model_card.md` per arch, evt. ny `ablation_results.json`. Skriver per-subject CSV/MD-rapporter.
- **Nøkkelfunksjoner:** Rent skript-flyt (ingen toppnivå-funksjoner). Reuser `train_nn.build_comparison`, `latency_benchmark`, `run_soft_sharing_ablation`.
- **Avhengigheter:** [src.pipeline.train_nn](src/pipeline/train_nn.py), [src.data.datasets](src/data/datasets.py), [src.eval.metrics](src/eval/metrics.py).
- **Gotchas:** Hardkodet `RUN_DIR`-konstant — kjøres bare for den spesifikke run-en. Endre konstanten eller kopier filen for nye kjøringer.

### finish_nn_features_full.py
- **Hva:** Større post-treningsdriver for `runs/20260427_121303_nn_features_full/`. Bygger `comparison.md`/`comparison.png`, `latency_table.md`, `multitask_ablation.md` og per-arch `model_card.md` med litteratur-siteringer.
- **Inn:** `RUN_DIR` hardkodet, `LGBM_METRICS_PATH = runs/20260427_110653_default/metrics.json`. Leser per-arch `cv_summary.json` + `phase2_summary.json` + ablation-resultater.
- **Ut:** Markdown-rapportene over + figurer i samme mappe.
- **Nøkkelfunksjoner:** [main()](src/pipeline/finish_nn_features_full.py#L641)
- **Avhengigheter:** [src.eval.plot_style](src/eval/plot_style.py) (lazy), [src.data.datasets](src/data/datasets.py) (kun via `WindowFeatureDataset` i lazy-import for ablation hot-fixes).
- **Gotchas:** Som `finish_artifacts.py` — `RUN_DIR` er bundet til én spesifikk kjøring.

### label.py
- **Hva:** `/label` CLI-entrypoint. Validerer at `inspections/findings.md` finnes, oppdager opptak under `dataset_aligned/`, og kaller `src.labeling.run.label_one_recording` per opptak. Skriver `data/labeled/_summary.md`.
- **Inn:** `--all` eller `--recording NNN` (kan gjentas). Hardkodet `PARTICIPANTS_XLSX = C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/Participants.xlsx` og `DATASET_ALIGNED_DIR = repo_root/dataset_aligned`.
- **Ut:** Per opptak: `data/labeled/recording_<NNN>/aligned_features.parquet` + `quality_report.md`. Globalt: `data/labeled/_summary.md`.
- **Nøkkelfunksjoner:** [_check_inspections()](src/pipeline/label.py#L51), [_discover_recordings()](src/pipeline/label.py#L74), [main()](src/pipeline/label.py#L195)
- **Avhengigheter:** [src.data.participants](src/data/participants.py) (lazy via `from src.data.participants import load_participants`), [src.labeling.run](src/labeling/run.py) (impliesert).
- **Gotchas:** Avbryter med exit-kode 1 hvis `inspections/`-mappen er tom eller mangler `findings.md` (CLAUDE.md hard-regel). `Participants.xlsx`-pathen er absolutt og maskinspesifikk — flyttes ikke automatisk på andre maskiner.

### resume_nn.py
- **Hva:** "TCN folds 1-4 manglet"-recovery-skript. Re-leser eksisterende fold 0-resultater fra disk og fortsetter med folds 1-4, deretter Phase 2 og artefakter.
- **Inn:** `RUN_DIR = runs/20260426_160754_nn_comparison`. Leser per-fold `metrics.json` der de finnes. Krever `WindowFeatureDataset`-input som i `run_all_nn.py`.
- **Ut:** Fyller inn manglende fold-mapper, oppdaterer `cv_summary.json`.
- **Nøkkelfunksjoner:** Skript-flyt, ingen funksjoner.
- **Avhengigheter:** [src.pipeline.train_nn](src/pipeline/train_nn.py), [src.data.datasets](src/data/datasets.py).
- **Gotchas:** Hardkodet TCN-fokus og bestemt run-mappe — utviklet som engang-recovery, ikke generell resume-driver.

### run_all_nn.py
- **Hva:** Kjører `/train-nn` ende-til-ende på CPU (Phase 1 4 archs × features × 5 folds × 1 seed → Phase 2 top-3 × 3 seeds → soft-sharing-ablation → latency → per-subject). Fixer `acc_rms`-overflow via IQR-normaliserer.
- **Inn:** `RUN_DIR = runs/20260426_160754_nn_comparison`, leser `runs/20260426_154705_default/features/window_features.parquet`.
- **Ut:** Hele run-mappen med per-arch `seed_*/fold_*/`, ablation, latency, comparison.
- **Nøkkelfunksjoner:** Skript-flyt; reuser `train_nn.run_variant`, `tune_hyperparams`, `run_soft_sharing_ablation`, etc.
- **Avhengigheter:** [src.pipeline.train_nn](src/pipeline/train_nn.py), [src.data.datasets](src/data/datasets.py).
- **Gotchas:** CPU-only — bruker `device = torch.device('cpu')` direkte, ignorerer GPU-deteksjonen i [src.utils.device](src/utils/device.py). For GPU bruk `run_train_nn_features_full.py`.

### run_train_nn_features_full.py
- **Hva:** Full-depth GPU-kjøring av features-veien — overstyrer `run_variant`-defaults med `subsample=False`, eksplisitte epochs/batch_size, og 30 Optuna-trials per arch. Phase 1 (4 archs × 1 seed) → Phase 2 (top 2-3 × 3 seeds).
- **Inn:** `RUN_DIR = runs/20260427_121303_nn_features_full`, `WINDOW_FEATURES_PATH = runs/20260427_110653_default/window_features.parquet`, `LGBM_METRICS_PATH = runs/20260427_110653_default/metrics.json`.
- **Ut:** Per-arch sub-mapper, comparison-rapport, latency-tabell.
- **Nøkkelfunksjoner:** [run_variant_gpu()](src/pipeline/run_train_nn_features_full.py#L97), [per_subject_metrics_gpu()](src/pipeline/run_train_nn_features_full.py#L214), [main()](src/pipeline/run_train_nn_features_full.py#L343)
- **Avhengigheter:** [src.utils.device.torch_device](src/utils/device.py), [src.data.datasets](src/data/datasets.py), [src.eval.metrics](src/eval/metrics.py), [src.training.losses](src/training/losses.py), [src.pipeline.train_nn](src/pipeline/train_nn.py) (gjenbruker mange helpers).
- **Gotchas:** Forventer GPU; `mixed_precision=True` (autocast på CUDA). Re-implementerer per-subject-metrikker i stedet for å arve `train_nn.per_subject_metrics` (som lekker `CPU_SUBSAMPLE_PER_FOLD` på subsample-veien).

### run_train_nn_raw_full.py
- **Hva:** Full GPU-kjøring av rå-veien (4 raw-arkitekturer på `(B, C, T)` vinduer). Phase 1 + Phase 2 + soft-sharing-ablation + latency-benchmark + comparison mot LightGBM og features-NN-runet. Mange Optuna trials per arch.
- **Inn:** `LABELED_DATA_ROOT = data/labeled/`, `SPLITS_PER_FOLD_PATH = configs/splits_per_fold.csv`, `LGBM_METRICS_PATH = runs/20260427_110653_default/metrics.json`, `FEATURES_RUN_PATH = runs/20260427_121303_nn_features_full`.
- **Ut:** `RUN_DIR = runs/20260427_153421_nn_raw_full/...` med per-arch fold-resultater, ablations, latency, comparison.
- **Nøkkelfunksjoner:** [load_baseline_splits()](src/pipeline/run_train_nn_raw_full.py#L157), [train_one_fold_raw()](src/pipeline/run_train_nn_raw_full.py#L342), [run_variant_raw()](src/pipeline/run_train_nn_raw_full.py#L612), [rank_variants()](src/pipeline/run_train_nn_raw_full.py#L711), [latency_benchmark_raw()](src/pipeline/run_train_nn_raw_full.py#L746), [run_soft_sharing_ablation_raw()](src/pipeline/run_train_nn_raw_full.py#L808), [build_comparison_raw()](src/pipeline/run_train_nn_raw_full.py#L1001), [main()](src/pipeline/run_train_nn_raw_full.py#L1380)
- **Avhengigheter:** [src.data.raw_window_dataset](src/data/raw_window_dataset.py), [src.models.raw.*](src/models/raw/FOLDER.md), [src.training.losses](src/training/losses.py), [src.eval.metrics](src/eval/metrics.py), [src.utils.device](src/utils/device.py).
- **Gotchas:** Kommentarer + signatur sier `(B, C=6, T=200)`, men `RAW_CHANNELS` definerer 4 kanaler etter ECG/EDA-eksklusjon. Optuna-trials er kostbare — Phase 1 kan ta timer på en RTX-klassemaskin.

### train_nn.py
- **Hva:** Hovedbiblioteket for `/train-nn` på features-veien — `FilteredWindowDataset`, fold-loader, IQR-baserte `FeatureNormalizer`/`FoldNormalizer`, `train_one_fold`, Optuna-`tune_hyperparams`, `run_variant`, `latency_benchmark`, `per_subject_metrics`, soft-sharing-ablation og comparison-builder. Brukes både direkte og som verktøykasse av andre pipeline-skript.
- **Inn:** `WINDOW_FEATURES_PATH = runs/20260426_154705_default/features/window_features.parquet`, `SPLITS_PER_FOLD_PATH = configs/splits_per_fold.csv`. Konstanter: `CPU_SUBSAMPLE_PER_FOLD=40000`, `CPU_EPOCHS_P1=20`, `CPU_EPOCHS_P2=30`, `CPU_OPTUNA_TRIALS=8`, `BATCH_SIZE=256`, `PATIENCE=6`, `LOSS_WEIGHTS={'exercise':1.0,'phase':1.0,'fatigue':1.0,'reps':0.5}`.
- **Ut:** Per-arch run-mapper med `seed_*/fold_*/{checkpoint_best.pt, history.json, metrics.json, test_preds.pt}`, `cv_summary.json`, `best_hp.json`, `comparison.md/png`, `latency_table.md`, `model_card.md`.
- **Nøkkelfunksjoner:** [FilteredWindowDataset](src/pipeline/train_nn.py#L106), [load_baseline_splits()](src/pipeline/train_nn.py#L160), [subsample_train_idx()](src/pipeline/train_nn.py#L199), [make_factory()](src/pipeline/train_nn.py#L228), [FeatureNormalizer](src/pipeline/train_nn.py#L270), [FoldNormalizer](src/pipeline/train_nn.py#L313), [train_one_fold()](src/pipeline/train_nn.py#L362), [tune_hyperparams()](src/pipeline/train_nn.py#L541), [run_variant()](src/pipeline/train_nn.py#L614), [latency_benchmark()](src/pipeline/train_nn.py#L727), [per_subject_metrics()](src/pipeline/train_nn.py#L797), [main()](src/pipeline/train_nn.py#L880), [rank_variants()](src/pipeline/train_nn.py#L1042), [run_soft_sharing_ablation()](src/pipeline/train_nn.py#L1075), [build_comparison()](src/pipeline/train_nn.py#L1207)
- **Avhengigheter:** [src.data.datasets](src/data/datasets.py), [src.data.phase_whitelist](src/data/phase_whitelist.py), [src.models.cnn1d](src/models/cnn1d.py), [src.models.lstm](src/models/lstm.py), [src.models.cnn_lstm](src/models/cnn_lstm.py), [src.models.tcn](src/models/tcn.py), [src.training.losses](src/training/losses.py), [src.eval.metrics](src/eval/metrics.py).
- **Gotchas:**
  - CPU-defaults er hardkodet — andre drivere må eksplisitt overstyre `subsample=False` og passere GPU-spesifikke epochs/batch.
  - `BiLSTM`-varianter merket `research_only` — håndteres av `make_factory` ved å hoppe over deployment-validering.
  - Optuna logger er stilt til WARNING-nivå.

## Dataflyt inn/ut av mappen

- **Leser:**
  - `dataset_aligned/recording_NNN/` (label-driver).
  - `data/labeled/recording_NNN/aligned_features.parquet` (rå-vei trening).
  - `runs/<slug>/features/window_features.parquet` (feature-vei trening).
  - `configs/splits_per_fold.csv` (subject-wise CV folds).
  - LightGBM-baseline `metrics.json` for sammenligning.
- **Skriver:**
  - `data/labeled/recording_NNN/aligned_features.parquet` + `quality_report.md` (label-driver).
  - `data/labeled/_summary.md`.
  - `runs/<slug>/<arch>/seed_*/fold_*/{checkpoint_best.pt, history.json, metrics.json, test_preds.pt, tb/}`.
  - `runs/<slug>/<arch>/{cv_summary.json, best_hp.json, model_card.md}`.
  - `runs/<slug>/{comparison.md, comparison.png, latency_table.md, multitask_ablation.md, baselines.json}`.

## Relaterte mapper

- **Importerer fra:** [src/labeling/](../labeling/FOLDER.md), [src/data/](../data/FOLDER.md), [src/features/](../features/FOLDER.md), [src/models/](../models/FOLDER.md), [src/models/raw/](../models/raw/FOLDER.md), [src/training/](../training/FOLDER.md), [src/eval/](../eval/FOLDER.md), [src/streaming/](../streaming/FOLDER.md) (latency_benchmark via eval).
- **Importeres av:** Tester og `scripts/`-utvalget av Optuna-/ablation-skript som gjenbruker pipeline-helpers.
