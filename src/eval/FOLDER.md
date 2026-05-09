# eval/

**Formål:** Evalueringsstack for prosjektet — per-task metrikker, dummy-baselines (gulv), latency-benchmark av streaming-pipelinen, treningskurve- og confusion-matrix-plotting, paired statistical tests og feature-relevansanalyse.

**Plass i pipeline:** Brukes i alle treningsstadier. `metrics.compute_all_metrics()` kalles fra `training/loop.py` per epoch; `baselines.py` produserer floor-tall som rapporteres i `model_card.md`; `latency_benchmark.py` validerer at p99-latency på streaming-pipelinen er innenfor budsjett.

## Filer

### baselines.py
- **Hva:** DummyClassifier/DummyRegressor floor-tall per task (subject-wise GroupKFold). Modeller MÅ slå disse for å være meningsfulle.
- **Inn:** `set_features.parquet` (for fatigue/reps), `window_features.parquet` (for exercise/phase). Trener `DummyRegressor(strategy='mean')` for fatigue, `DummyClassifier(strategy='most_frequent')` for klassifisering, mean-reps-per-set for reps.
- **Ut:** Dict per task med `fold_scores`, `mean`, `std`, `n_samples`. `main(run_dir)` skriver `<run_dir>/baselines.json`.
- **Nøkkelfunksjoner:** [fatigue_baseline()](src/eval/baselines.py#L23), [exercise_baseline()](src/eval/baselines.py#L44), [phase_baseline()](src/eval/baselines.py#L65), [rep_baseline()](src/eval/baselines.py#L88), [main()](src/eval/baselines.py#L111)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** `n_splits = min(n_splits, len(unique_groups))` — låser til minst antall grupper for å unngå sklearn-feil ved få subjekter.

### feature_analysis.py
- **Hva:** Feature relevance per task — Fisher LDA, ANOVA F-test, og mutual information. Aggregert rangering på tvers av tasks. Lekkasjesikker variant `select_features_within_fold()` skal kalles INNE i hver CV-fold sin trenings-side.
- **Inn:** `X: np.ndarray` (n_samples × n_features), `y: np.ndarray`, `task_kind: 'classification'|'regression'`. `per_task_feature_table()` tar window_df direkte.
- **Ut:** DataFrames med per-feature score per metode + aggregert rang. `write_report()` produserer Markdown; `write_top_k_list()` skriver topp-K liste.
- **Nøkkelfunksjoner:** [lda_scores()](src/eval/feature_analysis.py#L46), [anova_scores()](src/eval/feature_analysis.py#L63), [mutual_info_scores()](src/eval/feature_analysis.py#L81), [per_task_feature_table()](src/eval/feature_analysis.py#L113), [aggregate_across_tasks()](src/eval/feature_analysis.py#L158), [select_features_within_fold()](src/eval/feature_analysis.py#L176), [write_report()](src/eval/feature_analysis.py#L201), [write_top_k_list()](src/eval/feature_analysis.py#L272)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Modulen sin docstring advarer eksplisitt om lekkasje — bruk `select_features_within_fold` ikke per-task tabeller for feature-utvalg.

### latency_benchmark.py
- **Hva:** Replayer N minutter gjennom `StreamingFeaturePipeline` + trent inferens og måler per-hop wall-clock. Reporterer p50/p95/p99. PASS hvis p99 ≤ budsjett (default 100 ms).
- **Inn:** `--recording <rec_dir> --run <run_dir> --minutes <N>`. Leser native CSVer + `metadata.json` (PPG fs).
- **Ut:** `<run_dir>/latency_<rec_id>.json` + valgfri PNG av latency-distribusjon. Returnerer eksitkode 0/1.
- **Nøkkelfunksjoner:** [benchmark()](src/eval/latency_benchmark.py#L32), [write_artifacts()](src/eval/latency_benchmark.py#L177), [main()](src/eval/latency_benchmark.py#L211)
- **Avhengigheter:** [src.streaming.realtime](src/streaming/realtime.py), [src.data.loaders](src/data/loaders.py), `src.eval.plot_style` (lazy).
- **Gotchas:** `chunk_size=1` målet per-sample worst-case; `chunk_size=10` vil gi mer realistisk live-tall. Bruker samme `metadata.json`-leseing som streaming-pipelinen for å plukke PPG-fs (samme tension som notert i streaming-folderen).

### metrics.py
- **Hva:** `compute_all_metrics()` — F1-macro/balanced accuracy for klassifisering (exercise, phase), MAE + Pearson r for fatigue, MAE for reps. Per-subject breakdown og per-set integer rep-count fra soft targets.
- **Inn:** `preds`, `targets`, `masks` Dicts av CPU-tensorer; `n_exercise`, `n_phase`. Soft phase-target med `(B, K)` håndteres via `argmax`.
- **Ut:** Dict per task med metrikkene + `n` (antall validerte samples).
- **Nøkkelfunksjoner:** [compute_all_metrics()](src/eval/metrics.py#L18), [per_set_rep_count_metrics()](src/eval/metrics.py#L100), [per_subject_breakdown()](src/eval/metrics.py#L142)
- **Avhengigheter:** [src.eval.rep_aggregation](src/eval/rep_aggregation.py) (lazy).
- **Gotchas:** Returnerer NaN når mask er tom — kallere må håndtere det. Trenger ≥ 2 samples for Pearson.

### plot_style.py
- **Hva:** Prosjektomfattende seaborn-tema og despine-helper. Idempotent.
- **Inn:** Kall `apply_style()` én gang ved import; `despine(fig=fig)` per figur før save.
- **Ut:** Endringer i matplotlib-rcParams.
- **Nøkkelfunksjoner:** [apply_style()](src/eval/plot_style.py#L35), [despine()](src/eval/plot_style.py#L40)
- **Avhengigheter:** Ingen interne `src.*`-imports.

### plotting.py
- **Hva:** Treningskurver fra `history.json`, confusion matrices fra `test_preds.pt`, fatigue calibration scatter, reps-eval-plot. `plot_everything_for_run()` er one-shot driver.
- **Inn:** Per-arch run-mappe `runs/<slug>/<arch>/seed_*/fold_*/`. Leser `history.json`, `test_preds.pt`.
- **Ut:** PNG-filer i samme run-mappe (training_curves, confusion_matrix, fatigue_calibration, reps_eval).
- **Nøkkelfunksjoner:** [plot_history_for_fold()](src/eval/plotting.py#L34), [plot_all_histories()](src/eval/plotting.py#L86), [plot_aggregated_history()](src/eval/plotting.py#L106), [plot_confusion_matrix()](src/eval/plotting.py#L152), [plot_confusion_matrices_for_run()](src/eval/plotting.py#L186), [plot_fatigue_calibration()](src/eval/plotting.py#L246), [plot_reps_evaluation()](src/eval/plotting.py#L289), [plot_everything_for_run()](src/eval/plotting.py#L352)
- **Avhengigheter:** [src.eval.plot_style](src/eval/plot_style.py).
- **Gotchas:** Bruker `matplotlib.use('Agg')` for headless-servere — modulen kan ikke vise figurer interaktivt.

### rep_aggregation.py
- **Hva:** Aggregerer per-vindu soft rep-prediksjoner til per-sett integer counts via `total = sum(soft_pred) * (hop_s / window_s)`. Default-skala `0.05` for `window_s=2.0, hop_s=0.1`.
- **Inn:** `window_preds: Sequence[float]`, `set_ids: Sequence` (per-vindu identifikator, NaN ignoreres), `hop_s, window_s`.
- **Ut:** Integer count (round-half-to-even) eller `Dict[set_id, int]`.
- **Nøkkelfunksjoner:** [soft_to_set_count()](src/eval/rep_aggregation.py#L22), [soft_to_set_counts_grouped()](src/eval/rep_aggregation.py#L54)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** `clip_negative=True` er default — nettverkssoutputs kan bli litt negative ved grenser, og uten clipping vil totalsummen bias seg ned.

### significance.py
- **Hva:** Paired t-test og Wilcoxon signed-rank for sammenligning av to modeller på SAMME folds. Bonferroni-korreksjon for multiple comparisons. `compare_models_across_tasks()` returnerer en sammenligningstabell.
- **Inn:** Lister av per-fold metrikker fra to modeller (samme orden, samme lengde).
- **Ut:** Dict med `n, p_value, test, mean_diff, effect_size` (Cohen's d).
- **Nøkkelfunksjoner:** [paired_test()](src/eval/significance.py#L29), [bonferroni_correction()](src/eval/significance.py#L90), [compare_models_across_tasks()](src/eval/significance.py#L98), [render_significance_table()](src/eval/significance.py#L135)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Modulen sin egen docstring advarer eksplisitt: 5 folds × 3 seeds = 15 paired observations gir lav power; rapporter p-verdier sammen med effect size; juster for multiple comparisons.

## Dataflyt inn/ut av mappen

- **Leser:**
  - `runs/<slug>/<arch>/seed_*/fold_*/history.json`, `test_preds.pt` (plotting).
  - `<run_dir>/features/window_features.parquet`, `<run_dir>/features/set_features.parquet` (baselines, feature_analysis).
  - `dataset/<rec_id>/{ecg,emg,eda,...}.csv`, `dataset/<rec_id>/metadata.json` (latency_benchmark replay).
- **Skriver:**
  - `<run_dir>/baselines.json`.
  - PNG-filer i run-mappen: `training_curves_*.png`, `confusion_matrix_*.png`, `fatigue_calibration_*.png`, `reps_evaluation_*.png`.
  - `<run_dir>/latency_<rec_id>.json` (+ valgfri PNG).
  - Feature-analyse Markdown-rapport og topp-K-liste.

## Relaterte mapper

- **Importerer fra:** [src/streaming/](../streaming/FOLDER.md) (latency_benchmark), [src/data/](../data/FOLDER.md) (loaders for benchmark).
- **Importeres av:** [src/training/](../training/FOLDER.md) (`metrics.compute_all_metrics`), [src/pipeline/](../pipeline/FOLDER.md) (alle NN-trenings-drivere). `plot_style` brukes også av mange `scripts/`-plottere.
