# training/

**Formål:** Subject-wise CV-driver, multi-task treningsløkke og multi-task loss for nevrale nett. Arkitekturskript bruker disse byggeklossene direkte; mappen kjenner ikke til arkitekturer eller datasett bortsett fra at de er PyTorch-`nn.Module` og en `Dataset`-lignende objekt.

**Plass i pipeline:** Faseskjema 2 (NN). `run_cv()` reuser folds fra LightGBM-baseline (`configs/splits.csv`) for fair sammenligning. Kalles fra `src/pipeline/train_nn.py`, `run_train_nn_*_full.py` og diverse `scripts/*.py`-drivere.

## Filer

### cv.py
- **Hva:** Laster eller genererer subject-wise CV-folds. Foretrekker eksisterende `configs/splits.csv` fra Random Forest/LightGBM-kjøringen så NN trenes på samme deltakerinndeling.
- **Inn:** `subject_ids: np.ndarray`, valgfritt `splits_path` (default `configs/splits.csv`), `n_splits_if_new`. Leser CSV-kolonner `subject_id`, `fold` (og `split`).
- **Ut:** Liste av dicts `{fold, train_idx, test_idx, test_subjects}`. Skriver ny `configs/splits.csv` (kolonner `subject_id, fold, split`) når ingen finnes.
- **Nøkkelfunksjoner:** [load_or_generate_splits()](src/training/cv.py#L12), [loso_splits()](src/training/cv.py#L76)
- **Avhengigheter:** Ingen interne `src.*`-imports.
- **Gotchas:** Hvis `splits.csv` mangler subject-id-er fra kjøringen blir det `ValueError` — fair sammenligning krever full dekning. `loso_splits()` bør bare brukes når N ≤ ~24 (kompute).

### loop.py
- **Hva:** Multi-task NN-treningsløkke (én epoch + per-fold + per-(seed,fold)) og resultat-aggregering. Inkluderer GPU-resident batch-iterator (`_GPUBatchIterator`) som hopper over DataLoader når datasettet er materialisert på CUDA.
- **Inn:** `model_factory` (callable → `nn.Module`), `dataset` (med valgfri `gpu_resident=True`, `_gpu_x`, `_gpu_targets`, `_gpu_masks`), `train_idx`/`test_idx` (numpy), `TrainConfig`, `splits` fra `cv.py`, `out_root: Path`, `seeds`, `n_exercise`, `n_phase`. Leser ingen filer.
- **Ut:** Per fold-mappe: `checkpoint_best.pt`, `history.json`, `metrics.json`, `test_preds.pt` (sistnevnte gated av `cfg.save_checkpoint`). Per arkitektur-mappe: `cv_summary.json`. Returnerer `(summary, all_results)` fra `run_cv`.
- **Nøkkelfunksjoner:** [TrainConfig](src/training/loop.py#L96), [set_deterministic()](src/training/loop.py#L128), [train_one_fold()](src/training/loop.py#L224), [run_cv()](src/training/loop.py#L363), [aggregate_cv_results()](src/training/loop.py#L428)
- **Avhengigheter:** [src.training.losses.MultiTaskLoss](src/training/losses.py), [src.eval.metrics.compute_all_metrics](src/eval/metrics.py).
- **Gotchas:**
  - `cudnn.benchmark=True` og TF32 (`set_float32_matmul_precision('high')`) settes på modul-load — bytter bit-reproduserbarhet mot 1.5–2× speedup.
  - Når `use_uncertainty_weighting=True` legges `loss_fn`-parametre til optimizeren i en separat `param group` med `weight_decay=0` (Kendall et al. 2018).
  - Optimizer-skritt hoppes over når alle masker i batchen er `False` for å unngå AMP `GradScaler`-feil — relevant for `enabled_tasks=['fatigue']` med `active_only=False`.
  - `mixed_precision=True` ⇒ `autocast('cuda', ...)`; vil feile når trening kjører på CPU. `device` velges av `torch.cuda.is_available()` direkte i `run_cv()` (uavhengig av `src/utils/device.py`).
  - TensorBoard er valgfri import; mangler den hoppes logging.
  - `aggregate_cv_results()` markerer ikke-aktive heads `{'untrained': True}` slik at `untrained` ≠ dårlig metrikk.

### losses.py
- **Hva:** `MultiTaskLoss` — kombinerer fire tap (CE for exercise, CE eller KL for phase, L1 for fatigue, smooth-L1 for reps) med per-task masking. Valgfri Kendall et al. 2018 lærbar log-var.
- **Inn:** `preds`, `targets`, `masks` (alle `Dict[str, Tensor]`). `target_modes={'reps': 'soft_window'|'hard', 'phase': 'soft'|'hard'}`. `enabled_tasks` velger hvilke tasks som bidrar til total-loss.
- **Ut:** `(total: Tensor, parts: Dict[str, Tensor])` — `parts` holder per-task tap (alltid alle 4 nøkler, satt til 0 for tasks uten signal i batchen).
- **Nøkkelfunksjoner:** [MultiTaskLoss](src/training/losses.py#L16), [forward()](src/training/losses.py#L65)
- **Avhengigheter:** Bare `torch`/`torch.nn.functional`.
- **Gotchas:** Når `use_uncertainty_weighting=True` opprettes `nn.Parameter(log_var)` per ENABLED task — disse må eksplisitt legges til optimizeren (gjøres i `loop.py`). Tasks uten signal i batch (`mask.any()=False`) hopper over uncertainty-termen i stedet for å gi 0.5·log_var-bidrag som ville biasere de lærte vektene. `target_modes['phase']='soft'` forventer (B, K)-sannsynlighetsmål; `'hard'` forventer long-indices.

## Dataflyt inn/ut av mappen

- **Leser:** `configs/splits.csv` (kolonner: `subject_id`, `fold`, `split`).
- **Skriver:** `configs/splits.csv` ved første gangs generering. Per-fold/-seed under `out_root/<arch>/seed_<s>/fold_<f>/`: `checkpoint_best.pt`, `history.json`, `metrics.json`, `test_preds.pt`, `tb/` (TensorBoard). Per arkitektur: `<arch>/cv_summary.json`.

## Relaterte mapper

- **Importerer fra:** [src/eval/](../eval/FOLDER.md) (`metrics.compute_all_metrics`)
- **Importeres av:** [src/pipeline/](../pipeline/FOLDER.md) (`train_nn.py`, `run_train_nn_features_full.py`, `run_train_nn_raw_full.py`). Også brukt direkte fra `scripts/` (Optuna, top-K, ablation, raw-direct), tester og `_common.py`.
