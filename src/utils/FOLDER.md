# utils/

**Formål:** Felles hjelpefunksjoner som ikke hører hjemme i en domene-spesifikk subpakke. Akkurat nå er innholdet auto-deteksjon av GPU-backend for LightGBM og PyTorch.

**Plass i pipeline:** Brukes på tvers av faser. Lastes av treningsskript og pipeline-drivere før de instansierer modeller, slik at riktig device velges automatisk.

## Filer

### device.py
- **Hva:** GPU-deteksjonshjelpere som probet LightGBM (CUDA-build → OpenCL `gpu` → CPU) og PyTorch (`torch.cuda.is_available()`). Resultatet caches per modul-load.
- **Inn:** Ingen filer. Leser miljøvariabel `STRENGTH_RT_FORCE_CPU` for å overstyre til CPU.
- **Ut:** Strenger `'cuda' | 'gpu' | 'cpu'`. `lgbm_params_with_device(params)` returnerer kopiert params-dict beriket med `device` (+ OpenCL platform/device-id når `gpu` velges).
- **Nøkkelfunksjoner:** [lgbm_device()](src/utils/device.py#L24), [torch_device()](src/utils/device.py#L51), [lgbm_params_with_device()](src/utils/device.py#L62), [report()](src/utils/device.py#L76)
- **Avhengigheter:** Ingen interne `src.*`-imports. Lazy-importerer `lightgbm` og `torch` inne i probene slik at modulen kan importeres uten dem.
- **Gotchas:** Probe-kallene kjører en mini-`fit` på syntetiske data for LightGBM — første kall tar et lite ekstra øyeblikk. Modulen sluker alle exceptions stille og returnerer `'cpu'`. CUDA-bygget av LightGBM foretrekkes over OpenCL `gpu` på NVIDIA. Når `device='gpu'` velges settes `gpu_platform_id=0` og `gpu_device_id=0` for å unngå multi-GPU-feil.

## Dataflyt inn/ut av mappen

- **Leser:** Ingen filer.
- **Skriver:** Ingen filer.

## Relaterte mapper

- **Importeres av:** [src/pipeline/](../pipeline/FOLDER.md) (kalles av `run_train_nn_features_full.py`, `run_train_nn_raw_full.py`)
