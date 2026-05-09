# src3 — library-first refactor

Parallel implementation of the Strength-RT pipeline that swaps every
hand-rolled abstraction in [src/](../src/) for a maintained library.
[src/](../src/) and [src2/](../src2/) are untouched.

## Library mapping

| Concern in `src/` | Library used in `src3/` |
|---|---|
| `src/training/loop.py` (~486 LOC: AMP, GradScaler, early stop, TB, checkpoint, multi-seed × fold) | **PyTorch Lightning** ([Trainer](src3/training/lit_module.py) + callbacks) |
| `src/data/raw_window_dataset.py::_GPUBatchIterator` + `materialize_to_device` | Plain `DataLoader(pin_memory=True, persistent_workers=True)` driven by Lightning |
| `src/eval/metrics.py` (sklearn + scipy.stats) | **torchmetrics** `MetricCollection` ([metrics.py](eval/metrics.py)) |
| `src/training/cv.py` | `sklearn.model_selection.GroupKFold` / `LeaveOneGroupOut` ([splits.py](data/splits.py)) |
| `src/features/emg_features.py` (RMS/IEMG/MNF/MDF/Dimitrov/MFL/MSR/WAMP) | **libemg** `FeatureExtractor` + custom Dimitrov FInsm5 ([emg.py](features/emg.py)) |
| `src/features/ppg_features.py` | **neurokit2** `ppg_process` ([ppg.py](features/ppg.py)) |
| `src/features/eda_features.py` | **neurokit2** `eda_process` ([eda.py](features/eda.py)) |
| `src/features/acc_features.py` | **antropy** + scipy.signal ([acc.py](features/acc.py)) |
| `src/models/raw/tcn_raw.py` (custom causal TCN) | **pytorch-tcn** `TCN(causal=True)` ([encoders.py](models/encoders.py)) |
| `src/models/raw/{cnn1d,lstm,cnn_lstm}_raw.py` | torch.nn (kept idiomatic) |
| `src/streaming/{ecg,emg,eda,acc,ppg,temp}_streaming.py` (~1850 LOC across 7 files) | One [filters.py](streaming/filters.py) + one [modalities.py](streaming/modalities.py) |
| `configs/config.yaml` via raw `yaml.safe_load` | **OmegaConf** with dotted CLI overrides ([config.py](config.py)) |

## Scope

**In scope** — this folder is functional end-to-end:
- Reads `data/labeled/recording_NNN/aligned_features.parquet` produced by `src/labeling/` (labeling is one-shot, not refactored).
- Trains the same 4 architectures (CNN1D, LSTM, CNN-LSTM, TCN) on the same 4 tasks (exercise, phase, fatigue, reps) with the same multi-task loss.
- Reuses [configs/splits.csv](../configs/splits.csv) so results stay comparable to the LightGBM baseline.
- Loads [configs/config.yaml](../configs/config.yaml) via OmegaConf — no parallel config file.
- Library-backed feature extractors with in-house fallbacks if optional deps are missing.

**Out of scope** — keep using `src/`:
- Offline labeling (`src/labeling/`, `src/pipeline/label.py`) — Kinect joint-angle math is well-tested.
- LightGBM baselines — already library-based.

## Install

```bash
pip install -r src3/requirements.txt
```

PyTorch is assumed already installed (project pyproject + user's CUDA setup).
Optional deps (`libemg`, `neurokit2`, `antropy`, `pytorch-tcn`) all have
in-house fallbacks so the module imports cleanly without them.

## Run

```bash
# Default: TCN on raw windows, CV from configs/splits.csv:
python -m src3.pipeline.train

# All 4 raw architectures, 30 epochs each:
python -m src3.pipeline.train --arch all --variant raw --epochs 30

# MLP on engineered features, fatigue head only:
python -m src3.pipeline.train --arch mlp --variant features \
    --tasks fatigue \
    --features-parquet runs/.../window_features.parquet

# OmegaConf dotted overrides from CLI:
python -m src3.pipeline.train training.batch_size=128 training.lr=2e-3
```

## Layout

```
src3/
├── config.py                # OmegaConf loader + path resolver
├── data/
│   ├── encoders.py          # LabelEncoder (-1 = unknown)
│   ├── feature_dataset.py   # WindowFeatureDataset (Phase-1 input)
│   ├── raw_window_dataset.py# AlignedWindowDataset (Phase-2 input)
│   └── splits.py            # GroupKFold / LOSO via sklearn
├── features/
│   ├── _common.py           # nanfill + window iterator
│   ├── emg.py               # libemg + Dimitrov
│   ├── ppg.py               # neurokit2 PPG
│   ├── eda.py               # neurokit2 EDA (diagnostic only)
│   ├── acc.py               # antropy + scipy
│   ├── temp.py
│   └── pipeline.py          # orchestrator → window_features.parquet
├── models/
│   ├── encoders.py          # CNN1D / LSTM / CNN-LSTM / TCN / MLP
│   ├── heads.py             # MultiTaskHeads
│   └── multitask.py         # encoder + heads composition
├── training/
│   ├── data_module.py       # LightningDataModule per fold
│   ├── lit_module.py        # LightningModule (AMP, sched, ckpt)
│   └── losses.py            # MultiTaskLoss (uncertainty-weighted)
├── eval/
│   └── metrics.py           # torchmetrics MetricCollection per task
├── streaming/
│   ├── filters.py           # CausalBandpass / Notch / Lowpass / Chain
│   ├── online_stats.py      # Welford
│   └── modalities.py        # EMG/ACC/PPG/Temp streaming wrappers
├── pipeline/
│   └── train.py             # python -m src3.pipeline.train
├── utils/
│   └── device.py            # re-exports src/utils/device.py probes
├── requirements.txt
└── README.md
```

## Status

- [x] All modules import cleanly without optional deps (each library is wrapped in try/except with an in-house fallback).
- [ ] Parity tested against `src/` on one fold — TODO before promoting `src3/` to primary.
- [ ] EMG feature parity test against `src/features/emg_features.py` (libemg uses periodogram; ours uses Welch — 1–2 % drift expected on MNF/MDF).
- [ ] LightningModule logs match `src/training/loop.py` model_card.md — TODO write a `model_card.md` writer that consumes Lightning's `callback_metrics`.

## Why a parallel folder, not in-place

`src/` is currently driving thesis runs. Touching it risks breaking
re-runs. `src2/` is the previous library-first attempt — partial.
`src3/` continues that work with a wider scope (streaming, full config,
fallbacks for optional deps) without forcing a migration of either.
When parity is verified, atomic rename `src/` → `src_legacy/`,
`src3/` → `src/` and update imports.
