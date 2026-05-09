# src2 — library-first refactor

Parallel implementation of the Strength-RT pipeline that swaps every
hand-rolled abstraction in `src/` for a maintained library. `src/` is
untouched and remains the reference implementation.

## Library mapping

| Concern in `src/` | Library used in `src2/` |
|---|---|
| `src/training/loop.py` (custom train/val loop, AMP, GradScaler, early stop, TB, checkpointing, multi-seed×fold runner) | **PyTorch Lightning** |
| `src/data/raw_window_dataset.py::_GPUBatchIterator` | Lightning `Trainer(accelerator='gpu')` + `DataLoader(pin_memory=True)` (drop the custom iterator) |
| `src/eval/metrics.py` (sklearn + scipy.stats wrappers) | **torchmetrics** (`F1Score`, `Accuracy`, `MeanAbsoluteError`, `PearsonCorrCoef`) |
| `src/training/cv.py` | `sklearn.model_selection.{LeaveOneGroupOut,GroupKFold}` (already used; src2 just calls it directly) |
| `src/features/emg_features.py` (RMS/IEMG/MNF/MDF/Dimitrov/MFL/MSR/WAMP) | **libemg** `FeatureExtractor` |
| `src/features/ppg_features.py` | **neurokit2** `ppg_process` + `hrv_time` |
| `src/features/eda_features.py` | **neurokit2** `eda_process` |
| `src/features/acc_features.py` | **antropy** + scipy.signal |
| `src/features/temp_features.py` | scipy.stats / numpy (kept thin) |
| `src/models/raw/tcn_raw.py` (custom causal TCN with manual trim) | **pytorch-tcn** `TCN(causal=True)` |
| `src/models/raw/{cnn1d,lstm,cnn_lstm}_raw.py` | torch.nn (kept — these are already idiomatic) |
| `configs/config.yaml` (read via raw `yaml.safe_load`) | **OmegaConf** (dotted access, structured override CLI) |
| `src/training/losses.py` (Kendall uncertainty weighting) | **uw-loss** if installed; falls back to a tiny `nn.Module` |

## Scope

**In scope** (this folder is functional end-to-end):
- Reads existing `data/labeled/recording_NNN/aligned_features.parquet` produced by `src/labeling/` (we don't re-label — labeling is a 1-time offline cost, kept in `src/`).
- Trains the same 4 architectures (CNN1D, LSTM, CNN-LSTM, TCN) on the
  same 4 tasks (exercise, phase, fatigue, reps) with the same multi-task loss.
- Reuses `configs/splits.csv` so results stay comparable to the LightGBM baseline.
- Loads `configs/config.yaml` via OmegaConf — no parallel config file.

**Out of scope** (still use `src/`):
- Offline labeling (`src/labeling/`, `src/pipeline/label.py`) — Kinect joint-angle math is one-shot and well-tested.
- Real-time streaming (`src/streaming/`) — separate refactor; needs a streaming-friendly libemg adapter.
- LightGBM baselines — already library-based; no win from rewriting.

## Install

```bash
pip install -r src2/requirements.txt
```

Adds: `lightning`, `torchmetrics`, `libemg`, `neurokit2`, `antropy`,
`omegaconf`, `pytorch-tcn`. PyTorch is assumed already installed
(see top-level pyproject + the user's CUDA setup).

## Run

```bash
# Train one architecture across all CV folds × seeds:
python -m src2.pipeline.train --arch tcn --epochs 50

# All 4 architectures back-to-back:
python -m src2.pipeline.train --arch all --epochs 50

# Override any config.yaml key from CLI (OmegaConf dotted syntax):
python -m src2.pipeline.train --arch tcn training.batch_size=128 training.lr=2e-3
```

## Layout

```
src2/
├── config.py                # OmegaConf loader
├── data/
│   ├── loaders.py           # thin pandas wrappers (reuses src.data.loaders)
│   ├── parquet_dataset.py   # torch Dataset over aligned_features.parquet
│   └── splits.py            # sklearn GroupKFold / LOSO
├── features/
│   ├── emg.py               # libemg FeatureExtractor
│   ├── ppg.py               # neurokit2
│   ├── eda.py               # neurokit2
│   ├── acc.py               # antropy + scipy
│   ├── temp.py
│   └── pipeline.py          # orchestrator → window_features.parquet
├── models/
│   ├── encoders.py          # CNN1D / LSTM / CNN-LSTM / TCN encoders
│   ├── heads.py             # MultiTaskHeads
│   └── multitask.py         # encoder + heads composition
├── training/
│   ├── data_module.py       # LightningDataModule
│   ├── lit_module.py        # LightningModule
│   └── losses.py            # MultiTaskLoss (uncertainty-weighted)
├── eval/
│   └── metrics.py           # torchmetrics MetricCollection
└── pipeline/
    └── train.py             # python -m src2.pipeline.train
```
