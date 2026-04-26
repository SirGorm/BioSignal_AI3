# Scripts — Multi-task NN training

Four model-specific training scripts. **Each script trains a fundamentally
different architecture**. All four use multi-task hard parameter sharing:
one shared encoder + 4 task-specific linear heads (exercise, phase, fatigue,
reps). The encoder is what differs between scripts.

## Files

| Script | Architecture | Encoder type | Notes |
|--------|--------------|--------------|-------|
| `train_cnn1d.py`    | 1D-CNN   | 3 Conv1d blocks → GAP → Linear | Yang et al. 2015 |
| `train_lstm.py`     | BiLSTM   | 2-layer bidirectional LSTM | non-causal, research only |
| `train_cnn_lstm.py` | CNN-LSTM | 2 Conv1d blocks → BiLSTM | DeepConvLSTM (Ordóñez & Roggen 2016) |
| `train_tcn.py`      | TCN      | 4 dilated causal conv blocks | causal — deployment candidate (Bai 2018) |
| `compare_architectures.py` | — | — | aggregates 4 runs into comparison table |
| `_common.py`        | — | — | shared training entry point |

Model definitions live in `src/models/<arch>.py`. The training loop in
`src/training/loop.py` is **architecture-agnostic** — same loop runs all four.

## Run order

```bash
# 1. Smoke-test ONE script to verify pipeline (1 fold × 1 seed × 3 epochs)
python scripts/train_cnn1d.py --smoke-test

# 2. If smoke passes, run all 4 with full settings
python scripts/train_cnn1d.py    --run-slug cnn1d-baseline
python scripts/train_lstm.py     --run-slug lstm-baseline
python scripts/train_cnn_lstm.py --run-slug cnn-lstm-baseline
python scripts/train_tcn.py      --run-slug tcn-baseline

# 3. Aggregate into comparison
python scripts/compare_architectures.py \
    --runs runs/<timestamp>_cnn1d-baseline \
           runs/<timestamp>_lstm-baseline \
           runs/<timestamp>_cnn-lstm-baseline \
           runs/<timestamp>_tcn-baseline \
    --baseline-run runs/<lgbm_run_dir> \
    --output-slug nn-features-comparison
```

## Multi-task structure

Each architecture's encoder produces a representation `(B, repr_dim)` that 4
linear heads project into:

- `head_exercise`: multi-class classification (cross-entropy)
- `head_phase`: multi-class classification (cross-entropy)
- `head_fatigue`: regression on RPE 1–10 (L1 / MAE)
- `head_reps`: regression on `rep_count_in_set` (smooth L1)

The shared `MultiTaskHeads` class lives in `src/models/heads.py` and is
imported by all 4 model files.

Reference: Caruana 1997 (multitask learning, hard parameter sharing).

## Input handling

Per-window features arrive as `(B, n_features)`. Each architecture
reshapes/uses them differently:

- **1D-CNN**: reshape to `(B, 1, n_features)` and slide kernels across the
  feature dimension. Adjacent features should be related (e.g., all EMG
  features grouped together) for kernels to find local context.
- **LSTM**: reshape to `(B, n_features, 1)` and step through features
  one-by-one. Bidirectional helps because feature ordering is arbitrary.
- **CNN-LSTM**: 1D-CNN front-end, then BiLSTM over conv-pooled sequence.
- **TCN**: dilated causal convs over the feature dimension.

In Phase 2 (raw signals), these architectures will see actual time series.
The model classes are reusable as-is — just feed them sequences instead of
features.

## CV strategy

Reuses `configs/splits.csv` from the LightGBM baseline run (subject-wise) to
guarantee identical fold assignments. If `splits.csv` is missing, falls back
to GroupKFold(5) on subject_id and writes a new splits file.

Reference: Saeb et al. 2017 (subject-wise CV).

## Outputs per run

```
runs/<timestamp>_<slug>/
├── train_config.json
├── dataset_meta.json          # feature columns, encoder classes, arch
└── <arch>/
    ├── seed_42/
    │   ├── fold_0/
    │   │   ├── checkpoint_best.pt
    │   │   ├── history.json
    │   │   ├── metrics.json
    │   │   └── test_preds.pt
    │   ├── fold_1/...
    │   └── ...
    ├── seed_1337/...
    ├── seed_7/...
    └── cv_summary.json         # mean ± std across folds × seeds
```
