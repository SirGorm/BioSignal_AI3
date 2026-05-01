---
description: Phase 2 — train the 4 NN architectures on raw multimodal signal windows (NOT YET IMPLEMENTED). Run after /train-nn.
allowed-tools: Bash, Read, Write, Edit
argument-hint: [run_slug]
---

# Train Neural Networks on Raw Signals (Phase 2 — placeholder)

**STATUS: Models built, dataset built, but no training script exists yet.**
The dataset class (`RawWindowDataset` in `src/data/datasets.py`) and the
4 model classes for raw input (`*MultiTask` in `src/models/<arch>.py`) are
ready. What's missing is `scripts/train_<arch>_raw.py` for each architecture.

## When to run this

After Phase 1 (`/train-nn`) is complete and the comparison shows that:
- At least one NN matches or beats LightGBM on per-window tasks (exercise, phase)
- This justifies the larger compute investment of training on raw signals

If Phase 1 shows NN trails LightGBM significantly on all tasks, raw-signal
training is unlikely to help dramatically — the bottleneck is the data, not
the input representation.

## What this command should do (when implemented)

1. Verify `scripts/train_cnn1d_raw.py`, `scripts/train_lstm_raw.py`,
   `scripts/train_cnn_lstm_raw.py`, `scripts/train_tcn_raw.py` exist.
2. Verify `RawWindowDataset` shape tests pass.
3. Run the 4 raw-input scripts on the same CV splits as Phase 1.
4. Aggregate into `runs/<ts>_raw-comparison/comparison.md` showing:
   - LightGBM baseline
   - 4 features-input NN (Phase 1)
   - 4 raw-input NN (Phase 2)
5. The interesting comparison is **per-arch features vs raw**:
   does CNN-LSTM-raw beat CNN-LSTM-features? That answers the question
   "is feature engineering or representation learning the bottleneck?"

## To implement

When ready to do Phase 2:

1. Tell the user: "I'll create scripts/train_<arch>_raw.py for each arch
   based on the existing pattern in scripts/train_<arch>.py."
2. Each new script imports `RawWindowDataset` instead of `WindowFeatureDataset`.
3. The model classes accept either input shape (features or raw) by reshaping
   inside `encode()`. This may need verification — currently they assume
   features. The model files in src/models/ may need a `mode='raw'` flag added.
4. Latency benchmark per architecture (p99, batch 1, 2-second window).
5. Write `runs/<ts>_raw-comparison/comparison.md` with all 9 columns:
   LightGBM + 4 features-NN + 4 raw-NN, across all 4 tasks.

## Status check

If user invokes this command before implementation:

```
Phase 2 (raw signals) is not yet implemented.

Currently built:
  ✓ RawWindowDataset (src/data/datasets.py) — shape-tested
  ✓ Multi-task model classes (src/models/*.py) — work on features

Currently missing:
  ✗ scripts/train_<arch>_raw.py for each architecture
  ✗ Possible model-class adjustments to handle raw input shape
  ✗ comparison_raw.py to aggregate Phase 2 results

If you want to proceed, run /train-nn first (Phase 1 features) and review
the results. Then say "implement Phase 2" and I'll build the raw scripts.
```
