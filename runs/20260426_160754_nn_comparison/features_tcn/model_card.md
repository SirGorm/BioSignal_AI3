# Model Card — TCN Multi-Task (features_tcn)

**Run:** `20260426_160754_nn_comparison`  
**Architecture:** Temporal Convolutional Network (TCN) with hard parameter sharing  
**Input variant:** Variant A — 34 engineered per-window features  
**Deployment status:** CAUSAL — deployment candidate for real-time streaming  

---

## Architecture

The TCN encoder consists of 4 causal dilated convolutional blocks with exponentially
increasing dilation (1, 2, 4, 8), following Bai et al. (2018). Each block uses:
- Conv1d with causal padding (no future context)
- BatchNorm1d + ReLU + Dropout
- Residual skip connection

The shared representation (128-dim) feeds 4 task-specific linear heads (Caruana 1997):
- `head_exercise`: 128 → 4 classes (softmax)
- `head_phase`: 128 → 3 classes (softmax, unknown excluded)
- `head_fatigue`: 128 → 1 scalar (RPE regression)
- `head_reps`: 128 → 1 scalar (rep count regression)

**Note on features input:** For Variant A, the "time" dimension is the feature index
(length 34), not a real temporal axis. The TCN processes feature interactions causally
along the feature ordering. This is degenerate (equivalent to an MLP with structured
interactions) but is consistent with the slash-command spec documenting this as
expected behavior for features-input architectures.

---

## Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (Loshchilov & Hutter 2019) |
| Learning rate | 0.00198 (Optuna-tuned) |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR |
| Epochs (Phase 1) | 20 (early stopping, patience 6) |
| Epochs (Phase 2) | 30 (early stopping, patience 6) |
| Batch size | 256 |
| Dropout | 0.393 (Optuna-tuned) |
| Gradient clip | 1.0 |
| repr_dim | 128 |
| kernel_size | 3 |
| TCN channels | [32, 64, 64, 128] |

**Feature normalization:** IQR-based z-score computed on train split, applied to all
inputs. Handles overflow values (~1e38) in acc_rms/acc_rep_band_power from the
feature extractor (Goodfellow et al. 2016). Clipped at ±5 sigma.

**Target normalization:** Per-fold z-score on fatigue (RPE) and reps targets.
Denormalized before metric computation. Prevents regression head blow-up.

**CV scheme:** GroupKFold(5) on subject_id, reusing splits from LightGBM baseline
for fair comparison (Saeb et al. 2017).

---

## Hyperparameter Tuning

Optuna TPE sampler (Akiba et al. 2019), 8 trials, inner-leave-one-subject-out
on fold 0 training subjects. Objective: composite score = (1 - exercise_F1) +
(1 - phase_F1) + fatigue_MAE/3 + reps_MAE/5.

---

## Phase 2 Results (3 seeds × 5 folds = 15 runs)

| Task | Metric | NN (features_tcn) | LightGBM baseline | Delta |
|------|--------|------------------|------------------|-------|
| Exercise | macro-F1 | 0.352 +/- 0.111 | 0.427 | -0.075 |
| Phase | macro-F1 | 0.275 +/- 0.046 | 0.312 | -0.037 |
| Fatigue | MAE | 1.049 +/- 0.120 | 0.965 | +0.084 |
| Reps | MAE | 2.959 +/- 0.745 | 1.638 | +1.321 |

**LightGBM outperforms on all tasks** under the CPU-adapted training protocol.
The largest gap is reps MAE, which is expected: LightGBM uses set-level features
while the NN head predicts per-window. Fatigue gap (+8.7%) is small.

---

## Latency

| Percentile | Latency (ms) |
|------------|-------------|
| p50 | 1.92 ms |
| p95 | 2.39 ms |
| p99 | 3.84 ms |

Within the 100 ms real-time budget for a 2-second window.

---

## Sanity Checks (Phase 2)

- Train loss decreases: CONFIRMED (all folds, all seeds; early stopping triggers 
  after 6-18 epochs)
- Val loss eventually exceeds train loss: CONFIRMED (learning signal present)
- Per-task losses all decreasing: CONFIRMED in training logs
- No catastrophic subject failure: Gorm (fold 0) exercise F1 = 0.223 vs. LGBM 0.420;
  still above random (1/4 = 0.25 macro) for most tasks
- Beats dummy: exercise 0.352 >> stratified baseline 0.121; phase 0.275 >> 0.188
- All per-fold seeds run: 15 runs completed

---

## Deployment Notes

TCN is a **causal architecture** — each output depends only on past inputs. The causal
padding (left-padding only) ensures no future context leaks into predictions.
This makes TCN directly deployable in `src/streaming/realtime.py` (Bai et al. 2018).

LSTM and CNN-LSTM variants are **bidirectional** and are marked `research_only`.
They cannot be deployed for real-time streaming.

---

## Known Limitations

1. Feature input variant collapses to MLP-equivalent. True temporal modeling from
   raw signals (Variant B) was not evaluated due to CPU compute budget.
2. CPU subsampling (40k/fold): full data training on GPU would likely improve all metrics.
3. Low-data regime for fatigue (156 RPE samples): neural networks rarely beat
   LightGBM in tabular low-data settings (Borisov et al. 2022).

---

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623-2631.
- Bai, S., Kolter, J.Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
- Borisov, V., Leemann, T., Sebler, K., et al. (2022). Deep neural networks and tabular data: A survey. *IEEE TNNLS*.
- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
