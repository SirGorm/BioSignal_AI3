# Neural Network vs. LightGBM Comparison

**Run:** `20260426_160754_nn_comparison`  
**Date:** 2026-04-26  
**Baseline:** `runs/20260426_154705_default` (LightGBM, 5-fold GroupKFold, LOSO per fold)  
**NN strategy:** Phase 1 screening (5 folds, 1 seed) + Phase 2 depth (5 folds, 3 seeds) on top-3 variants  

---

## Compute-Budget Note (GPU Unavailable)

PyTorch 2.9.1+cpu — no CUDA device detected. CPU-adapted protocol applied:

- Active-set windows subsampled to 40,000 per fold (stratified by subject)
- Phase 1: 20 epochs max, early stopping patience 6
- Phase 2: 30 epochs max, 3 seeds × 5 folds = 15 runs per variant
- Optuna: 8 trials inner-CV (vs. 30 planned)
- Feature normalization: IQR-based z-score to handle acc_rms/acc_rep_band_power
  overflow values (~1e38) in raw features (Goodfellow et al. 2016)

**Variant B (raw signals) was not run** — estimated ~16 CPU-hours per architecture
for sequence processing. Variant A (engineered features) fully covers the comparison
question for the feature-input pathway.

---

## Architectures Tested

All architectures use **hard parameter sharing** (Caruana 1997): one shared encoder
produces a 64-128 dimensional representation, then 4 task-specific linear heads output
exercise logits, phase logits, fatigue scalar, and rep-count scalar.

| Variant | Architecture | Input | Causal | Deployment |
|---------|-------------|-------|--------|------------|
| features_cnn1d | 1D-CNN (3-layer) | 34 engineered features | Yes (streaming with causal padding) | Candidate |
| features_lstm | BiLSTM (2-layer) | 34 engineered features | No (bidirectional) | Research only |
| features_cnn_lstm | CNN + BiLSTM | 34 engineered features | No (bidirectional) | Research only |
| features_tcn | TCN (4 blocks, causal) | 34 engineered features | Yes | Primary candidate |

For the features-input variants, the "time" dimension is the feature index (length 34),
not a real temporal axis. This collapses CNN/TCN/LSTM to effective MLPs on feature
interactions, as documented in the slash-command spec (expected behavior).

---

## Phase 1 Screening Results (1 seed, 5 folds)

| Variant | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE | Mean Rank |
|---------|-------------|----------|-------------|----------|-----------|
| features_tcn | 0.364 | 0.255 | 1.005 | 2.864 | **1.75** |
| features_cnn1d | 0.336 | 0.265 | 1.141 | 2.986 | 2.25 |
| features_cnn_lstm | 0.319 | 0.259 | 1.067 | 3.039 | 2.75 |
| features_lstm | 0.290 | 0.259 | 1.021 | 3.057 | 3.25 |

**Top-3 selected for Phase 2:** TCN, CNN1D, CNN-LSTM

---

## Phase 2 Final Comparison (3 seeds x 5 folds = 15 runs)

| Model | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE | Latency p99 | Causal |
|-------|-------------|----------|-------------|----------|-------------|--------|
| **LightGBM (baseline)** | **0.427** | **0.312** | **0.965** | **1.638** | ~2 ms* | Yes |
| TCN (features) | 0.352 +/- 0.111 | 0.275 +/- 0.046 | 1.049 +/- 0.120 | 2.959 +/- 0.745 | 3.84 ms | Yes |
| CNN1D (features) | 0.339 +/- 0.076 | 0.247 +/- 0.033 | 1.017 +/- 0.069 | 2.984 +/- 0.844 | 0.37 ms | Yes |
| CNN-LSTM (features) | 0.349 +/- 0.088 | 0.274 +/- 0.021 | 1.047 +/- 0.085 | 3.014 +/- 0.861 | 0.72 ms | No** |

*LightGBM latency estimated from `eval/latency_benchmark.py` in the baseline run.
**CNN-LSTM uses bidirectional LSTM — research_only, not deployable for streaming.

### Delta vs. LightGBM

| Model | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |
|-------|-------------|----------|-------------|----------|
| TCN | -0.075 | -0.037 | +0.084 | +1.321 |
| CNN1D | -0.088 | -0.065 | +0.052 | +1.346 |
| CNN-LSTM | -0.078 | -0.038 | +0.082 | +1.376 |

**No neural network variant beats LightGBM on any task under the CPU-adapted protocol.**

---

## Statistical Tests (Paired t-test, 5 Folds, Low Power)

Using per-fold mean across 3 seeds vs. LightGBM per-fold scores.

| Task | TCN vs LGBM | t | p | Interpretation |
|------|-------------|---|---|----------------|
| Exercise F1 | -0.075 | -1.152 | 0.313 | Not significant (n=5, low power) |
| Fatigue MAE | +0.084 | +2.019 | 0.114 | Not significant (n=5, low power) |

**Caution:** With 5 folds, paired t-tests have very low statistical power (~0.15 for
medium effect size). The p-values should be interpreted as indicative, not conclusive.
Report effect sizes (differences) rather than p-values for the main finding.

---

## Per-Subject Analysis (TCN Winner, Fold 0, Subject: Gorm)

| Subject | Exercise F1 | Phase F1 | Fatigue MAE | Reps MAE |
|---------|-------------|----------|-------------|----------|
| Gorm (TCN) | 0.223 | 0.288 | 0.882 | 4.212 |
| Gorm (LGBM) | 0.420 | 0.324 | 0.897 | 2.373 |

No catastrophic failures (F1 not below 0.1). Gorm is the only test subject in fold 0;
per-subject breakdown for all 11 subjects would require per-subject LOSO (not computed
in this run due to compute budget). TCN fatigue MAE for Gorm (0.882) is better than
LightGBM (0.897) on this single subject.

---

## Root-Cause Analysis: Why NN Does Not Beat LightGBM

1. **Low data, high regularization needed:** 556k active-set windows, but subsampled
   to 40k per fold for CPU budget. LightGBM was trained on full data (~374k train
   windows). This 9x data disadvantage is the primary explanation.

2. **Feature-input pathway is an MLP on 34 features:** For Variant A, all four
   architectures effectively become feature-interaction MLPs. LightGBM's tree-based
   interactions are empirically competitive on tabular feature sets (Borisov et al. 2022).

3. **Low-data fatigue regime:** 156 RPE observations across 11 subjects. Any regression
   model struggles to beat the LightGBM MAE of 0.965 here.

4. **CPU budget limitation:** 20-30 epochs vs. the recommended 50+ for convergence.
   TCN Phase 1 (20 epochs) already shows convergence behavior; more epochs would likely
   close the gap on exercise classification.

5. **Reps MAE inflation:** The NN regression heads predict a continuous value per window
   (same window as exercise/phase), which is an ill-posed formulation since reps is
   naturally a per-set count. LightGBM uses set-level features for reps, giving it
   an inherent advantage. This is documented as a known architectural mismatch.

---

## Verdict

- **Best per-task (research):** LightGBM on all 4 tasks under these conditions
- **Best causal NN for deployment:** TCN (features_tcn)
  - Closest to LightGBM on exercise (0.352 vs 0.427)
  - Closest to LightGBM on phase (0.275 vs 0.312)
  - Fatigue within 8.7% of LGBM (1.049 vs 0.965)
  - Latency p99 = 3.84 ms (within 100 ms budget)

**Recommendation:** Keep LightGBM as production model. TCN is the NN candidate to
retrain on full data with GPU and 50+ epochs if NN is required for deployment.

---

## References

Bai, S., Kolter, J.Z., & Koltun, V. (2018). An empirical evaluation of generic
convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.

Borisov, V., Leemann, T., Sebler, K., et al. (2022). Deep neural networks and
tabular data: A survey. *IEEE TNNLS*.

Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
Learning* (2nd ed.). Springer.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*,
9(8), 1735-1780.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.

Ordóñez, F., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks
for multimodal wearable activity recognition. *Sensors*, 16(1), 115.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks.
*arXiv:1706.05098*.

Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). The need to
approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
