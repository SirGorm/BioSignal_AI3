# Model Card — 1D-CNN Multi-Task (features_cnn1d)

**Run:** `20260426_160754_nn_comparison`  
**Architecture:** 1D Convolutional Neural Network with hard parameter sharing  
**Input variant:** Variant A — 34 engineered per-window features  
**Deployment status:** CAUSAL — deployment candidate for real-time streaming  

---

## Architecture

Three Conv1d layers (1→32→64→128 channels, k=5/3/3) with BatchNorm1d, ReLU, Dropout
and a final AdaptiveAvgPool1d(1) → Linear(128, repr_dim). The architecture follows
the 1D-CNN sensor-data approach of Yang et al. (2015), adapted for per-window
features by treating feature index as the "sequence" dimension.

Shared representation (64-dim, Optuna-tuned) feeds 4 task heads (Caruana 1997).

**Causal note:** In the features-input variant, the CNN sees all 34 features
simultaneously (symmetric padding). For deployment with raw-signal Variant B,
causal (left-only) padding must be used. Document as `causal=True` for streaming.

---

## Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (Loshchilov & Hutter 2019) |
| Learning rate | 0.00116 (Optuna-tuned) |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR |
| Epochs (Phase 2) | 30 (early stopping, patience 6) |
| Batch size | 256 |
| Dropout | 0.217 (Optuna-tuned) |
| repr_dim | 64 |

Feature normalization: IQR-based z-score on train split (Goodfellow et al. 2016).
Target normalization: per-fold z-score on regression targets.
CV: GroupKFold(5) on subject_id, reusing LightGBM baseline splits (Saeb et al. 2017).

---

## Phase 2 Results (3 seeds × 5 folds = 15 runs)

| Task | Metric | NN (features_cnn1d) | LightGBM baseline | Delta |
|------|--------|--------------------|--------------------|-------|
| Exercise | macro-F1 | 0.339 +/- 0.076 | 0.427 | -0.088 |
| Phase | macro-F1 | 0.247 +/- 0.033 | 0.312 | -0.065 |
| Fatigue | MAE | 1.017 +/- 0.069 | 0.965 | +0.052 |
| Reps | MAE | 2.984 +/- 0.844 | 1.638 | +1.346 |

---

## Latency

| Percentile | Latency (ms) |
|------------|-------------|
| p50 | 0.26 ms |
| p99 | **0.37 ms** |

Fastest of all 4 architectures. Preferred choice when latency budget is tight.

---

## Known Limitations

1. Features-input is degenerate — CNN operates on feature ordering, not time.
   Expected behavior per slash-command spec: collapses to MLP-equivalent.
2. CPU-only training limits convergence; GPU retraining recommended before production.
3. Beats dummy baseline (exercise 0.339 >> 0.121) but does not beat LightGBM.

---

## References

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD 2019*, 2623-2631.
- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
- Yang, J., Nguyen, M.N., San, P.P., Li, X.L., & Krishnaswamy, S. (2015). Deep convolutional neural networks on multichannel time series for human activity recognition. *IJCAI*, 15, 3995-4001.
