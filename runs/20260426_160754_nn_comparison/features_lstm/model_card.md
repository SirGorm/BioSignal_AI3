# Model Card — BiLSTM Multi-Task (features_lstm)

**Run:** `20260426_160754_nn_comparison`  
**Architecture:** Bidirectional LSTM with hard parameter sharing  
**Input variant:** Variant A — 34 engineered per-window features  
**Deployment status:** RESEARCH ONLY — bidirectional LSTM is not causal  

---

## Architecture

Input (B, 34) is reshaped to (B, 34, 1) — each feature becomes a "timestep"
with a single channel. A 2-layer BiLSTM (Hochreiter & Schmidhuber 1997) with
hidden=64 processes the feature sequence bidirectionally. The bidirectional
output is mean-pooled and projected to repr_dim via a linear layer.

Hard parameter sharing (Caruana 1997): one encoder, 4 task heads.

**Causal note:** Bidirectional = non-causal. CANNOT be used in real-time streaming.
A unidirectional LSTM variant would be causal; retrain with `bidirectional=False`
for deployment. Marked `research_only`.

---

## Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (Loshchilov & Hutter 2019) |
| Learning rate | Optuna-tuned |
| LSTM hidden | 64 |
| LSTM layers | 2 |
| Batch size | 256 |
| Epochs | 20-30 (early stopping, patience 6) |

Feature normalization: IQR-based z-score on train split (Goodfellow et al. 2016).
CV: GroupKFold(5) on subject_id, reusing LightGBM baseline splits (Saeb et al. 2017).

---

## Phase 1 Results (1 seed, 5 folds)

| Task | Metric | BiLSTM | LightGBM |
|------|--------|--------|----------|
| Exercise | macro-F1 | 0.290 | 0.427 |
| Phase | macro-F1 | 0.259 | 0.312 |
| Fatigue | MAE | 1.021 | 0.965 |
| Reps | MAE | 3.057 | 1.638 |

Not selected for Phase 2 (phase 1 mean rank 3.25, lowest of 4 architectures).

---

## Latency

| Percentile | Latency (ms) |
|------------|-------------|
| p99 | 0.80 ms |

Fast on CPU, but non-causal — irrelevant for deployment.

---

## Known Limitations

1. Bidirectional LSTM is non-causal: cannot stream in real time.
2. LSTM treating feature index as temporal sequence is architecturally forced:
   no genuine temporal structure exists in the 34-feature vector ordering.
3. Weakest Phase 1 performer on exercise (0.290 vs TCN 0.364).
4. For genuine temporal modeling, retrain on raw-signal Variant B input.

---

## References

- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
- Schuster, M., & Paliwal, K.K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.
