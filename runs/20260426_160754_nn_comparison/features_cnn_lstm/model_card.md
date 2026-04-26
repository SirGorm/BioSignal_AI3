# Model Card — CNN-LSTM Multi-Task (features_cnn_lstm)

**Run:** `20260426_160754_nn_comparison`  
**Architecture:** DeepConvLSTM (CNN front-end + Bidirectional LSTM) with hard parameter sharing  
**Input variant:** Variant A — 34 engineered per-window features  
**Deployment status:** RESEARCH ONLY — bidirectional LSTM component is not causal  

---

## Architecture

DeepConvLSTM following Ordóñez & Roggen (2016), adapted to the feature-input
variant. Two Conv1d layers (1→32→64 channels) extract local feature interactions,
then a 1-layer BiLSTM processes the resulting feature "sequence". Mean-pooled LSTM
output is projected to repr_dim. Hard parameter sharing with 4 task heads (Caruana 1997).

**Causal note:** Non-causal due to bidirectional LSTM. Replace with unidirectional
LSTM for a deployable version. Marked `research_only`.

---

## Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (Loshchilov & Hutter 2019) |
| Conv channels | 64 |
| LSTM hidden | 64 |
| Batch size | 256 |
| Epochs | 20-30 (early stopping, patience 6) |

Feature normalization: IQR-based z-score on train split (Goodfellow et al. 2016).
CV: GroupKFold(5) on subject_id, reusing LightGBM baseline splits (Saeb et al. 2017).

---

## Phase 2 Results (3 seeds × 5 folds = 15 runs)

| Task | Metric | CNN-LSTM | LightGBM |
|------|--------|----------|----------|
| Exercise | macro-F1 | 0.349 +/- 0.088 | 0.427 |
| Phase | macro-F1 | 0.274 +/- 0.021 | 0.312 |
| Fatigue | MAE | 1.047 +/- 0.085 | 0.965 |
| Reps | MAE | 3.014 +/- 0.861 | 1.638 |

Second-best Phase 2 performer on phase (0.274, tied with TCN 0.275).

---

## Latency

| Percentile | Latency (ms) |
|------------|-------------|
| p99 | 0.72 ms |

---

## Known Limitations

1. Non-causal: research only.
2. CNN-LSTM on feature-input is architecturally forced — same limitation as other
   Variant A architectures.
3. Does not beat LightGBM on any task.

---

## References

- Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., & Fei-Fei, L. (2014). Large-scale video classification with convolutional neural networks. *CVPR 2014*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
- Ordóñez, F., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition. *Sensors*, 16(1), 115.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D.C., & Kording, K.P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5).
