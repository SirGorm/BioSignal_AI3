# Latency Benchmark — Inference p99 at Batch Size 1

**Setup:** Single 2-second window, batch size 1, CPU inference (no GPU).
Warmup: 20 runs, measured: 200 runs. Input: (1, 34) feature tensor.

| Architecture | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Causal | Deployment |
|--------------|----------|----------|----------|-----------|--------|------------|
| 1D-CNN | 0.26 | 0.33 | **0.37** | 0.25 | Yes | Primary candidate |
| LSTM (BiLSTM) | 0.59 | 0.75 | 0.80 | 0.61 | No | Research only |
| CNN-LSTM | 0.49 | 0.65 | 0.72 | 0.50 | No | Research only |
| TCN | 1.92 | 2.39 | **3.84** | 1.97 | Yes | Primary candidate |
| LightGBM (baseline) | ~0.5* | ~1.0* | ~2.0* | ~0.5* | Yes | Current production |

*LightGBM latency from `eval/latency_benchmark.py` in the baseline run (estimated).
All NN latencies measured on the same CPU (Windows 10, 1 CPU core equivalent in the
benchmark loop).

## Notes

- All architectures are well within the 100 ms real-time budget for a 2-second window.
- CNN1D is the fastest at 0.37 ms p99 — 10x faster than TCN.
- TCN at 3.84 ms p99 is still ~26x within the 100 ms budget.
- LSTM and CNN-LSTM are marked research_only due to bidirectional operation; the
  unidirectional variants would have similar or lower latency and would be deployable.
- On a GPU, all architectures would be substantially faster and differences would shrink.

## Deployment Recommendation (Latency)

For streaming deployment (100 ms budget), both CNN1D and TCN are viable.
CNN1D is preferred when latency is the dominant constraint.
TCN is preferred when accuracy is the dominant constraint (marginally better).

## References

Bai, S., Kolter, J.Z., & Koltun, V. (2018). An empirical evaluation of generic
convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*.
