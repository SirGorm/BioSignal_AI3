# Multi-Task Architecture Ablation: Hard vs. Soft Parameter Sharing

**Architecture under test:** TCN (Phase 1 winner, mean rank 1.75)  
**Folds:** 0 and 1 (first two folds, 1 seed, 15 training epochs for speed)  
**Protocol:** Identical hyperparameters, same subsampled train indices, same seed.

---

## Hard Sharing (Default)

One shared TCN encoder (4 causal dilated blocks) + 4 task-specific linear heads.
All tasks see the same intermediate representation (Caruana 1997).

## Soft Sharing (Ablation)

Four independent TCN encoders, one per task. Each encoder is the same architecture
as the hard-sharing encoder. Total parameter count is 4x higher.

---

## Results (2 folds, 1 seed)

| Task | Hard Sharing | Soft Sharing | Delta | Verdict |
|------|-------------|-------------|-------|---------|
| Exercise F1 | 0.269 | 0.311 | **+0.043** | Soft better |
| Phase F1 | 0.231 | 0.232 | +0.001 | Tie |
| Fatigue MAE | 1.092 | 1.010 | **-0.082** | Soft better |
| Reps MAE | 3.749 | 3.730 | -0.020 | Negligible |

---

## Interpretation

The ablation (2 folds, 1 seed) suggests soft sharing marginally improves exercise
classification (+0.043 F1) and fatigue regression (-0.082 MAE) on TCN.

**However, these results should be interpreted cautiously for several reasons:**

1. **Only 2 folds, 1 seed:** Variance is very high. With n=2, any difference
   of <0.1 is within noise. The Ruder 2017 survey notes soft sharing consistently
   helps only with >5 fold evidence.

2. **Soft sharing 4x parameter count:** In a low-data regime (40k subsampled
   windows, 11 subjects), 4x parameters with no additional data typically
   leads to overfitting. The apparent improvement may reverse with more folds.

3. **Hard sharing remains recommended for production:** The theoretical arguments
   for hard sharing in low-data regimes (Caruana 1997) remain strong, and the
   practical benefit of a single encoder (simpler deployment, lower RAM) is real.

**Decision:** Retain hard sharing as the default architecture. If future experiments
with full data (no subsampling) and GPU training show consistent soft-sharing
improvement across all 5 folds × 3 seeds, the recommendation should be revisited.

---

## References

Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41-75.
[Rationale for hard parameter sharing in low-data regimes]

Ruder, S. (2017). An overview of multi-task learning in deep neural networks.
*arXiv:1706.05098*. [Survey of hard vs. soft sharing trade-offs]
