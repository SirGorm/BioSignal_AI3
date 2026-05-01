---
name: feature-importance
description: Use when assessing which features contribute most, doing feature selection, or running ablations on smaller feature subsets. Covers LDA, ANOVA F-test, and mutual information per task; aggregation across tasks; and crucially the leakage-safe pattern (selection inside CV folds, not on the full dataset).
---

# Feature Importance and Selection

Three complementary tests answer "which features matter":

| Test | What it captures | Strengths | Weaknesses |
|------|------------------|-----------|------------|
| **Fisher LDA** | Linear class separation | Compact, theory-grounded | Assumes linearity, assumes Gaussian classes |
| **ANOVA F-test** | Univariate parametric relevance | Fast, well-understood | Assumes linearity, ignores interactions |
| **Mutual information** | Nonparametric, nonlinear | Captures complex relationships | Slower, sensitive to small samples |

Use **all three together**. If a feature ranks high on all three it's almost certainly relevant. If only one test highlights it, dig in — it might be a real nonlinearity (MI flags but LDA misses) or a statistical fluke.

## Per task vs combined ranking

Compute **per-task ranking** as the primary output. A feature that's strong for exercise classification (e.g., `acc_dom_freq`) may be useless for fatigue regression. Reporting only a combined rank obscures this.

Combined ranking is useful when picking ONE feature subset for the multi-task model. Use `rank_min` (best across tasks) if you want to keep features that excel at any task; use `rank_mean` if you want features broadly useful across all tasks.

## The leakage trap (critical)

When using these scores for actual feature selection, the analysis MUST run **inside each CV fold's training set only**. The common mistake:

```python
# WRONG — leakage
scores = compute_lda_anova_mi(X_full, y_full)
top_features = top_k_by_score(scores, k=30)
X_subset = X_full[:, top_features]
# now do CV on X_subset → test fold information already leaked into selection
```

```python
# RIGHT — leakage-safe
for train_idx, test_idx in cv.split(X_full, groups=subject_ids):
    scores = compute_lda_anova_mi(X_full[train_idx], y_full[train_idx])
    top_features = top_k_by_score(scores, k=30)
    X_train_sub = X_full[train_idx][:, top_features]
    X_test_sub  = X_full[test_idx][:, top_features]
    model.fit(X_train_sub, y_full[train_idx])
    metrics.append(model.score(X_test_sub, y_full[test_idx]))
```

The selected features may differ slightly between folds — that's expected and correct. Reporting "the top features are X, Y, Z" is fine; reporting metrics from a single full-dataset selection is not (Saeb et al. 2017).

## Two-mode pipeline in this project

`scripts/analyze_features.py` runs the full-dataset analysis for **exploration only** and writes a report. The output `top_K_features.json` is meant for sanity-checking — "do the features I expected to dominate actually dominate?" — not for publication metrics.

`scripts/train_with_top_k.py` has two modes:

- **Lazy** (`--feature-list path/to/top_K.json`): use the precomputed list. Documents the leakage warning in `model_card.md`. Use only for quick sanity checks.
- **Strict** (`--leakage-safe`): re-run selection per fold. The publishable approach. Slightly slower but correct.

## Recommended workflow

1. **First: full-feature baseline.** Run `/train-nn` without selection. This gives you the upper bound — if reduced-feature models are within 1-2% of this, the simpler model is preferable for deployment.

2. **Exploratory analysis.** Run `python scripts/analyze_features.py`. Open the report. Look for surprises:
   - Are EMG fatigue features (Dimitrov, MNF-slope) in the top for `fatigue`? They should be (Dimitrov et al. 2006).
   - Are accelerometer features in the top for `exercise`? They should be (Bonomi et al. 2009).
   - Does `set_number` or `time_into_session` dominate `fatigue`? If yes, the model may be gaming session structure rather than learning fatigue — investigate.

3. **Ablation (lazy mode):** quick sanity check that top-30 features get close to full performance. Use `--feature-list`.

4. **Final ablation (strict mode):** for the publishable comparison. Use `--leakage-safe`. Compare to step 1's full-feature baseline.

5. **Document everything in model_card.md** — both ranking criteria, both modes, with full citations.

## Decision rule for keeping reduced-feature model

Keep the reduced-feature model if **all** of:
- Exercise F1 within 0.02 of full-feature
- Phase F1 within 0.02 of full-feature
- Fatigue MAE within 0.15 RPE of full-feature
- Reps MAE within 0.1 of full-feature
- Per-subject worst-case has not gotten dramatically worse

Otherwise the features dropped were carrying real signal — keep them.

## References

For documentation in code comments and deliverables, cite from these (full entries in `literature-references` skill):

- **Fisher 1936** — original LDA
- **Hastie, Tibshirani & Friedman 2009** — *Elements of Statistical Learning* (LDA, F-tests, MI in chapters 3–4)
- **Guyon & Elisseeff 2003** — feature selection survey
- **Saeb et al. 2017** — leakage-aware CV (CRITICAL for the per-fold pattern)

These need to be added to the central `literature-references` skill before citing in `model_card.md` (the verify-references hook will block deliverables otherwise).
