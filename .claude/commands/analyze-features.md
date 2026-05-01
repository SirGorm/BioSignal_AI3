---
description: Run LDA + ANOVA + mutual information feature relevance analysis. Outputs per-task and combined rankings with a markdown report.
allowed-tools: Bash, Read, Write
argument-hint: [--top-k 30]
---

# Analyze Features

Computes feature relevance per task using three complementary tests:
- Fisher LDA (linear class separation) — Fisher 1936
- ANOVA F-test (univariate parametric) — Hastie et al. 2009
- Mutual information (nonparametric, nonlinear) — Kraskov et al. 2004

Loads `window_features.parquet`, runs all three tests per task, writes per-task
rankings + combined ranking + markdown report to `runs/<ts>_feature-analysis/`.

## Steps

1. **Verify** that `data/labeled/<subject>/<session>/window_features.parquet`
   files exist. Halt if not — run `/label` and the
   `biosignal-feature-extractor` first.

2. **Run analysis**:
   ```bash
   python scripts/analyze_features.py --top-k 30
   ```

3. **Open the report** at `runs/<ts>_feature-analysis/feature_relevance_report.md`
   and look for sanity-checks:
   - EMG fatigue features (Dimitrov, MNF-slope) should top the `fatigue` list
     (Dimitrov et al. 2006)
   - Acc-based features should top the `exercise` list (Bonomi et al. 2009)
   - If `set_number` or `time_into_session` dominates `fatigue`, the model
     may be gaming session structure — flag this to the user

4. **Summarize in chat** (5-10 lines): top-5 features per task, any surprises,
   path to full report and `top_K_features.json`.

## Important: leakage warning

The output of this analysis is for **exploratory inspection only**. Using
the precomputed `top_K_features.json` directly in training is leakage-prone
because the list was computed on the full dataset including future test folds
(Saeb et al. 2017).

For publishable feature-reduction results, run:
```bash
python scripts/train_with_top_k.py --arch tcn --top-k 30 --leakage-safe
```
which re-runs the selection inside each CV fold's training set only.

## Output

```
Feature analysis complete: runs/<ts>_feature-analysis/

Top 5 features (combined ranking):
  1. emg_dimitrov            rank_min=1   rank_mean=2.3
  2. emg_mnf_slope            rank_min=2   rank_mean=3.5
  3. acc_dom_freq             rank_min=1   rank_mean=4.0
  4. emg_rms                  rank_min=3   rank_mean=4.8
  5. hrv_rmssd                rank_min=4   rank_mean=6.0

Per-task highlights:
  - fatigue:   EMG features dominate (expected per Dimitrov 2006)
  - exercise:  acc features dominate (expected per Bonomi 2009)
  - phase:     acc + EMG mixed
  - reps:      acc_dom_freq + acc_jerk

For leakage-safe ablation training:
  python scripts/train_with_top_k.py --arch tcn --top-k 30 --leakage-safe

Open runs/<ts>_feature-analysis/feature_relevance_report.md for details.
```
