---
name: gradient-boosting-baselines
description: Use when training tabular gradient boosting baselines for the strength-RT project. The project uses BOTH LightGBM and XGBoost as parallel baselines so neural networks must beat both to claim improvement. Both run on per-window features with subject-wise CV; both are reported side-by-side in model_card.md.
---

# Gradient Boosting Baselines (LightGBM + XGBoost)

This project uses **two gradient boosting baselines** rather than one. NN models must beat both to claim improvement. Reasons:

1. **Stronger paper narrative**: "we compare against two established gradient boosting baselines" pre-empts the reviewer question "why not the other one?"
2. **Sanity check built in**: significant divergence between the two suggests hyperparameter tuning matters more than algorithm choice — useful diagnostic
3. **Marginal cost**: both train quickly on ~84k windows; total compute increase ~20%

References:
- **Ke et al. 2017** — LightGBM (histogram-based, leaf-wise growth)
- **Chen & Guestrin 2016** — XGBoost (block-based, level-wise growth)
- **Friedman 2001** — original gradient boosting

## Default hyperparameters per task

The two libraries use different parameter names but tune analogous concepts. Defaults are conservative for the low-data regime (~24 subjects, ~216 RPE rows).

### LightGBM

```python
from lightgbm import LGBMRegressor, LGBMClassifier

# Classification (exercise, phase)
lgbm_clf = LGBMClassifier(
    objective='multiclass',
    n_estimators=300,
    max_depth=6,
    num_leaves=15,           # smaller than default 31 for low-data
    learning_rate=0.05,
    min_child_samples=10,    # critical for small N
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Regression (fatigue, reps)
lgbm_reg = LGBMRegressor(
    objective='regression_l1',  # MAE — robust to RPE outliers
    n_estimators=300,
    max_depth=5,
    num_leaves=15,
    learning_rate=0.05,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
```

### XGBoost

```python
from xgboost import XGBRegressor, XGBClassifier

# Classification
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,     # analogous to lightgbm's min_child_samples
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='hist',      # faster, similar to LightGBM's approach
)

# Regression
xgb_reg = XGBRegressor(
    objective='reg:absoluteerror',  # L1 / MAE
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
)
```

## Parameter equivalence cheat-sheet

| Concept | LightGBM | XGBoost |
|---------|----------|---------|
| Tree depth | `max_depth` (-1 = no limit) | `max_depth` |
| Leaves per tree | `num_leaves` | implicit via depth |
| Min samples per leaf | `min_child_samples` | `min_child_weight` |
| Learning rate | `learning_rate` | `learning_rate` (alias `eta`) |
| L1 reg | `reg_alpha` | `reg_alpha` (alias `alpha`) |
| L2 reg | `reg_lambda` | `reg_lambda` (alias `lambda`) |
| Row sampling | `subsample` (with `bagging_freq`) | `subsample` |
| Feature sampling | `colsample_bytree` | `colsample_bytree` |
| Verbosity | `verbose=-1` | `verbosity=0` |

Set the same `random_state` for both so seed effects are comparable.

## Subject-wise CV pattern (identical for both)

```python
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# Build a sklearn pipeline so preprocessing is fit per fold (no leakage)
def make_lgbm_pipeline(task='classification'):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),  # robust to physiological outliers
        ('model',   lgbm_clf if task == 'classification' else lgbm_reg),
    ])

def make_xgb_pipeline(task='classification'):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
        ('model',   xgb_clf if task == 'classification' else xgb_reg),
    ])

cv = LeaveOneGroupOut() if n_subjects <= 24 else GroupKFold(5)
for train_idx, test_idx in cv.split(X, y, groups=subject_ids):
    # Train both side by side
    lgbm = make_lgbm_pipeline().fit(X[train_idx], y[train_idx])
    xgb  = make_xgb_pipeline().fit(X[train_idx], y[train_idx])
    # Evaluate both on the same test fold
    ...
```

## Hyperparameter tuning

For each library separately with Optuna, inside an inner GroupKFold so the
test fold never participates. Budget: ~50 trials per (library × task) is
realistic. Tune on the same metric you'll report (F1-macro for classification,
MAE for regression).

## Output for model_card.md

Always report **both libraries side by side**:

```markdown
| Task | Metric | LightGBM | XGBoost | Δ |
|------|--------|----------|---------|---|
| Exercise | F1-macro | 0.86 ± 0.04 | 0.85 ± 0.05 | +0.01 |
| Phase    | F1-macro | 0.91 ± 0.02 | 0.92 ± 0.02 | -0.01 |
| Fatigue  | MAE      | 0.82 ± 0.10 | 0.81 ± 0.11 | +0.01 |
| Reps     | MAE      | 0.31 ± 0.05 | 0.33 ± 0.06 | -0.02 |
```

If the two differ by more than ~0.05 on any metric, investigate:
- Hyperparameter tuning likely insufficient on one of them
- Or one library is mishandling something (e.g., NaN handling, categorical
  encoding) — check feature engineering
- Or the dataset has properties that genuinely favor one (rare, but possible)

The "consensus baseline" for NN comparison is the **better of the two** per
task. NN must beat that, not just one of them.

## When to use which

- **LightGBM only**: prototyping, when you need iteration speed
- **XGBoost only**: paper deadline, conservative reviewer
- **Both** (default for this project): final research results, sensitivity
  analysis, demonstrating robustness

## References

When documenting baseline choices in `model_card.md`, cite from these (full
entries in `literature-references` skill):

- **Ke et al. 2017** — LightGBM
- **Chen & Guestrin 2016** — XGBoost
- **Friedman 2001** — gradient boosting foundations
- **Saeb et al. 2017** — subject-wise CV for both baselines
