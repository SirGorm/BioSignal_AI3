"""Optuna RF tuning on the 4 Phinyomark descriptors x EMG + ACC.

Feature subset (fixed, 8 columns -- matches src/features/extra_features.py):
    emg_x_ls, emg_x_mfl, emg_x_msr, emg_x_wamp,
    acc_x_ls, acc_x_mfl, acc_x_msr, acc_x_wamp

  L-Score (LS)             = mean(log(|x|+1))               (Phinyomark et al. 2018)
  Maximum Fractal Length   = log10(sqrt(sum diff^2))        (Phinyomark et al. 2012)
  Mean Square Root (MSR)   = mean(sqrt(|x|))                (Phinyomark et al. 2018)
  Willison Amplitude (WAMP)= sum I(|dx|>thr)                (Willison 1964;
                                                             Phinyomark et al. 2012)

These are computed for both the EMG signal (forearm/biceps surface EMG, 2000 Hz
native) and the bandpass-filtered acc-magnitude (wrist accelerometer, 100 Hz).
Both feature sets are produced by src/features/extra_features.py.

CV: 5-fold GroupKFold on subject_id (matches scripts/tune_rf_extras.py).
    Per CLAUDE.md the canonical group is subject_name; pass --group-col
    subject_name if the parquet has it (Saeb et al. 2017).

Search:
    Optuna TPE sampler with median pruner, default 50 trials (Akiba et al. 2019).

Targets supported:
    --target exercise   -> 4-class macro-F1, RandomForestClassifier
    --target phase      -> macro-F1, RandomForestClassifier
    --target fatigue    -> R^2, RandomForestRegressor (column rpe_for_this_set)

Run:
    python scripts/tune_rf_phinyomark_optuna.py \
        --features runs/20260508_144721_extras_exercise/features.parquet \
        --target exercise --n-trials 50

References
----------
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. KDD '19.
- Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for
  multifunction myoelectric control. IEEE TBME, 40(1), 82-94.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature
  reduction and selection for EMG signal classification. Expert Systems
  with Applications, 39(8), 7420-7431.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017).
  The need to approximate the use-case in clinical machine learning.
  GigaScience, 6(5), gix019.
- [REF NEEDED: Breiman 2001 for Random Forest itself -- not in
  literature-references skill; ask user to add if cited in deliverables.]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

PHINYOMARK_FEATURES = [
    "emg_x_ls", "emg_x_mfl", "emg_x_msr", "emg_x_wamp",
    "acc_x_ls", "acc_x_mfl", "acc_x_msr", "acc_x_wamp",
]

TARGET_COLS = {
    "exercise": "exercise",
    "phase":    "phase_label",
    "fatigue":  "rpe_for_this_set",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True,
                   help="Path to features.parquet (must contain the 8 Phinyomark "
                        "columns + group column + target column).")
    p.add_argument("--target", choices=list(TARGET_COLS.keys()), default="exercise",
                   help="Prediction task (default: exercise).")
    p.add_argument("--target-col", type=str, default=None,
                   help="Override target column name (default: per --target).")
    p.add_argument("--group-col", type=str, default="subject_id",
                   help="Column to group folds on. CLAUDE.md prefers "
                        "'subject_name'; default 'subject_id' matches existing "
                        "RF scripts.")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir. Default: runs/<ts>_rf_phinyomark_<target>/.")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Joblib workers per RF (per fold). -1 = all cores.")
    return p.parse_args()


def load_data(features_path: Path, target: str, target_col: str | None,
              group_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]:
    """Load parquet, select 8 Phinyomark columns, return X / y / groups."""
    df = pd.read_parquet(features_path)

    missing = [c for c in PHINYOMARK_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"features.parquet is missing required columns: {missing}\n"
            f"Available: {sorted(df.columns)[:30]}...\n"
            f"Re-run feature extraction with src/features/extra_features.py "
            f"-- it produces all 8 columns under the *_x_{{ls,mfl,msr,wamp}} schema."
        )

    tcol = target_col or TARGET_COLS[target]
    if tcol not in df.columns:
        raise ValueError(f"Target column '{tcol}' not in parquet. "
                          f"Available: {sorted(df.columns)[:30]}...")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not in parquet. "
                          f"Try --group-col subject_id or recording_id.")

    keep = df[PHINYOMARK_FEATURES + [tcol, group_col]].dropna()
    n_dropped = len(df) - len(keep)
    if n_dropped:
        print(f"Dropped {n_dropped:,} rows with NaN in features/target/group.")

    X = keep[PHINYOMARK_FEATURES].to_numpy(dtype=float)
    y = keep[tcol].to_numpy()
    groups = keep[group_col].to_numpy()

    is_regression = target == "fatigue" or pd.api.types.is_float_dtype(keep[tcol])
    return X, y, groups, tcol, is_regression


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             rf_kwargs: dict, n_folds: int, is_regression: bool,
             n_jobs: int, seed: int, trial: optuna.Trial | None = None) -> tuple[float, float]:
    splitter = GroupKFold(n_splits=n_folds)
    fold_scores = []
    Cls = RandomForestRegressor if is_regression else RandomForestClassifier

    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        clf = Cls(n_jobs=n_jobs, random_state=seed, **rf_kwargs)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        if is_regression:
            score = float(r2_score(y[te], pred))
        else:
            score = float(f1_score(y[te], pred, average="macro", zero_division=0))
        fold_scores.append(score)
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def suggest_rf_hps(trial: optuna.Trial, is_regression: bool) -> dict:
    """Optuna search space for RF.

    Categorical sentinels:
      - max_depth=0 -> unlimited (None)
      - max_features='all' -> 1.0 (all features, makes RF act like bagging)
    """
    cfg = {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 2000, step=100),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
    }
    md = trial.suggest_int("max_depth", 0, 30)
    cfg["max_depth"] = None if md == 0 else md

    mf = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0])
    cfg["max_features"] = mf

    cfg["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
    if cfg["bootstrap"]:
        cfg["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

    if is_regression:
        cfg["criterion"] = trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error", "friedman_mse"]
        )
    else:
        cfg["criterion"] = trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        )
        cw = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", "none"]
        )
        cfg["class_weight"] = None if cw == "none" else cw

    return cfg


def baseline_score(X, y, groups, n_folds, is_regression, n_jobs, seed) -> float:
    """Reference score with project default RF config (n_estimators=800, max_depth=8)."""
    cfg = dict(n_estimators=800, max_depth=8)
    if not is_regression:
        cfg["class_weight"] = "balanced"
    s, _ = evaluate(X, y, groups, cfg, n_folds, is_regression, n_jobs, seed)
    return s


def main() -> None:
    args = parse_args()

    X, y, groups, tcol, is_regression = load_data(
        args.features, args.target, args.target_col, args.group_col
    )

    metric_name = "R^2" if is_regression else "macro-F1"
    n_groups = len(np.unique(groups))
    print(f"Loaded {len(X):,} rows x 8 features ({n_groups} groups, target='{tcol}')")
    print(f"Features: {PHINYOMARK_FEATURES}")
    print(f"Metric: {metric_name}")

    if n_groups < args.n_folds:
        print(f"WARNING: only {n_groups} groups, reducing folds to {n_groups}")
        args.n_folds = n_groups

    print("\nBaseline (project default n_estimators=800, max_depth=8) ...")
    t0 = time.time()
    base = baseline_score(X, y, groups, args.n_folds, is_regression,
                           args.n_jobs, args.seed)
    print(f"  baseline {metric_name} = {base:.4f}  ({time.time()-t0:.0f}s)")

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_rf_hps(trial, is_regression)
        mean_s, std_s = evaluate(X, y, groups, cfg, args.n_folds,
                                   is_regression, args.n_jobs, args.seed,
                                   trial=trial)
        trial.set_user_attr("std", std_s)
        return mean_s

    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print(f"\nRunning {args.n_trials} Optuna trials (TPE, median pruner)...")
    t_search_0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    t_search = time.time() - t_search_0

    best = study.best_trial
    print(f"\nDone in {t_search/60:.1f} min.")
    print(f"Best {metric_name} = {best.value:.4f}  (delta vs baseline: "
          f"{best.value - base:+.4f})")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    # Output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = Path("runs") / f"{ts}_rf_phinyomark_{args.target}"
    args.out.mkdir(parents=True, exist_ok=True)

    # Persist trial history
    history = []
    for t in study.trials:
        history.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
            "std": t.user_attrs.get("std"),
        })
    out_payload = {
        "features_path": str(args.features),
        "target": args.target,
        "target_col": tcol,
        "group_col": args.group_col,
        "n_folds": args.n_folds,
        "n_trials": args.n_trials,
        "n_rows": int(len(X)),
        "n_groups": int(n_groups),
        "metric": metric_name,
        "is_regression": bool(is_regression),
        "feature_subset": PHINYOMARK_FEATURES,
        "baseline_score": base,
        "best_score": best.value,
        "best_params": best.params,
        "search_time_s": t_search,
        "trials": history,
    }
    with open(args.out / "study.json", "w") as f:
        json.dump(out_payload, f, indent=2, default=str)

    # Minimal model_card stub (cite-only; user fills in narrative)
    card = f"""# Model card -- RF Phinyomark/Hudgins (8 features) -- {args.target}

## Setup
- Features: {", ".join(PHINYOMARK_FEATURES)}
- Target: `{tcol}` ({metric_name})
- CV: {args.n_folds}-fold GroupKFold on `{args.group_col}` (n_groups={n_groups})
- Optuna: TPE sampler + median pruner, {args.n_trials} trials, seed={args.seed}

## Result
- Baseline (n_estimators=800, max_depth=8): {metric_name} = {base:.4f}
- Best Optuna trial:                          {metric_name} = {best.value:.4f}
- Delta:                                      {best.value - base:+.4f}

## Best params
```json
{json.dumps(best.params, indent=2)}
```

## References
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD '19*.
- Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction myoelectric control. *IEEE Transactions on Biomedical Engineering*, 40(1), 82-94.
- Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications*, 39(8), 7420-7431.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019.
- [REF NEEDED: Breiman 2001 -- Random Forests algorithm citation; not in literature-references skill]
"""
    (args.out / "model_card.md").write_text(card)

    print(f"\nWrote {args.out}/study.json")
    print(f"Wrote {args.out}/model_card.md")


if __name__ == "__main__":
    main()
