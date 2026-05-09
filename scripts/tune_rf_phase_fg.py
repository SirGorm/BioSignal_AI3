"""Phase F (add-back) + Phase G (micro-tweaks) starting from the 8-feature subset
that Phase D converged on (F1=0.6411).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "runs/20260508_170518_extras_exercise_emg_acc_k5/features.parquet"
META_COLS = {"subject_id", "recording_id", "exercise"}

PATIENCE = 3
BEST_SO_FAR = 0.6411
DROPPED_BY_D = [
    "emg_x_mfl", "emg_x_msr", "emg_x_ls", "emg_x_ls4",
    "emg_rms", "emg_iemg", "emg_mnf", "emg_mdf", "emg_dimitrov",
    "emg_lscore", "emg_wamp",
    "acc_x_wamp", "acc_x_mfl", "acc_x_msr", "acc_x_ls", "acc_x_ls4",
]
BEST_HP = dict(n_estimators=800, max_depth=8)


def evaluate(X, y, groups, **kw) -> tuple[float, float]:
    splitter = GroupKFold(n_splits=5)
    fold_f1 = []
    use_et = kw.pop("_estimator", "rf") == "et"
    cw = kw.pop("class_weight", "balanced")  # default balanced, but allow override
    Cls = ExtraTreesClassifier if use_et else RandomForestClassifier
    for tr, te in splitter.split(X, y, groups):
        clf = Cls(n_jobs=-1, random_state=42, class_weight=cw, **kw)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        fold_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return float(np.mean(fold_f1)), float(np.std(fold_f1))


def main():
    full = pd.read_parquet(FEAT_PATH)
    feat_cols = [c for c in full.columns if c not in META_COLS]
    y = full["exercise"].values
    groups = full["subject_id"].values

    kept = [c for c in feat_cols if c not in DROPPED_BY_D]
    print(f"Starting subset ({len(kept)} features): {kept}")
    Xs = full[kept].values
    base_f1, base_std = evaluate(Xs, y, groups, **BEST_HP)
    print(f"Baseline (replicating Phase D): F1={base_f1:.4f}+/-{base_std:.3f}\n")

    # Phase F — add-back: for each dropped feature, try adding it
    print("=== Phase F -- add-back search ===")
    f_results = []
    for feat in DROPPED_BY_D:
        cols = kept + [feat]
        Xa = full[cols].values
        t0 = time.time()
        f1, std = evaluate(Xa, y, groups, **BEST_HP)
        d = f1 - base_f1
        f_results.append({"add": feat, "f1": f1, "std": std, "delta": d})
        marker = "*" if f1 > base_f1 else " "
        print(f"  {marker} add {feat:>22s}  F1={f1:.4f}+/-{std:.3f}  "
              f"d={d:+.4f}  ({time.time()-t0:.0f}s)")
    f_results.sort(key=lambda r: -r["delta"])
    print(f"  Top-3 helpful add-backs:")
    for r in f_results[:3]:
        print(f"    {r['add']:>22s}  d={r['delta']:+.4f}")
    f_best = max(r["f1"] for r in f_results) if f_results else base_f1
    print(f"  Phase F best: F1={f_best:.4f}\n")

    # If add-back helped, fold the helpful ones back in
    helpful = [r["add"] for r in f_results if r["delta"] > 0.001]
    cur_set = list(kept)
    if helpful:
        print(f"  Folding in {len(helpful)} helpful: {helpful}")
        # Add greedily, accept only those that compound
        for feat in [r["add"] for r in f_results if r["delta"] > 0]:
            test_cols = cur_set + [feat]
            Xa = full[test_cols].values
            f1, _ = evaluate(Xa, y, groups, **BEST_HP)
            if f1 > base_f1:
                base_f1 = f1
                cur_set = test_cols
                print(f"    + {feat}: F1={f1:.4f} (kept)")
            else:
                print(f"    - {feat}: F1={f1:.4f} (rejected)")
        Xs = full[cur_set].values
        print(f"  After greedy add-back: {len(cur_set)} features, F1={base_f1:.4f}")

    # Phase G — micro-tweaks on final subset
    print(f"\n=== Phase G -- micro-tweaks on final subset ({len(cur_set)} features) ===")
    print(f"Baseline: {base_f1:.4f}")
    g_trials = [
        ("balanced_subsample",       dict(**BEST_HP, class_weight="balanced_subsample")),
        ("class_weight=None",        dict(**BEST_HP, class_weight=None)),
        ("max_samples=0.7",          dict(**BEST_HP, max_samples=0.7)),
        ("max_samples=0.5",          dict(**BEST_HP, max_samples=0.5)),
        ("bootstrap=False",          dict(**BEST_HP, bootstrap=False)),
        ("ExtraTrees on subset",     dict(**BEST_HP, _estimator="et")),
        ("max_depth=7",              dict(n_estimators=800, max_depth=7)),
        ("max_depth=9",              dict(n_estimators=800, max_depth=9)),
        ("n_estimators=2000",        dict(n_estimators=2000, max_depth=8)),
        ("entropy criterion",        dict(**BEST_HP, criterion="entropy")),
        ("max_features=0.5",         dict(**BEST_HP, max_features=0.5)),
        ("max_features=None",        dict(**BEST_HP, max_features=None)),
    ]
    g_best = base_f1
    g_best_cfg = None
    no_improve = 0
    for label, kw in g_trials:
        kw_eval = dict(kw)  # evaluate() pops _estimator
        t0 = time.time()
        f1, std = evaluate(Xs, y, groups, **kw_eval)
        d = f1 - g_best
        marker = "*" if f1 > g_best else " "
        print(f"  {marker} {label:30s} F1={f1:.4f}+/-{std:.3f}  d={d:+.4f}  "
              f"({time.time()-t0:.0f}s)")
        if f1 > g_best:
            g_best = f1
            g_best_cfg = kw
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> stopped after {PATIENCE} non-improving trials")
                break

    print(f"\n========== SUMMARY ==========")
    print(f"  Phase D end:                  0.6411")
    print(f"  Phase F (add-back):           {f_best:.4f}")
    print(f"  After greedy add-back:        {base_f1:.4f}  ({len(cur_set)} features)")
    print(f"  Phase G (micro-tweaks):       {g_best:.4f}  cfg={g_best_cfg}")
    final = max(0.6411, base_f1, g_best)
    print(f"  Best overall:                 {final:.4f}")
    print(f"  Final feature set ({len(cur_set)}): {cur_set}")


if __name__ == "__main__":
    main()
