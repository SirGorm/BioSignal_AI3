"""Hyperparameter tuning for the extras_exercise RF -- uses cached features.parquet.

Optimizes mean fold macro-F1 on subject-wise k=5 GroupKFold. Stops after
N_PATIENCE trials without improvement over the running best.

Tunes (in order):
  Phase A -- RF hyperparameters
  Phase B -- feature-subset ablations (drop low-importance features)
  Phase C -- combined best-of-A x best-of-B

Run:
  python scripts/tune_rf_extras.py \
      --features runs/20260508_170518_extras_exercise_emg_acc_k5/features.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

PATIENCE = 3
META_COLS = {"subject_id", "recording_id", "exercise"}


def evaluate(X, y, groups, **rf_kwargs) -> tuple[float, float]:
    splitter = GroupKFold(n_splits=5)
    fold_f1 = []
    cls_kwargs = dict(rf_kwargs)
    use_et = cls_kwargs.pop("_estimator", "rf") == "et"
    Cls = ExtraTreesClassifier if use_et else RandomForestClassifier
    for tr, te in splitter.split(X, y, groups):
        clf = Cls(n_jobs=-1, random_state=42, class_weight="balanced",
                  **cls_kwargs)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        fold_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
    return float(np.mean(fold_f1)), float(np.std(fold_f1))


def run_phase(name, trials, X, y, groups, baseline_score):
    print(f"\n=== {name} ===")
    print(f"Baseline: {baseline_score:.4f}")
    best_score = baseline_score
    best_cfg = None
    no_improve = 0
    history = []
    for i, (label, cfg) in enumerate(trials):
        t0 = time.time()
        mean_f1, std_f1 = evaluate(X, y, groups, **cfg)
        elapsed = time.time() - t0
        delta = mean_f1 - best_score
        marker = "*" if mean_f1 > best_score else " "
        print(f"  [{i+1:2d}/{len(trials)}] {marker} {label:55s} "
              f"F1={mean_f1:.4f}+/-{std_f1:.3f}  d={delta:+.4f}  ({elapsed:.0f}s)")
        history.append({"label": label, "cfg": cfg, "f1": mean_f1, "f1_std": std_f1})
        if mean_f1 > best_score:
            best_score = mean_f1
            best_cfg = cfg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> stopped after {PATIENCE} non-improving trials")
                break
    return best_score, best_cfg, history


def main(features_path: Path, out_path: Path) -> None:
    full = pd.read_parquet(features_path)
    feat_cols = [c for c in full.columns if c not in META_COLS]
    X = full[feat_cols].values
    y = full["exercise"].values
    groups = full["subject_id"].values
    print(f"Loaded {len(full):,} rows x {len(feat_cols)} features "
          f"({full['subject_id'].nunique()} subjects)")

    # Baseline: replicate train_rf_extras_exercise.py default
    baseline_cfg = dict(n_estimators=400)
    baseline_score, _ = evaluate(X, y, groups, **baseline_cfg)
    print(f"\nBaseline (n_estimators=400, defaults): F1={baseline_score:.4f}")

    all_history = {"baseline": baseline_score, "phases": {}}
    best_score = baseline_score
    best_cfg = baseline_cfg.copy()

    # ----------------- Phase A: RF hyperparameters -----------------
    # Earlier sweep showed max_depth=10 helps. Centre this round there and
    # explore neighbours + combinations + ExtraTrees.
    phase_a_trials = [
        ("max_depth=10, n_estimators=800",           dict(n_estimators=800, max_depth=10)),
        ("max_depth=8, n_estimators=800",            dict(n_estimators=800, max_depth=8)),
        ("max_depth=12, n_estimators=800",           dict(n_estimators=800, max_depth=12)),
        ("max_depth=15, n_estimators=800",           dict(n_estimators=800, max_depth=15)),
        ("max_depth=10 + min_leaf=5",                dict(n_estimators=800, max_depth=10, min_samples_leaf=5)),
        ("max_depth=10 + min_leaf=2",                dict(n_estimators=800, max_depth=10, min_samples_leaf=2)),
        ("max_depth=10 + max_features=log2",         dict(n_estimators=800, max_depth=10, max_features="log2")),
        ("max_depth=10 + max_features=0.3",          dict(n_estimators=800, max_depth=10, max_features=0.3)),
        ("max_depth=10 + entropy",                   dict(n_estimators=800, max_depth=10, criterion="entropy")),
        ("max_depth=10, n_estimators=1500",          dict(n_estimators=1500, max_depth=10)),
        ("ExtraTrees max_depth=10",                  dict(n_estimators=800, max_depth=10, _estimator="et")),
        ("ExtraTrees max_depth=10, n_est=1500",      dict(n_estimators=1500, max_depth=10, _estimator="et")),
    ]
    a_best, a_cfg, a_hist = run_phase("Phase A -- RF hyperparameters",
                                        phase_a_trials, X, y, groups, best_score)
    all_history["phases"]["A"] = {"history": a_hist, "best_cfg": a_cfg, "best_f1": a_best}
    if a_cfg is not None:
        best_score = a_best
        best_cfg = a_cfg
        print(f"  -> Phase A best: F1={a_best:.4f}  cfg={a_cfg}")

    # ----------------- Phase B: feature-subset ablations -----------
    # Build importance ranking from a single fit using best Phase-A cfg.
    a_estimator = best_cfg.pop("_estimator", "rf") if best_cfg else "rf"
    Cls = ExtraTreesClassifier if a_estimator == "et" else RandomForestClassifier
    if best_cfg:
        best_cfg.pop("_estimator", None)
    rf_full = Cls(n_jobs=-1, random_state=42, class_weight="balanced",
                  **{k: v for k, v in best_cfg.items() if not k.startswith("_")})
    rf_full.fit(X, y)
    importances = sorted(zip(feat_cols, rf_full.feature_importances_),
                         key=lambda kv: -kv[1])
    print("\nFeature ranking (current best cfg):")
    for n, v in importances:
        print(f"    {n:>22s}  {v:.4f}")

    base_a_cfg = {**best_cfg}
    if a_estimator == "et":
        base_a_cfg["_estimator"] = "et"

    # Iterate over BOTTOM of the importance list (lowest first), incrementally
    # adding to drop_set. importances[::-1] is ascending; take the worst 12.
    drop_trials = []
    drop_set = []
    for n, _ in importances[::-1][:12]:
        drop_set.append(n)
        sub_cols = [c for c in feat_cols if c not in drop_set]
        col_idx = [feat_cols.index(c) for c in sub_cols]
        drop_trials.append(
            (f"drop bottom {len(drop_set):d} (latest: {drop_set[-1]})",
             {"_subset_idx": col_idx, **base_a_cfg})
        )

    def eval_with_subset(cfg):
        cfg = dict(cfg)
        idx = cfg.pop("_subset_idx", None)
        Xs = X[:, idx] if idx is not None else X
        return evaluate(Xs, y, groups, **cfg)

    print(f"\n=== Phase B -- feature-subset ablations ===")
    print(f"Baseline (Phase A best): {best_score:.4f}")
    no_improve = 0
    b_hist = []
    b_best = best_score
    b_best_cfg = None
    for i, (label, cfg) in enumerate(drop_trials):
        t0 = time.time()
        mean_f1, std_f1 = eval_with_subset(cfg)
        elapsed = time.time() - t0
        delta = mean_f1 - b_best
        marker = "*" if mean_f1 > b_best else " "
        print(f"  [{i+1:2d}/{len(drop_trials)}] {marker} {label:55s} "
              f"F1={mean_f1:.4f}+/-{std_f1:.3f}  d={delta:+.4f}  ({elapsed:.0f}s)")
        b_hist.append({"label": label, "f1": mean_f1, "f1_std": std_f1,
                       "n_features": len(cfg["_subset_idx"])})
        if mean_f1 > b_best:
            b_best = mean_f1
            b_best_cfg = cfg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> stopped after {PATIENCE} non-improving trials")
                break
    all_history["phases"]["B"] = {"history": b_hist,
                                    "best_f1": b_best,
                                    "best_n_features": (
                                        len(b_best_cfg["_subset_idx"])
                                        if b_best_cfg else len(feat_cols)
                                    )}

    # ----------------- Phase C: leave-one-out feature search ---------
    # For each individual feature, try dropping it. Catches mid-rank
    # features that might be hurting (the earlier emg_mnf finding).
    print(f"\n=== Phase C -- leave-one-out feature drops ===")
    print(f"Baseline (best so far): {b_best:.4f}")
    base_subset = (b_best_cfg.get("_subset_idx") if b_best_cfg
                    else list(range(len(feat_cols))))
    base_subset_cols = [feat_cols[i] for i in base_subset]
    c_hist = []
    c_best = b_best
    c_best_cfg = b_best_cfg
    leave_results = []
    for feat in base_subset_cols:
        sub_idx = [i for i in base_subset if feat_cols[i] != feat]
        cfg = {"_subset_idx": sub_idx, **base_a_cfg}
        t0 = time.time()
        mean_f1, std_f1 = eval_with_subset(cfg)
        elapsed = time.time() - t0
        delta = mean_f1 - c_best
        leave_results.append({"dropped": feat, "f1": mean_f1, "f1_std": std_f1,
                                "delta": delta})
        marker = "*" if mean_f1 > c_best else " "
        print(f"  {marker} drop {feat:>22s}  F1={mean_f1:.4f}+/-{std_f1:.3f}  "
              f"d={delta:+.4f}  ({elapsed:.0f}s)")
        if mean_f1 > c_best:
            c_best = mean_f1
            c_best_cfg = cfg
        c_hist.append({"label": f"loo: {feat}", "f1": mean_f1, "f1_std": std_f1})
    leave_results.sort(key=lambda r: -r["delta"])
    print("  Top-3 most-helpful drops:")
    for r in leave_results[:3]:
        print(f"    {r['dropped']:>22s}  d={r['delta']:+.4f}")
    all_history["phases"]["C"] = {"history": c_hist, "loo_results": leave_results,
                                    "best_f1": c_best}

    # ----------------- Phase D: greedy compounding drops ---------
    # Greedily drop the single most-helpful feature, then recompute
    # importances (or use leave-one-out delta) and repeat.
    print(f"\n=== Phase D -- greedy compounding drops ===")
    print(f"Baseline: {c_best:.4f}")
    cur_subset = (c_best_cfg.get("_subset_idx") if c_best_cfg
                   else list(range(len(feat_cols))))
    d_hist = []
    d_best = c_best
    d_best_cfg = c_best_cfg
    no_improve = 0
    step = 0
    while len(cur_subset) > 5 and no_improve < PATIENCE and step < 18:
        step += 1
        # leave-one-out within current subset
        deltas = []
        for i, idx in enumerate(cur_subset):
            test_subset = cur_subset[:i] + cur_subset[i+1:]
            cfg = {"_subset_idx": test_subset, **base_a_cfg}
            mean_f1, _ = eval_with_subset(cfg)
            deltas.append((feat_cols[idx], idx, mean_f1))
        deltas.sort(key=lambda r: -r[2])  # best first
        feat_drop, idx_drop, score = deltas[0]
        delta = score - d_best
        marker = "*" if score > d_best else " "
        print(f"  step {step}: {marker} drop {feat_drop:>22s}  "
              f"F1={score:.4f}  d={delta:+.4f}  ({len(cur_subset)-1} feats remain)")
        d_hist.append({"step": step, "dropped": feat_drop, "f1": score,
                        "n_remaining": len(cur_subset) - 1})
        if score > d_best:
            d_best = score
            d_best_cfg = {"_subset_idx": [i for i in cur_subset if i != idx_drop],
                            **base_a_cfg}
            cur_subset = [i for i in cur_subset if i != idx_drop]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> stopped after {PATIENCE} non-improving greedy steps")
                break
            cur_subset = [i for i in cur_subset if i != idx_drop]  # still drop, may improve later
    all_history["phases"]["D"] = {"history": d_hist, "best_f1": d_best}

    # ----------------- Phase E: re-tune RF HPs on best subset --------
    print(f"\n=== Phase E -- re-tune RF HPs on best subset ===")
    print(f"Baseline: {d_best:.4f}  ({len(d_best_cfg.get('_subset_idx', []))} features)")
    fixed_subset_idx = d_best_cfg.get("_subset_idx") if d_best_cfg else None
    Xs = X[:, fixed_subset_idx] if fixed_subset_idx is not None else X
    e_trials = [
        ("max_depth=6, n_est=800",       dict(n_estimators=800, max_depth=6)),
        ("max_depth=8, n_est=800",       dict(n_estimators=800, max_depth=8)),
        ("max_depth=10, n_est=800",      dict(n_estimators=800, max_depth=10)),
        ("max_depth=12, n_est=800",      dict(n_estimators=800, max_depth=12)),
        ("max_depth=15, n_est=800",      dict(n_estimators=800, max_depth=15)),
        ("max_depth=None, n_est=800",    dict(n_estimators=800)),
        ("max_depth=8, min_leaf=2",      dict(n_estimators=800, max_depth=8, min_samples_leaf=2)),
        ("max_depth=8, min_leaf=5",      dict(n_estimators=800, max_depth=8, min_samples_leaf=5)),
        ("max_depth=8, max_features=log2",
                                            dict(n_estimators=800, max_depth=8, max_features="log2")),
        ("max_depth=8, max_features=0.4", dict(n_estimators=800, max_depth=8, max_features=0.4)),
        ("max_depth=8, n_est=2000",      dict(n_estimators=2000, max_depth=8)),
        ("ExtraTrees max_depth=8",       dict(n_estimators=800, max_depth=8, _estimator="et")),
        ("ExtraTrees max_depth=10",      dict(n_estimators=800, max_depth=10, _estimator="et")),
    ]
    e_best, e_cfg, e_hist = run_phase("Phase E -- HP retune on subset",
                                        e_trials, Xs, y, groups, d_best)
    all_history["phases"]["E"] = {"history": e_hist, "best_cfg": e_cfg, "best_f1": e_best}

    # ----------------- Final summary -----------------
    final_score = max(baseline_score, a_best, b_best, c_best, d_best, e_best)
    if e_best == final_score:
        final_cfg = {"_subset_idx": fixed_subset_idx, **(e_cfg or {})}
    elif d_best == final_score:
        final_cfg = d_best_cfg
    elif c_best == final_score:
        final_cfg = c_best_cfg
    elif b_best == final_score:
        final_cfg = b_best_cfg
    else:
        final_cfg = a_cfg
    final_n_features = (len(final_cfg.get("_subset_idx"))
                         if final_cfg and "_subset_idx" in final_cfg
                         else len(feat_cols))
    print(f"\n========== SUMMARY ==========")
    print(f"  Baseline (n_estimators=400):  {baseline_score:.4f}")
    print(f"  Phase A (RF HPs):             {a_best:.4f}  cfg={a_cfg}")
    print(f"  Phase B (top-down drops):     {b_best:.4f}")
    print(f"  Phase C (leave-one-out):      {c_best:.4f}")
    print(f"  Phase D (greedy compound):    {d_best:.4f}")
    print(f"  Phase E (HP retune on subset):{e_best:.4f}  cfg={e_cfg}")
    print(f"  Best overall:                 {final_score:.4f}  ({final_n_features} features)")
    delta = final_score - baseline_score
    print(f"  Improvement over baseline:    {delta:+.4f} ({delta/baseline_score*100:+.1f}%)")
    if final_cfg and "_subset_idx" in final_cfg:
        kept = [feat_cols[i] for i in final_cfg["_subset_idx"]]
        dropped = [c for c in feat_cols if c not in kept]
        print(f"  Dropped features ({len(dropped)}): {dropped}")

    out_path.write_text(json.dumps({
        "baseline_f1": baseline_score,
        "phase_a_best_f1": a_best,
        "phase_a_best_cfg": a_cfg,
        "phase_b_best_f1": b_best,
        "phase_c_best_f1": c_best,
        "phase_d_best_f1": d_best,
        "phase_e_best_f1": e_best,
        "phase_e_best_cfg": e_cfg,
        "best_overall_f1": final_score,
        "best_overall_n_features": final_n_features,
        "phases": all_history["phases"],
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path,
                   help="Path to cached features.parquet from "
                        "train_rf_extras_exercise.py.")
    p.add_argument("--out", type=Path, default=Path("runs/tune_rf_extras_summary.json"))
    args = p.parse_args()
    main(args.features, args.out)
