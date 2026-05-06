"""Aggregate every v12 cv_summary.json into one wide CSV.

One row per (arch, window). Columns include macro-F1 AND balanced accuracy
for exercise/phase, plus MAE/r for fatigue and MAE for reps. Also pulls in
the v15 Random Forest baselines (per window) so the table is a single
side-by-side comparison vs. NN.

Output: runs/comparison_v12/summary_table.csv
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_DIR = RUNS / "comparison_v12"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = ["1s", "2s", "5s"]
ARCHS = [
    "multi-feat-mlp", "multi-feat-lstm",
    "multi-raw-cnn1d", "multi-raw-lstm", "multi-raw-cnn_lstm", "multi-raw-tcn",
    "fatigue-raw-tcn", "fatigue-raw-lstm",
]

COLS = [
    "arch", "window",
    "composite_score", "n_tasks_in_score",
    "exercise_f1_mean", "exercise_f1_std",
    "exercise_balacc_mean", "exercise_balacc_std",
    "phase_f1_mean", "phase_f1_std",
    "phase_balacc_mean", "phase_balacc_std",
    "fatigue_mae_mean", "fatigue_mae_std",
    "fatigue_r_mean", "fatigue_r_std",
    "reps_mae_mean", "reps_mae_std",
    "n_folds", "source",
]


# Same composite as scripts/train_optuna.py:score_summary — lower is better.
# Each enabled task gets equal weight; fatigue blends MAE + r 50/50 internally.
def composite_score(ex_f1, ph_f1, fa_mae, fa_r, rp_mae):
    parts = []
    n_tasks = 0
    if isinstance(ex_f1, (int, float)) and ex_f1 != "":
        parts.append(1.0 - float(ex_f1)); n_tasks += 1
    if isinstance(ph_f1, (int, float)) and ph_f1 != "":
        parts.append(1.0 - float(ph_f1)); n_tasks += 1
    fa_terms = []
    if isinstance(fa_mae, (int, float)) and fa_mae != "":
        fa_terms.append(min(float(fa_mae) / 3.0, 1.0))
    if isinstance(fa_r, (int, float)) and fa_r != "":
        fa_terms.append((1.0 - float(fa_r)) / 2.0)
    if fa_terms:
        parts.append(sum(fa_terms) / len(fa_terms)); n_tasks += 1
    if isinstance(rp_mae, (int, float)) and rp_mae != "":
        parts.append(min(float(rp_mae) / 1.0, 1.0)); n_tasks += 1
    if not parts:
        return "", 0
    return sum(parts) / len(parts), n_tasks


def fmt(d, key):
    if d is None or key not in d:
        return ("", "")
    sub = d[key]
    return (sub.get("mean", ""), sub.get("std", ""))


def load_rf_row(window: str):
    """Load v15 Random Forest baseline for a given window into the same row schema.

    RF metrics.json does NOT report balanced_accuracy — those cells stay empty.
    Pearson r mean/std are computed across the per-subject values.
    """
    rf_dir = ROOT / "runs" / f"optuna_clean_v15rf-w{window}"
    metrics_path = rf_dir / "metrics.json"
    if not metrics_path.exists():
        return {c: "" for c in COLS} | {"arch": "rf-baseline-v15", "window": window, "source": "MISSING"}
    m = json.loads(metrics_path.read_text())

    fa = m.get("fatigue", {})
    ex = m.get("exercise", {})
    ph = m.get("phase", {})
    rp = m.get("reps", {})

    pr_values = list((fa.get("pearson_r_per_subj") or {}).values())
    pr_mean = mean(pr_values) if pr_values else ""
    pr_std = pstdev(pr_values) if len(pr_values) > 1 else ""

    ex_f1 = ex.get("f1_mean", "")
    ph_f1 = ph.get("ml_f1_mean", "")
    fa_mae = fa.get("mae_mean", "")
    rp_mae = rp.get("ml_mae_mean", "")
    cs, nt = composite_score(ex_f1, ph_f1, fa_mae, pr_mean, rp_mae)

    return {
        "arch": "rf-baseline-v15",
        "window": window,
        "composite_score": cs,
        "n_tasks_in_score": nt,
        "exercise_f1_mean": ex_f1,
        "exercise_f1_std": ex.get("f1_std", ""),
        "exercise_balacc_mean": "",
        "exercise_balacc_std": "",
        "phase_f1_mean": ph_f1,
        "phase_f1_std": ph.get("ml_f1_std", ""),
        "phase_balacc_mean": "",
        "phase_balacc_std": "",
        "fatigue_mae_mean": fa_mae,
        "fatigue_mae_std": fa.get("mae_std", ""),
        "fatigue_r_mean": pr_mean,
        "fatigue_r_std": pr_std,
        "reps_mae_mean": rp_mae,
        "reps_mae_std": rp.get("ml_mae_std", ""),
        "n_folds": m.get("n_subjects", ""),
        "source": str(metrics_path.relative_to(ROOT)).replace("\\", "/"),
    }


def main():
    rows = []
    # RF baseline first so it sits at the top of the CSV
    for w in WINDOWS:
        rows.append(load_rf_row(w))
    for arch in ARCHS:
        for w in WINDOWS:
            run_dir = RUNS / f"optuna_clean_v12eqw-w{w}-{arch}"
            cv_path = next(iter((run_dir / "phase2").rglob("cv_summary.json")), None) if run_dir.exists() else None
            if cv_path is None:
                rows.append({c: "" for c in COLS} | {"arch": arch, "window": w, "source": "MISSING"})
                continue
            summary = json.loads(cv_path.read_text())["summary"]

            ex_f1 = fmt(summary.get("exercise"), "f1_macro")
            ex_ba = fmt(summary.get("exercise"), "balanced_accuracy")
            ph_f1 = fmt(summary.get("phase"), "f1_macro")
            ph_ba = fmt(summary.get("phase"), "balanced_accuracy")
            fa_mae = fmt(summary.get("fatigue"), "mae")
            fa_r = fmt(summary.get("fatigue"), "pearson_r")
            rp_mae = fmt(summary.get("reps"), "mae")

            n = ""
            for task in ("exercise", "phase", "fatigue", "reps"):
                td = summary.get(task) or {}
                for v in td.values():
                    if isinstance(v, dict) and "n" in v:
                        n = v["n"]
                        break
                if n != "":
                    break

            cs, nt = composite_score(ex_f1[0], ph_f1[0], fa_mae[0], fa_r[0], rp_mae[0])
            rows.append({
                "arch": arch, "window": w,
                "composite_score": cs, "n_tasks_in_score": nt,
                "exercise_f1_mean": ex_f1[0], "exercise_f1_std": ex_f1[1],
                "exercise_balacc_mean": ex_ba[0], "exercise_balacc_std": ex_ba[1],
                "phase_f1_mean": ph_f1[0], "phase_f1_std": ph_f1[1],
                "phase_balacc_mean": ph_ba[0], "phase_balacc_std": ph_ba[1],
                "fatigue_mae_mean": fa_mae[0], "fatigue_mae_std": fa_mae[1],
                "fatigue_r_mean": fa_r[0], "fatigue_r_std": fa_r[1],
                "reps_mae_mean": rp_mae[0], "reps_mae_std": rp_mae[1],
                "n_folds": n,
                "source": str(cv_path.relative_to(ROOT)).replace("\\", "/"),
            })

    print("\n=== Best complete model (lowest composite, all 4 tasks) ===")
    print("Composite = mean over enabled tasks of:")
    print("  exercise: 1 - F1 | phase: 1 - F1 | fatigue: 0.5*(MAE/3) + 0.5*(1-r)/2 | reps: MAE")
    print("(same formula as Optuna's score_summary; lower = better)\n")
    full_models = [
        r for r in rows
        if isinstance(r.get("composite_score"), (int, float)) and r.get("n_tasks_in_score") == 4
    ]
    full_models.sort(key=lambda r: r["composite_score"])
    print(f"{'rank':<5}{'arch':<22}{'win':<5}{'composite':<12}{'ex_F1':<8}{'ph_F1':<8}{'fa_MAE':<8}{'fa_r':<8}{'rp_MAE':<8}")
    print("-" * 84)
    for i, r in enumerate(full_models, 1):
        def s(k):
            v = r.get(k, "")
            return f"{float(v):.3f}" if isinstance(v, (int, float)) and v != "" else "—"
        print(f"{i:<5}{r['arch']:<22}{r['window']:<5}{s('composite_score'):<12}"
              f"{s('exercise_f1_mean'):<8}{s('phase_f1_mean'):<8}"
              f"{s('fatigue_mae_mean'):<8}{s('fatigue_r_mean'):<8}{s('reps_mae_mean'):<8}")

    out_csv = OUT_DIR / "summary_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out_csv} ({len(rows)} rows)")
    # Pretty print to stdout for quick eyeballing
    w = max(len(r["arch"]) for r in rows) + 2
    hdr = f"{'arch':<{w}}{'win':<5}{'ex_F1':<14}{'ex_balAcc':<14}{'ph_F1':<14}{'ph_balAcc':<14}{'fa_MAE':<14}{'fa_r':<14}{'rp_MAE':<14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def cell(m, s):
            if m == "" or m is None:
                return "—"
            try:
                return f"{float(m):.3f}±{float(s):.3f}"
            except (TypeError, ValueError):
                return "—"
        print(
            f"{r['arch']:<{w}}{r['window']:<5}"
            f"{cell(r['exercise_f1_mean'], r['exercise_f1_std']):<14}"
            f"{cell(r['exercise_balacc_mean'], r['exercise_balacc_std']):<14}"
            f"{cell(r['phase_f1_mean'], r['phase_f1_std']):<14}"
            f"{cell(r['phase_balacc_mean'], r['phase_balacc_std']):<14}"
            f"{cell(r['fatigue_mae_mean'], r['fatigue_mae_std']):<14}"
            f"{cell(r['fatigue_r_mean'], r['fatigue_r_std']):<14}"
            f"{cell(r['reps_mae_mean'], r['reps_mae_std']):<14}"
        )


if __name__ == "__main__":
    main()
