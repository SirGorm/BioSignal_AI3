"""Generate JASA + all 4 exercise fatigue dashboards for every recording,
then rank (recording, exercise) by how clearly they show fatigue.

Score per (recording, exercise) averages over the 3 sets:
  +1 if MNF slope  < 0  (spectral compression — fatigue, Cifrek 2009)
  +1 if MDF slope  < 0  (same; less noise-sensitive)
  +1 if EMG RMS slope > 0 (motor-unit recruitment, Luttmann 1996)
  +1 if Acc RMS slope < 0 (velocity loss, González-Badillo 2010)
  → max 4.0 per set, averaged over the 3 sets

Tie-broken by RPE progression (set3.rpe - set1.rpe; higher = clearer fatigue).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_fatigue_dashboard import (
    plot_dashboard_exercise,
    plot_jasa,
)
from scripts.plot_set_emg_acc_rms import load_window_features

EXERCISES = ["pullup", "squat", "deadlift", "benchpress"]


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return np.nan
    s, _ = np.polyfit(t[mask], y[mask], 1)
    return float(s)


def score_exercise(df: pd.DataFrame, exercise: str) -> dict | None:
    rows = df[(df["exercise"] == exercise) & (df["in_active_set"] == True)]  # noqa: E712
    if rows.empty:
        return None
    set_numbers = sorted(int(n) for n in rows["set_number"].dropna().unique())
    per_set = []
    rpes = []
    for set_n in set_numbers:
        s = rows[rows["set_number"] == float(set_n)]
        t = s["t_session_s"].to_numpy()
        mnf_s = _slope(t, s["emg_mnf"].to_numpy())
        mdf_s = _slope(t, s["emg_mdf"].to_numpy())
        rms_s = _slope(t, s["emg_rms"].to_numpy())
        acc_s = _slope(t, s["acc_rms"].to_numpy())
        score = 0
        score += int(mnf_s < 0) if not np.isnan(mnf_s) else 0
        score += int(mdf_s < 0) if not np.isnan(mdf_s) else 0
        score += int(rms_s > 0) if not np.isnan(rms_s) else 0
        score += int(acc_s < 0) if not np.isnan(acc_s) else 0
        per_set.append({
            "set": set_n, "score": score,
            "mnf_slope": mnf_s, "mdf_slope": mdf_s,
            "rms_slope": rms_s, "acc_slope": acc_s,
        })
        rpes.append(int(s["rpe_for_this_set"].iloc[0]))
    mean_score = float(np.mean([p["score"] for p in per_set]))
    rpe_prog = rpes[-1] - rpes[0] if len(rpes) >= 2 else 0
    return {
        "score": mean_score,
        "rpe_first": rpes[0] if rpes else None,
        "rpe_last": rpes[-1] if rpes else None,
        "rpe_progression": rpe_prog,
        "n_sets": len(per_set),
        "per_set": per_set,
    }


def main() -> None:
    aligned = sorted(Path("dataset_aligned").glob("recording_*"))
    recordings = [p.name.split("_")[1] for p in aligned
                  if (Path("data/labeled") / p.name / "window_features.parquet").exists()]
    print(f"Found {len(recordings)} labeled recordings: {recordings}\n")

    summary_rows = []
    for rid in recordings:
        print(f"=== recording {rid} ===")
        try:
            plot_jasa(rid)
        except Exception as e:
            print(f"  JASA failed: {e}")
        df = load_window_features(rid)
        subj = str(df["subject_name"].iloc[0]) if "subject_name" in df.columns else "?"
        for ex in EXERCISES:
            try:
                plot_dashboard_exercise(rid, ex)
                plot_dashboard_exercise(rid, ex, use_rel=True)
            except Exception as e:
                print(f"  {ex} dashboard skipped: {e}")
                continue
            score = score_exercise(df, ex)
            if score is None:
                continue
            summary_rows.append({
                "recording": rid,
                "subject": subj,
                "exercise": ex,
                "score": score["score"],
                "rpe_first": score["rpe_first"],
                "rpe_last": score["rpe_last"],
                "rpe_prog": score["rpe_progression"],
                "n_sets": score["n_sets"],
            })
        print()

    if not summary_rows:
        print("No data scored.")
        return

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        by=["score", "rpe_prog", "rpe_last"], ascending=[False, False, False]
    ).reset_index(drop=True)

    out = Path("inspections/fatigue_ranking.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    print("=" * 78)
    print("FATIGUE QUALITY RANKING -- (recording, exercise) sorted by score")
    print("=" * 78)
    print(f"{'rank':<5}{'rec':<6}{'subject':<14}{'exercise':<12}"
          f"{'score':<8}{'RPE':<10}{'dRPE':<6}{'sets':<5}")
    print("-" * 78)
    for i, r in summary.iterrows():
        rpe = f"{int(r['rpe_first'])}->{int(r['rpe_last'])}"
        print(f"{i+1:<5}{int(r['recording']):<6}{str(r['subject'])[:13]:<14}"
              f"{r['exercise']:<12}{r['score']:<8.2f}{rpe:<10}"
              f"{int(r['rpe_prog']):<+6}{int(r['n_sets']):<5}")
    print()
    print(f"Saved ranking: {out}")
    print()
    print("TOP 3 RECOMMENDATIONS:")
    for i, r in summary.head(3).iterrows():
        rid = str(int(r['recording'])).zfill(3)
        print(f"  {i+1}. recording_{rid} ({r['subject']}) -- "
              f"{r['exercise']}, score {r['score']:.2f}/4, "
              f"RPE {int(r['rpe_first'])}->{int(r['rpe_last'])}")
        print(f"     plot: inspections/recording_{rid}/"
              f"{r['exercise']}_fatigue_dashboard.png")


if __name__ == "__main__":
    main()
