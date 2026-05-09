"""Explore whether the wrist accelerometer carries a measurable fatigue
signature across sets.

Strategy
--------
Per rep (rep window = [rep_start_unix, next_rep_unix or set_end_unix]):
  - peak_acc       : max(acc_mag)
  - mean_acc       : mean(acc_mag)
  - rep_duration_s : window length in seconds
  - dom_freq_hz    : dominant frequency (Welch)
  - jerk_rms       : RMS of d(acc_mag)/dt

Per set: linear regression of each feature vs rep index -> slope (1 number
per set). Slope is the within-set fatigue signature
(Gonzalez-Badillo & Sanchez-Medina 2010).

Outputs (results/acc_fatigue/):
  - per_rep.parquet            : raw per-rep features
  - per_set_slopes.parquet     : per-set slopes + RPE
  - spearman_per_exercise.csv  : rho/p per (feature, exercise)
  - slope_vs_rpe.png           : 4 features x 4 exercises scatter
  - subject_heatmap_<feat>.png : subject x set heatmap (z-scored)
  - rep_curves_<exercise>.png  : rep-by-rep mean_acc for sets 1 / mid / last

Usage
-----
    python scripts/explore_acc_fatigue.py
    python scripts/explore_acc_fatigue.py --labeled-root data/labeled \
                                          --aligned-root dataset_aligned \
                                          --out results/acc_fatigue
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal as sps
from scipy import stats as sst


REP_RE = re.compile(r"Set:(\d+)_Rep:(\d+)")
SET_END_RE = re.compile(r"Set_(\d+)_End|Set:(\d+)_End")
FS_HZ = 100.0  # aligned_features grid

FEATURES = ["peak_acc", "mean_acc", "rep_duration_s", "dom_freq_hz", "jerk_rms"]
EXERCISES = ["squat", "deadlift", "benchpress", "pullup"]


def load_rep_intervals(markers_path: Path) -> pd.DataFrame:
    """Return DataFrame [set_number, rep_index, t_start_unix, t_end_unix]."""
    with markers_path.open() as f:
        m = json.load(f)["markers"]
    reps, set_ends = [], {}
    for entry in m:
        label = entry["label"]
        rm = REP_RE.fullmatch(label)
        if rm:
            reps.append(
                dict(
                    set_number=int(rm.group(1)),
                    rep_index=int(rm.group(2)),
                    t_start_unix=float(entry["unix_time"]),
                )
            )
            continue
        em = SET_END_RE.fullmatch(label)
        if em:
            n = int(em.group(1) or em.group(2))
            set_ends[n] = float(entry["unix_time"])
    if not reps:
        return pd.DataFrame(columns=["set_number", "rep_index",
                                     "t_start_unix", "t_end_unix"])

    df = pd.DataFrame(reps).sort_values(["set_number", "rep_index"])
    # End-of-rep: next rep in same set, else set_end
    t_end = []
    for (s, _), grp in df.groupby(["set_number", "rep_index"]):
        pass  # placeholder
    # Vectorized: compute t_end via shift within set, fill NaN with set end
    df["t_end_unix"] = df.groupby("set_number")["t_start_unix"].shift(-1)
    last_mask = df["t_end_unix"].isna()
    df.loc[last_mask, "t_end_unix"] = df.loc[last_mask, "set_number"].map(set_ends)
    # If no set_end marker for last rep, fall back to start + median rep length
    median_dur = (df["t_end_unix"] - df["t_start_unix"]).median()
    df["t_end_unix"] = df["t_end_unix"].fillna(
        df["t_start_unix"] + (median_dur if pd.notna(median_dur) else 3.0)
    )
    return df.reset_index(drop=True)


def per_rep_features(acc: np.ndarray, fs: float) -> dict:
    """Compute the per-rep feature set on an acc_mag window."""
    n = acc.size
    if n < 8:
        return {k: np.nan for k in
                ("peak_acc", "mean_acc", "rep_duration_s",
                 "dom_freq_hz", "jerk_rms")}

    peak = float(np.max(acc))
    mean = float(np.mean(acc))
    duration = n / fs

    # dominant frequency via Welch (segment at most window length)
    nperseg = max(16, min(n, 64))
    f, pxx = sps.welch(acc - acc.mean(), fs=fs, nperseg=nperseg)
    if pxx.size and pxx.sum() > 0:
        dom_freq = float(f[np.argmax(pxx)])
    else:
        dom_freq = np.nan

    # jerk = d(acc)/dt
    jerk = np.diff(acc) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else np.nan

    return dict(
        peak_acc=peak,
        mean_acc=mean,
        rep_duration_s=duration,
        dom_freq_hz=dom_freq,
        jerk_rms=jerk_rms,
    )


def extract_recording(labeled_path: Path, aligned_root: Path) -> pd.DataFrame:
    rec_id = labeled_path.parent.name
    feats = pd.read_parquet(
        labeled_path,
        columns=["t_unix", "acc_mag", "set_number", "exercise",
                 "rpe_for_this_set", "subject_id"],
    )
    if feats.empty:
        return pd.DataFrame()

    subject = feats["subject_id"].iloc[0]
    feats = feats[["t_unix", "acc_mag", "set_number", "exercise",
                   "rpe_for_this_set"]].sort_values("t_unix").reset_index(drop=True)

    markers = aligned_root / rec_id / "markers.json"
    if not markers.exists():
        print(f"[skip] no markers for {rec_id}")
        return pd.DataFrame()
    reps = load_rep_intervals(markers)
    if reps.empty:
        print(f"[skip] no rep markers for {rec_id}")
        return pd.DataFrame()

    t = feats["t_unix"].to_numpy()
    a = feats["acc_mag"].to_numpy()

    rows = []
    for r in reps.itertuples(index=False):
        i0 = int(np.searchsorted(t, r.t_start_unix, side="left"))
        i1 = int(np.searchsorted(t, r.t_end_unix, side="left"))
        if i1 - i0 < 8:
            continue
        win = a[i0:i1]
        feat = per_rep_features(win, fs=FS_HZ)

        # set-level metadata at the rep midpoint
        mid_idx = (i0 + i1) // 2
        meta = feats.iloc[mid_idx]

        rows.append(
            dict(
                recording_id=rec_id,
                subject_id=subject,
                set_number=int(r.set_number),
                rep_index=int(r.rep_index),
                exercise=str(meta["exercise"]) if pd.notna(meta["exercise"]) else None,
                rpe=float(meta["rpe_for_this_set"]) if pd.notna(meta["rpe_for_this_set"]) else np.nan,
                **feat,
            )
        )

    return pd.DataFrame(rows)


def compute_set_slopes(per_rep: pd.DataFrame) -> pd.DataFrame:
    """Per (recording, set) linear slope of each feature vs rep_index."""
    out = []
    for (rec, s), grp in per_rep.groupby(["recording_id", "set_number"]):
        if grp["rep_index"].nunique() < 3:
            continue
        row = dict(
            recording_id=rec,
            subject_id=grp["subject_id"].iloc[0],
            set_number=int(s),
            exercise=grp["exercise"].iloc[0],
            rpe=float(grp["rpe"].iloc[0]),
            n_reps=int(grp["rep_index"].nunique()),
        )
        x = grp["rep_index"].to_numpy(dtype=float)
        for feat in FEATURES:
            y = grp[feat].to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 3:
                row[f"{feat}_slope"] = np.nan
                row[f"{feat}_pct_change"] = np.nan
                continue
            slope, intercept, *_ = sst.linregress(x[mask], y[mask])
            row[f"{feat}_slope"] = slope
            # relative change rep1 -> repN as percentage of rep1 prediction
            r1 = intercept + slope * x[mask].min()
            rN = intercept + slope * x[mask].max()
            row[f"{feat}_pct_change"] = (
                100.0 * (rN - r1) / r1 if r1 not in (0, np.nan) else np.nan
            )
        out.append(row)
    return pd.DataFrame(out)


def spearman_table(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ex in EXERCISES:
        sub = slopes[slopes["exercise"] == ex]
        for feat in FEATURES:
            col = f"{feat}_slope"
            x = sub[col].to_numpy(dtype=float)
            y = sub["rpe"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 4:
                rho, p, n = np.nan, np.nan, int(mask.sum())
            else:
                rho, p = sst.spearmanr(x[mask], y[mask])
                n = int(mask.sum())
            rows.append(dict(exercise=ex, feature=feat, n=n, rho=rho, p=p))
    return pd.DataFrame(rows)


def plot_slope_vs_rpe(slopes: pd.DataFrame, out_path: Path) -> None:
    long = slopes.melt(
        id_vars=["recording_id", "subject_id", "set_number", "exercise", "rpe"],
        value_vars=[f"{f}_slope" for f in FEATURES],
        var_name="feature", value_name="slope",
    )
    long["feature"] = long["feature"].str.replace("_slope", "", regex=False)
    long = long[long["exercise"].isin(EXERCISES)]

    g = sns.FacetGrid(
        long, row="feature", col="exercise",
        row_order=FEATURES, col_order=EXERCISES,
        sharey="row", sharex=False, height=2.6, aspect=1.2,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.regplot, x="rpe", y="slope",
        scatter_kws=dict(alpha=0.55, s=22), line_kws=dict(color="black"),
        ci=None,
    )
    for ax in g.axes.flat:
        ax.axhline(0, color="grey", lw=0.8, ls=":")
    g.set_axis_labels("RPE (1-10)", "within-set slope")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.figure.suptitle(
        "wrist acc within-set slope vs RPE - rows: feature, cols: exercise",
        y=1.02, fontsize=12,
    )
    g.figure.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(g.figure)


def plot_subject_heatmap(per_rep: pd.DataFrame, feature: str,
                          out_path: Path) -> None:
    """Heatmap subject x set, value = mean(feature) z-scored within subject."""
    agg = (
        per_rep.groupby(["subject_id", "set_number"])[feature]
        .mean()
        .reset_index()
    )
    # z-score within subject
    agg["z"] = agg.groupby("subject_id")[feature].transform(
        lambda v: (v - v.mean()) / (v.std() + 1e-9)
    )
    pivot = agg.pivot(index="subject_id", columns="set_number", values="z")
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * pivot.shape[1] + 4),
                                    0.45 * pivot.shape[0] + 2))
    sns.heatmap(
        pivot, cmap="vlag", center=0, ax=ax,
        cbar_kws=dict(label=f"z({feature}) within subject"),
        linewidths=0.3, linecolor="white",
    )
    ax.set_xlabel("set number")
    ax.set_ylabel("subject")
    ax.set_title(f"{feature}: per-subject z across sets 1-12")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_rep_curves(per_rep: pd.DataFrame, out_dir: Path) -> list[Path]:
    """For each exercise, lineplot of mean_acc vs rep_index for the
    earliest, middle, and last set of that exercise per recording."""
    written = []
    for ex in EXERCISES:
        sub = per_rep[per_rep["exercise"] == ex].copy()
        if sub.empty:
            continue
        # for each recording pick first/mid/last set of this exercise
        picks = []
        for rec, g in sub.groupby("recording_id"):
            sets = sorted(g["set_number"].unique())
            if not sets:
                continue
            chosen = {sets[0]: "first", sets[-1]: "last"}
            if len(sets) >= 3:
                chosen[sets[len(sets) // 2]] = "mid"
            for s, kind in chosen.items():
                picks.append(g[g["set_number"] == s].assign(set_phase=kind))
        if not picks:
            continue
        plot_df = pd.concat(picks, ignore_index=True)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.lineplot(
            data=plot_df,
            x="rep_index", y="mean_acc",
            hue="set_phase", style="set_phase",
            hue_order=["first", "mid", "last"],
            errorbar="sd", ax=ax,
        )
        ax.set_title(f"{ex}: mean_acc per rep across set phase")
        ax.set_xlabel("rep index")
        ax.set_ylabel("mean acc magnitude")
        ax.legend(title=None, frameon=False)
        fig.tight_layout()
        out = out_dir / f"rep_curves_{ex}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labeled-root", type=Path, default=Path("data/labeled"))
    p.add_argument("--aligned-root", type=Path, default=Path("dataset_aligned"))
    p.add_argument("--out", type=Path, default=Path("results/acc_fatigue"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid")

    parquets = sorted(args.labeled_root.glob("recording_*/aligned_features.parquet"))
    if not parquets:
        raise SystemExit(f"no labeled parquets under {args.labeled_root}")

    per_rep_frames = []
    for pq in parquets:
        print(f"[load] {pq.parent.name}")
        df = extract_recording(pq, args.aligned_root)
        if not df.empty:
            per_rep_frames.append(df)

    per_rep = pd.concat(per_rep_frames, ignore_index=True)
    per_rep_path = args.out / "per_rep.parquet"
    per_rep.to_parquet(per_rep_path, index=False)
    print(f"[save] {per_rep_path}  ({len(per_rep)} reps)")

    slopes = compute_set_slopes(per_rep)
    slopes_path = args.out / "per_set_slopes.parquet"
    slopes.to_parquet(slopes_path, index=False)
    print(f"[save] {slopes_path}  ({len(slopes)} sets)")

    spearman = spearman_table(slopes)
    spearman_path = args.out / "spearman_per_exercise.csv"
    spearman.to_csv(spearman_path, index=False)
    print(f"[save] {spearman_path}")
    print(spearman.to_string(index=False))

    plot_slope_vs_rpe(slopes, args.out / "slope_vs_rpe.png")
    print(f"[plot] {args.out / 'slope_vs_rpe.png'}")

    for feat in FEATURES:
        out = args.out / f"subject_heatmap_{feat}.png"
        plot_subject_heatmap(per_rep, feat, out)
        print(f"[plot] {out}")

    for p_ in plot_rep_curves(per_rep, args.out):
        print(f"[plot] {p_}")


if __name__ == "__main__":
    main()
