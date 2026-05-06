"""Plot predicted vs EMG-measured fatigue as two time-series on the same axis.

For each recording (or all), runs the v7 fatigue-only TCN forward over active
windows, then derives a "measured fatigue" trace from EMG spectral features
(Dimitrov FInsm5 baseline-normalized, mapped to a 1-10 RPE-like scale). Both
traces are plotted on the same y-axis so you can visually inspect whether
the model's prediction tracks the EMG-derived fatigue signal across the
session.

Pearson r between the two traces (and vs true per-set RPE) is reported in
the figure title.

    python scripts/plot_fatigue_traces.py --recording 014
    python scripts/plot_fatigue_traces.py --all-recordings

Caveat: "measured fatigue" here is *one* mapping of EMG spectral compression
into 1-10 space. It is NOT ground truth — it's a literature-supported proxy
(Dimitrov et al. 2006). Within-set drops in MNF + rises in Dimitrov are
genuine fatigue indicators; across-set comparisons confound exercise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import LabelEncoder
from src.data.loaders import load_biosignal, load_imu
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.eval.plot_style import apply_style, despine
from src.features.acc_features import extract_acc_features, FS_ACC
from src.features.emg_features import (
    EmgBaselineNormalizer, extract_emg_features, FS_EMG,
)
from scripts.plot_fatigue_predictions import (
    DEFAULT_RUN, LABELED_CLEAN, SPLITS_CSV,
    load_tcn_checkpoint, predict_fatigue, resolve_fold,
)

DATASET_ALIGNED = ROOT / "dataset_aligned"

apply_style()

EXERCISE_COLORS = {
    "pullup":     "#4C78A8",
    "squat":      "#F58518",
    "deadlift":   "#54A24B",
    "benchpress": "#E45756",
}


def compute_emg_hires(rec_id: str, hop_ms: float) -> pd.DataFrame:
    """Recompute baseline-normalized MNF/MDF/RMS at fine hop_ms.

    Uses the same offline pipeline as src/features/window_features.py
    (filtfilt 20-450 Hz BP + 50 Hz notch -> sliding 500 ms window -> Welch PSD),
    just with a smaller hop. Returns t_unix-indexed DataFrame.
    """
    rec_dir = DATASET_ALIGNED / f"recording_{rec_id}"
    emg_df = load_biosignal(rec_dir, "emg", "emg")
    t0 = float(emg_df["timestamp"].min())
    baseline_end_unix = t0 + 60.0
    norm = EmgBaselineNormalizer()
    feats = extract_emg_features(
        emg_df["emg"].to_numpy(),
        emg_df["timestamp"].to_numpy(),
        fs=FS_EMG,
        window_ms=500,
        hop_ms=hop_ms,
        normalizer=norm,
        baseline_end_unix=baseline_end_unix,
    )
    return feats


def compute_acc_hires(rec_id: str, hop_ms: float) -> pd.DataFrame:
    """Recompute acc_dom_freq (and other acc features) at fine hop_ms.

    Same pipeline as src/features/window_features.py — sliding 2 s window
    over filtered acc-magnitude with Welch PSD, just at smaller hop.
    """
    rec_dir = DATASET_ALIGNED / f"recording_{rec_id}"
    imu = load_imu(rec_dir)
    return extract_acc_features(
        imu["ax"].to_numpy(),
        imu["ay"].to_numpy(),
        imu["az"].to_numpy(),
        imu["timestamp"].to_numpy(),
        fs=FS_ACC,
        window_ms=2000,
        hop_ms=hop_ms,
    )


def measured_fatigue_from_acc_freq(freq: np.ndarray) -> np.ndarray:
    """Map acc_dom_freq (rep cadence Hz) to a 1-10 RPE-like scale.

    Higher rep frequency = less fatigued (Sanchez-Medina & Gonzalez-Badillo
    2011 — velocity loss as fatigue marker). Direction inverted vs MDF:
      95th-pct freq (fastest reps) -> fatigue = 1
      5th-pct freq  (slowest reps) -> fatigue = 10
    """
    finite = freq[np.isfinite(freq)]
    if finite.size == 0:
        return np.full_like(freq, 1.0)
    hi = float(np.percentile(finite, 95))  # rested = high freq
    lo = float(np.percentile(finite, 5))   # fatigued = low freq
    if hi <= lo:
        hi = lo + 1e-3
    out = 1.0 + 9.0 * np.clip((hi - freq) / (hi - lo), 0.0, 1.0)
    return out


def measured_fatigue_from_relfreq(rel: np.ndarray) -> np.ndarray:
    """Map a baseline-normalized EMG frequency (MNF or MDF, relative to the
    first 90 s rest baseline) to a 1-10 RPE-like scale.

    MNF and MDF both DECREASE with fatigue (spectral compression toward
    lower frequencies; Lindstrom et al. 1970, De Luca 1984). We invert
    direction so that:
      rel == 1.0 (baseline)            -> fatigue = 1
      rel <= 5th-percentile of session -> fatigue = 10

    Per-recording rescaling — only the trajectory within a recording is
    comparable, not absolute values across subjects.
    """
    finite = rel[np.isfinite(rel)]
    if finite.size == 0:
        return np.full_like(rel, 1.0)
    hi = 1.0  # baseline -> fatigue = 1
    lo = float(np.percentile(finite, 5))  # most-fatigued -> fatigue = 10
    if hi <= lo:
        hi = lo + 1e-3
    out = 1.0 + 9.0 * np.clip((hi - rel) / (hi - lo), 0.0, 1.0)
    return out


def predict_traces(rec_id: str, seed: int, run_dir: Path,
                    device: torch.device,
                    hop_s: float | None = None
                    ) -> tuple[pd.DataFrame, str]:
    """Return per-window dataframe with t_session_s, pred_fatigue,
    measured_fatigue, set_number, exercise, rpe_true."""
    fold, subj = resolve_fold(rec_id)
    ckpt = run_dir / f"seed_{seed}" / f"fold_{fold}" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    hps = json.loads((run_dir.parent / "hps.json").read_text())
    dataset_meta = json.loads((run_dir.parent / "dataset_meta.json").read_text())
    train_cfg = json.loads((run_dir.parent / "train_config.json").read_text())

    parquet = LABELED_CLEAN / f"recording_{rec_id}" / "aligned_features.parquet"
    ex_enc = LabelEncoder(classes=dataset_meta["exercise_classes"])
    ph_enc = LabelEncoder(classes=dataset_meta["phase_classes"])
    n_t = int(dataset_meta["n_timesteps"])
    window_s = n_t / 100.0

    dataset = RawMultimodalWindowDataset(
        parquet_paths=[parquet],
        active_only=True,
        exercise_encoder=ex_enc,
        phase_encoder=ph_enc,
        target_modes=train_cfg["target_modes"],
        window_s=window_s,
        hop_s=hop_s,
        verbose=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"recording {rec_id}: 0 active windows")

    model = load_tcn_checkpoint(ckpt, hps, dataset_meta)
    preds = predict_fatigue(model, dataset, device)

    df_aligned = dataset._dfs[0]

    # When the prediction hop is finer than 100 ms, recompute MNF/MDF at the
    # same hop instead of looking them up from window_features.parquet (which
    # was computed at 100 ms hop). Otherwise use the cached parquet.
    use_hires = hop_s is not None and hop_s < 0.1
    if use_hires:
        hop_ms_hires = max(1.0, hop_s * 1000.0)
        print(f"[traces]   recomputing EMG/ACC features at hop={hop_ms_hires:g} ms ...",
              flush=True)
        hires_emg = compute_emg_hires(rec_id, hop_ms=hop_ms_hires)
        hires_acc = compute_acc_hires(rec_id, hop_ms=hop_ms_hires)
        feat_t_unix = hires_emg["t_unix"].to_numpy()
        mnf_arr = hires_emg["emg_mnf_rel"].to_numpy()
        mdf_arr = hires_emg["emg_mdf_rel"].to_numpy()
        # acc has its own t_unix grid (same hop, but different start). Build a
        # second lookup array.
        acc_t_unix = hires_acc["t_unix"].to_numpy()
        acc_freq_arr = hires_acc["acc_dom_freq"].to_numpy()
    else:
        df_feat = pd.read_parquet(
            LABELED_CLEAN / f"recording_{rec_id}" / "window_features.parquet",
            columns=["t_session_s", "emg_mnf_rel", "emg_mdf_rel", "acc_dom_freq"],
        )
        feat_t_session = df_feat["t_session_s"].to_numpy()
        mnf_arr = df_feat["emg_mnf_rel"].to_numpy()
        mdf_arr = df_feat["emg_mdf_rel"].to_numpy()
        acc_freq_arr = df_feat["acc_dom_freq"].to_numpy()

    t_session = df_aligned["t_session_s"].to_numpy()
    t_unix_arr = df_aligned["t_unix"].to_numpy()
    set_arr = df_aligned["set_number"].to_numpy()
    ex_arr = df_aligned["exercise"].astype(str).to_numpy()
    rpe_arr = df_aligned["rpe_for_this_set"].to_numpy()

    rows = []
    win_size = dataset.window_size
    for k, (_, start) in enumerate(dataset._window_idx):
        end = start + win_size - 1
        t_end = float(t_session[end])
        sn = set_arr[end]
        if pd.isna(sn):
            continue
        if use_hires:
            t_end_unix = float(t_unix_arr[end])
            ji = int(np.searchsorted(feat_t_unix, t_end_unix, side="right") - 1)
            ji = max(0, min(ji, len(feat_t_unix) - 1))
            ja = int(np.searchsorted(acc_t_unix, t_end_unix, side="right") - 1)
            ja = max(0, min(ja, len(acc_t_unix) - 1))
            acc_freq_val = float(acc_freq_arr[ja])
        else:
            ji = int(np.searchsorted(feat_t_session, t_end, side="right") - 1)
            ji = max(0, min(ji, len(feat_t_session) - 1))
            acc_freq_val = float(acc_freq_arr[ji])
        rows.append({
            "t_end_s": t_end,
            "set_number": int(sn),
            "exercise": ex_arr[end],
            "rpe_true": (float(rpe_arr[end])
                          if pd.notna(rpe_arr[end]) else np.nan),
            "pred_fatigue": float(preds[k]),
            "emg_mnf_rel": float(mnf_arr[ji]),
            "emg_mdf_rel": float(mdf_arr[ji]),
            "acc_dom_freq": acc_freq_val,
        })
    table = pd.DataFrame(rows)
    table["mnf_fatigue"] = measured_fatigue_from_relfreq(
        table["emg_mnf_rel"].to_numpy())
    table["mdf_fatigue"] = measured_fatigue_from_relfreq(
        table["emg_mdf_rel"].to_numpy())
    table["acc_fatigue"] = measured_fatigue_from_acc_freq(
        table["acc_dom_freq"].to_numpy())
    return table, subj


def per_set_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("set_number")
        .agg(t_start=("t_end_s", "min"),
             t_end=("t_end_s", "max"),
             exercise=("exercise", "first"),
             rpe_true=("rpe_true", "first"),
             pred_mean=("pred_fatigue", "mean"),
             mnf_fat_mean=("mnf_fatigue", "mean"),
             mdf_fat_mean=("mdf_fatigue", "mean"))
        .reset_index()
        .sort_values("set_number")
    )


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return np.nan
    return float(pearsonr(a[mask], b[mask])[0])


def plot_recording(table: pd.DataFrame, rec_id: str, subj: str,
                    out_path: Path) -> Path:
    sets = per_set_summary(table)
    rpe_rows = table[table["rpe_true"].notna()]

    fig, ax = plt.subplots(figsize=(15, 6.0))

    # Set-region shading only.
    for _, row in sets.iterrows():
        color = EXERCISE_COLORS.get(row["exercise"], "#888")
        ax.axvspan(row["t_start"], row["t_end"], color=color, alpha=0.15,
                    zorder=0)

    # Three fatigue traces on same y-axis (1-10): MDF + ACC rep cadence + predicted.
    ax.plot(table["t_end_s"], table["mdf_fatigue"],
             color="#9467bd", lw=1.2, alpha=0.7,
             label="MDF")
    ax.plot(table["t_end_s"], table["acc_fatigue"],
             color="#2ca02c", lw=1.2, alpha=0.7,
             label="ACC rep freq")
    ax.plot(table["t_end_s"], table["pred_fatigue"],
             color="#C0392B", lw=1.6, alpha=0.9,
             label="predicted")

    # True per-set RPE step plot
    if not rpe_rows.empty:
        for _, row in sets.iterrows():
            if not np.isfinite(row["rpe_true"]):
                continue
            ax.hlines(row["rpe_true"], row["t_start"], row["t_end"],
                       colors="#000", lw=2.0, label=None)
        ax.plot([], [], color="#000", lw=2.0, label="true RPE (per set)")

    ax.set_xlabel("Session time (s)")
    ax.set_ylabel("fatigue (1-10)")
    ax.set_ylim(0.5, 10.5)
    ax.grid(alpha=0.25)

    ax.set_title(
        f"recording {rec_id} ({subj}) — predicted vs EMG-measured fatigue",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    despine(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_single_set(table: pd.DataFrame, rec_id: str, subj: str,
                     set_number: int, out_path: Path) -> Path:
    """Zoomed view of one set: pred + measured + true RPE on RPE 1-10 axis."""
    sub = table[table["set_number"] == set_number].sort_values("t_end_s")
    if sub.empty:
        raise SystemExit(f"set {set_number} not in recording {rec_id}")

    exercise = str(sub["exercise"].iloc[0])
    rpe_true = float(sub["rpe_true"].iloc[0]) if pd.notna(sub["rpe_true"].iloc[0]) else np.nan
    color = EXERCISE_COLORS.get(exercise, "#888")

    t0 = float(sub["t_end_s"].iloc[0])
    t = sub["t_end_s"].to_numpy() - t0  # seconds into set

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(t[0], t[-1], color=color, alpha=0.12, zorder=0)

    # Drop markers when there are too many points (>= 60 windows ≈ dense).
    use_markers = len(sub) < 60
    mk = dict(marker="s", ms=4) if use_markers else dict()
    ax.plot(t, sub["mdf_fatigue"], color="#9467bd", lw=1.4, alpha=0.8,
             label="MDF", **mk)
    mk = dict(marker="^", ms=4) if use_markers else dict()
    ax.plot(t, sub["acc_fatigue"], color="#2ca02c", lw=1.4, alpha=0.8,
             label="ACC rep freq", **mk)
    mk = dict(marker="D", ms=5) if use_markers else dict()
    ax.plot(t, sub["pred_fatigue"], color="#C0392B", lw=1.6,
             label="predicted", **mk)
    if np.isfinite(rpe_true):
        ax.axhline(rpe_true, color="#000", lw=1.5, ls="--", alpha=0.85,
                    label="true RPE")

    ax.set_xlabel("Time in set (s)")
    ax.set_ylabel("fatigue (1-10)")
    ax.set_ylim(0.5, 10.5)
    ax.grid(alpha=0.3)
    ax.set_title(
        f"recording {rec_id} ({subj}) — set {set_number} ({exercise})",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower left", fontsize=10, framealpha=0.92)
    fig.tight_layout()
    despine(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid_all(tables: dict[str, pd.DataFrame],
                   out_path: Path) -> Path:
    """One row per recording — vertically stacked time-series."""
    n = len(tables)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.6 * n + 1.2),
                              squeeze=False, sharey=True)
    axes = axes[:, 0]

    rs = []
    for ax, (rec_id, table) in zip(axes, tables.items()):
        sets = per_set_summary(table)
        for _, row in sets.iterrows():
            color = EXERCISE_COLORS.get(row["exercise"], "#888")
            ax.axvspan(row["t_start"], row["t_end"], color=color, alpha=0.15,
                        zorder=0)
            if np.isfinite(row["rpe_true"]):
                ax.hlines(row["rpe_true"], row["t_start"], row["t_end"],
                           colors="#000", lw=2.0)
        ax.plot(table["t_end_s"], table["mdf_fatigue"],
                 color="#9467bd", lw=1.0, alpha=0.7, label="MDF")
        ax.plot(table["t_end_s"], table["acc_fatigue"],
                 color="#2ca02c", lw=1.0, alpha=0.7, label="ACC")
        ax.plot(table["t_end_s"], table["pred_fatigue"],
                 color="#C0392B", lw=1.6, alpha=0.9, label="pred")

        r_pd = _safe_pearson(table["pred_fatigue"].to_numpy(),
                              table["mdf_fatigue"].to_numpy())
        rpe_rows = table[table["rpe_true"].notna()]
        r_pr = _safe_pearson(rpe_rows["pred_fatigue"].to_numpy(),
                              rpe_rows["rpe_true"].to_numpy())
        rs.append((rec_id, r_pd, r_pr))
        ax.set_ylim(0.5, 10.5)
        ax.set_ylabel(f"rec {rec_id}\nfat (1-10)", fontsize=9)
        ax.set_title(
            f"recording {rec_id}: r(pred,MDF)={r_pd:+.3f}  "
            f"r(pred,RPE)={r_pr:+.3f}",
            fontsize=10, loc="left")
        ax.grid(alpha=0.25)

    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.92)
    axes[-1].set_xlabel("Session time (s)")

    fig.suptitle(
        "predicted vs EMG MDF fatigue across LOSO subjects",
        fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.96, hspace=0.55)
    despine(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recording", help="single recording, e.g. 014")
    g.add_argument("--all-recordings", action="store_true")
    ap.add_argument("--set", type=int, default=None, dest="set_number",
                     help="zoom to a single set (1-12); requires --recording")
    ap.add_argument("--hz", type=float, default=1.0,
                     help="prediction sample rate (Hz); 100 = run model every "
                          "10 ms, default 1 = run every 1 s like training")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 1337, 7])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    hop_s = 1.0 / args.hz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[traces] device={device}, seed={args.seed}")

    if args.recording:
        rec_id = args.recording.zfill(3)
        table, subj = predict_traces(rec_id, args.seed, args.run_dir, device,
                                       hop_s=hop_s)
        out_dir = ROOT / "inspections" / f"recording_{rec_id}"
        if args.set_number is not None:
            out_path = (out_dir
                / f"fatigue_traces_seed{args.seed}_set{args.set_number:02d}.png")
            out_path = plot_single_set(table, rec_id, subj,
                                          args.set_number, out_path)
        else:
            out_path = out_dir / f"fatigue_traces_seed{args.seed}.png"
            out_path = plot_recording(table, rec_id, subj, out_path)
        csv = out_dir / f"fatigue_traces_seed{args.seed}.csv"
        table.to_csv(csv, index=False)
        print(f"[traces] saved: {out_path}")
        print(f"[traces] csv:   {csv}")
    else:
        splits = pd.read_csv(SPLITS_CSV)
        rec_for_subj = {}
        for p in sorted(LABELED_CLEAN.glob("recording_*/aligned_features.parquet")):
            df = pd.read_parquet(p, columns=["subject_id"])
            rec_for_subj[str(df["subject_id"].iloc[0])] = (
                p.parent.name.split("_")[1])
        tables: dict[str, pd.DataFrame] = {}
        for s in splits["subject_id"]:
            if s not in rec_for_subj:
                continue
            rid = rec_for_subj[s]
            print(f"[traces]   {rid} ({s}) ...", flush=True)
            tables[rid], _ = predict_traces(rid, args.seed, args.run_dir,
                                              device, hop_s=hop_s)
        out_dir = ROOT / "inspections" / "_pooled"
        out_path = out_dir / f"fatigue_traces_all_seed{args.seed}.png"
        out_path = plot_grid_all(tables, out_path)
        print(f"[traces] saved: {out_path}")


if __name__ == "__main__":
    main()
