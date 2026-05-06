"""Correlate predicted fatigue with every feature in window_features.parquet.

For each recording (or all recordings pooled), runs the v7 fatigue-only TCN
forward pass over active windows, joins per-window predictions with the full
feature table, and computes Pearson + Spearman correlations vs pred_fatigue
(and vs true RPE). Saves a sorted bar chart + CSV.

Single recording:
    python scripts/correlate_fatigue_predictions.py --recording 014

All 7 LOSO recordings (each predicted by its held-out fold's checkpoint):
    python scripts/correlate_fatigue_predictions.py --all-recordings

Caveat: pooled correlation is contaminated by exercise effects — features
that vary across exercises (e.g. emg_dimitrov is much higher in deadlift
than pullup) will look spuriously correlated. The Fisher-z per-subject
average (added when --all-recordings is used) controls for cross-subject
distribution differences but NOT cross-exercise. Treat |r| < 0.3 as noise.
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
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import LabelEncoder
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.eval.plot_style import apply_style, despine
from scripts.plot_fatigue_predictions import (
    DEFAULT_RUN, LABELED_CLEAN, SPLITS_CSV,
    load_tcn_checkpoint, predict_fatigue, resolve_fold,
)

apply_style()

# Columns that are labels/metadata, not features.
NON_FEATURE_COLS = {
    "subject_id", "recording_id", "t_unix", "t_session_s", "in_active_set",
    "set_number", "exercise", "phase_label", "rep_count_in_set",
    "rep_density_hz", "rpe_for_this_set", "t_window_center_s",
    "has_rep_intervals",
    # Phase fractions are derived from labels, exclude.
    "phase_frac_rest", "phase_frac_concentric", "phase_frac_eccentric",
    "phase_frac_isometric", "phase_frac_unknown",
    # Soft-target rep columns are also derived from labels.
    "soft_overlap_reps", "soft_overlap_reps_1s", "soft_overlap_reps_2_5s",
    "soft_overlap_reps_4s", "soft_overlap_reps_3s", "soft_overlap_reps_5s",
    "reps_in_window_2s",
}


def predict_recording_table(
    rec_id: str, seed: int, run_dir: Path, device: torch.device
) -> tuple[pd.DataFrame, str]:
    """Return (per-window table joined with all features, subject_name)."""
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
        verbose=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"recording {rec_id}: 0 active windows")

    model = load_tcn_checkpoint(ckpt, hps, dataset_meta)
    preds = predict_fatigue(model, dataset, device)

    df_aligned = dataset._dfs[0]
    df_feat = pd.read_parquet(
        LABELED_CLEAN / f"recording_{rec_id}" / "window_features.parquet"
    )

    feat_t = df_feat["t_session_s"].to_numpy()
    t_session = df_aligned["t_session_s"].to_numpy()
    set_arr = df_aligned["set_number"].to_numpy()
    ex_arr = df_aligned["exercise"].astype(str).to_numpy()
    rpe_arr = df_aligned["rpe_for_this_set"].to_numpy()

    feature_cols = [c for c in df_feat.columns
                     if c not in NON_FEATURE_COLS
                     and pd.api.types.is_numeric_dtype(df_feat[c])]
    feat_vals = {c: df_feat[c].to_numpy() for c in feature_cols}

    rows = []
    win_size = dataset.window_size
    for k, (_, start) in enumerate(dataset._window_idx):
        end = start + win_size - 1
        t_end = float(t_session[end])
        sn = set_arr[end]
        if pd.isna(sn):
            continue
        ji = int(np.searchsorted(feat_t, t_end, side="right") - 1)
        ji = max(0, min(ji, len(feat_t) - 1))
        row = {
            "recording_id": f"recording_{rec_id}",
            "subject_id": subj,
            "t_end_s": t_end,
            "set_number": int(sn),
            "exercise": ex_arr[end],
            "rpe_true": (float(rpe_arr[end])
                          if pd.notna(rpe_arr[end]) else np.nan),
            "pred_fatigue": float(preds[k]),
        }
        for c in feature_cols:
            row[c] = float(feat_vals[c][ji])
        rows.append(row)
    return pd.DataFrame(rows), subj


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    xs, ys = x[mask], y[mask]
    if np.nanstd(xs) == 0 or np.nanstd(ys) == 0:
        return np.nan
    func = pearsonr if kind == "pearson" else spearmanr
    try:
        r, _ = func(xs, ys)
        return float(r)
    except Exception:
        return np.nan


def correlations_table(table: pd.DataFrame, target_col: str,
                        feature_cols: list[str]) -> pd.DataFrame:
    target = table[target_col].to_numpy(dtype=np.float64)
    rows = []
    for c in feature_cols:
        v = table[c].to_numpy(dtype=np.float64)
        rows.append({
            "feature": c,
            "pearson_r": _safe_corr(v, target, "pearson"),
            "spearman_r": _safe_corr(v, target, "spearman"),
        })
    out = pd.DataFrame(rows)
    out["abs_r"] = out["pearson_r"].abs()
    return out.sort_values("abs_r", ascending=False).reset_index(drop=True)


def per_subject_fisher_z(table: pd.DataFrame, target_col: str,
                          feature_cols: list[str]) -> pd.DataFrame:
    """Fisher-z mean Pearson r across subjects (controls for cross-subject
    distribution differences). Per-subject r is z-transformed, averaged,
    then back-transformed."""
    rows = []
    subjects = table["subject_id"].unique().tolist()
    for c in feature_cols:
        zs = []
        for s in subjects:
            sub = table[table["subject_id"] == s]
            r = _safe_corr(
                sub[c].to_numpy(dtype=np.float64),
                sub[target_col].to_numpy(dtype=np.float64),
                "pearson",
            )
            if not np.isfinite(r):
                continue
            r = max(min(r, 0.9999), -0.9999)
            zs.append(np.arctanh(r))
        if not zs:
            rows.append({"feature": c, "fisher_z_r": np.nan, "n_subjects": 0})
            continue
        z_mean = float(np.mean(zs))
        rows.append({
            "feature": c,
            "fisher_z_r": float(np.tanh(z_mean)),
            "n_subjects": len(zs),
        })
    out = pd.DataFrame(rows)
    out["abs_z_r"] = out["fisher_z_r"].abs()
    return out.sort_values("abs_z_r", ascending=False).reset_index(drop=True)


def plot_corr_bar(corr: pd.DataFrame, title: str, out_path: Path,
                   extra_col: str | None = None,
                   extra_label: str | None = None) -> Path:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.28 * n)))
    y = np.arange(n)
    bars1 = ax.barh(y - 0.2, corr["pearson_r"], height=0.36,
                     color=["#C0392B" if v < 0 else "#1f77b4"
                             for v in corr["pearson_r"]],
                     label="Pearson r")
    bars2 = ax.barh(y + 0.2, corr["spearman_r"], height=0.36,
                     color="#2ca02c", alpha=0.55, label="Spearman r")
    if extra_col is not None and extra_col in corr.columns:
        ax.scatter(corr[extra_col], y, marker="D", color="black",
                    s=28, zorder=5, label=extra_label or extra_col)
    ax.set_yticks(y)
    ax.set_yticklabels(corr["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("correlation coefficient")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    despine(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--recording", help="Single recording id, e.g. 014")
    g.add_argument("--all-recordings", action="store_true",
                    help="Pool over all 7 LOSO subjects (uses fold checkpoints)")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 1337, 7])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[corr] device={device}, seed={args.seed}")

    if args.recording:
        rec_id = args.recording.zfill(3)
        print(f"[corr] recording_{rec_id} ...")
        table, subj = predict_recording_table(rec_id, args.seed, args.run_dir,
                                                device)
        out_dir = ROOT / "inspections" / f"recording_{rec_id}"
        tag = f"recording_{rec_id}_seed{args.seed}"
        title_subj = f"recording {rec_id} ({subj})"
    else:
        splits = pd.read_csv(SPLITS_CSV)
        # Map subject -> recording from labeled_clean (skip excluded ones).
        rec_for_subj = {}
        for p in sorted(LABELED_CLEAN.glob("recording_*/aligned_features.parquet")):
            df = pd.read_parquet(p, columns=["subject_id"])
            rec_for_subj[str(df["subject_id"].iloc[0])] = p.parent.name.split("_")[1]
        rec_ids = []
        for s in splits["subject_id"]:
            if s in rec_for_subj:
                rec_ids.append(rec_for_subj[s])
        print(f"[corr] {len(rec_ids)} recordings: {rec_ids}")
        all_tables = []
        for rec_id in rec_ids:
            print(f"[corr]   {rec_id} ...", flush=True)
            t, _ = predict_recording_table(rec_id, args.seed, args.run_dir,
                                             device)
            all_tables.append(t)
        table = pd.concat(all_tables, ignore_index=True)
        out_dir = ROOT / "inspections" / "_pooled"
        tag = f"pooled_seed{args.seed}"
        title_subj = f"pooled across {len(rec_ids)} subjects (n={len(table)})"

    feature_cols = [c for c in table.columns
                     if c not in {"recording_id", "subject_id", "t_end_s",
                                   "set_number", "exercise", "rpe_true",
                                   "pred_fatigue"}
                     and pd.api.types.is_numeric_dtype(table[c])]
    print(f"[corr] {len(feature_cols)} features, {len(table)} windows")

    # vs pred_fatigue
    corr_pred = correlations_table(table, "pred_fatigue", feature_cols)
    # vs true RPE (only on rows where rpe is defined)
    rpe_rows = table[table["rpe_true"].notna()]
    corr_rpe = correlations_table(rpe_rows, "rpe_true", feature_cols)

    if args.all_recordings:
        fz_pred = per_subject_fisher_z(table, "pred_fatigue", feature_cols)
        fz_rpe = per_subject_fisher_z(rpe_rows, "rpe_true", feature_cols)
        corr_pred = corr_pred.merge(
            fz_pred[["feature", "fisher_z_r"]].rename(
                columns={"fisher_z_r": "fisher_z_pearson"}),
            on="feature", how="left",
        )
        corr_rpe = corr_rpe.merge(
            fz_rpe[["feature", "fisher_z_r"]].rename(
                columns={"fisher_z_r": "fisher_z_pearson"}),
            on="feature", how="left",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_pred = out_dir / f"corr_vs_pred_fatigue_{tag}.csv"
    csv_rpe = out_dir / f"corr_vs_rpe_true_{tag}.csv"
    full_csv = out_dir / f"window_table_{tag}.csv"
    corr_pred.to_csv(csv_pred, index=False)
    corr_rpe.to_csv(csv_rpe, index=False)
    table.to_csv(full_csv, index=False)

    extra_col = "fisher_z_pearson" if args.all_recordings else None
    extra_label = "Fisher-z mean Pearson (per subject)" if extra_col else None

    png_pred = plot_corr_bar(
        corr_pred, f"correlation vs pred_fatigue — {title_subj}",
        out_dir / f"corr_vs_pred_fatigue_{tag}.png",
        extra_col, extra_label,
    )
    png_rpe = plot_corr_bar(
        corr_rpe, f"correlation vs true RPE — {title_subj}",
        out_dir / f"corr_vs_rpe_true_{tag}.png",
        extra_col, extra_label,
    )

    print()
    print("Top 10 features by |Pearson r| vs pred_fatigue:")
    print(corr_pred.head(10).to_string(index=False,
                                          float_format=lambda v: f"{v:+.3f}"))
    print()
    print("Top 10 features by |Pearson r| vs true RPE:")
    print(corr_rpe.head(10).to_string(index=False,
                                         float_format=lambda v: f"{v:+.3f}"))
    print()
    print(f"[corr] saved: {png_pred}")
    print(f"[corr] saved: {png_rpe}")
    print(f"[corr] csv:   {csv_pred}")
    print(f"[corr] csv:   {csv_rpe}")
    print(f"[corr] csv:   {full_csv}")


if __name__ == "__main__":
    main()
