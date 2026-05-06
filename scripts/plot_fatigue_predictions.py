"""Plot per-window fatigue predictions alongside EMG MNF/MDF, all sets.

Uses the best fatigue-only TCN model (v7) as scored by the cv_summary across
the optuna runs (MAE=0.746, Pearson r=0.46 on phase1; MAE=0.80, r=0.42 on
multi-seed phase2). For each recording, loads the checkpoint from the LOSO
fold where the recording was the held-out test subject, runs forward over
all active windows, and renders a 4x3 grid (one subplot per set) showing:

  - left y-axis : EMG MNF and MDF (Hz)
  - right y-axis: predicted fatigue (RPE 1-10), per-window
  - title       : set #, exercise, true RPE

Run:
    python scripts/plot_fatigue_predictions.py --recording 014
    python scripts/plot_fatigue_predictions.py --recording 014 --seed 42
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.datasets import LabelEncoder
from src.data.raw_window_dataset import RawMultimodalWindowDataset
from src.eval.plot_style import apply_style, despine
from src.models.raw.tcn_raw import TCNRawMultiTask

apply_style()

DEFAULT_RUN = ROOT / "runs" / "optuna_clean_v7-fatigue-raw-tcn" / "phase2" / "tcn_raw"
LABELED_CLEAN = ROOT / "data" / "labeled_clean"
SPLITS_CSV = ROOT / "configs" / "splits_clean_loso.csv"

EXERCISE_COLORS = {
    "pullup":     "#4C78A8",
    "squat":      "#F58518",
    "deadlift":   "#54A24B",
    "benchpress": "#E45756",
}


def resolve_fold(recording_id: str) -> tuple[int, str]:
    """Return (fold, subject_name) for a recording_id like '014'."""
    aligned = pd.read_parquet(
        LABELED_CLEAN / f"recording_{recording_id}" / "aligned_features.parquet",
        columns=["subject_id"],
    )
    subj = str(aligned["subject_id"].iloc[0])
    splits = pd.read_csv(SPLITS_CSV)
    row = splits[splits["subject_id"] == subj]
    if row.empty:
        raise SystemExit(
            f"subject {subj!r} (rec {recording_id}) not in {SPLITS_CSV.name}"
        )
    return int(row["fold"].iloc[0]), subj


def load_tcn_checkpoint(ckpt_path: Path, hps: dict, dataset_meta: dict
                         ) -> TCNRawMultiTask:
    model = TCNRawMultiTask(
        n_channels=int(dataset_meta["n_channels"]),
        n_timesteps=int(dataset_meta["n_timesteps"]),
        n_exercise=len(dataset_meta["exercise_classes"]),
        n_phase=len(dataset_meta["phase_classes"]),
        kernel_size=int(hps["tcn_kernel"]),
        dropout=float(hps["dropout"]),
        repr_dim=int(hps["repr_dim"]),
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def predict_fatigue(model: TCNRawMultiTask, dataset: RawMultimodalWindowDataset,
                     device: torch.device, batch: int = 256
                     ) -> np.ndarray:
    """Return per-window fatigue predictions (N,)."""
    model.to(device)
    n = len(dataset)
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, batch):
            i1 = min(i0 + batch, n)
            xs = torch.stack([dataset[i]["x"] for i in range(i0, i1)]).to(device)
            preds = model(xs)
            out[i0:i1] = preds["fatigue"].cpu().numpy()
    return out


def build_window_table(dataset: RawMultimodalWindowDataset,
                        df_aligned: pd.DataFrame,
                        df_features: pd.DataFrame,
                        preds: np.ndarray) -> pd.DataFrame:
    """One row per active window: t_session_s_end, set, exercise, rpe, mnf, mdf, pred_fatigue."""
    t_session = df_aligned["t_session_s"].to_numpy()
    set_arr = df_aligned["set_number"].to_numpy()
    ex_arr = df_aligned["exercise"].astype(str).to_numpy()
    rpe_arr = df_aligned["rpe_for_this_set"].to_numpy()

    feat_t = df_features["t_session_s"].to_numpy()
    mnf_arr = df_features["emg_mnf"].to_numpy()
    mdf_arr = df_features["emg_mdf"].to_numpy()
    dim_arr = df_features["emg_dimitrov_rel"].to_numpy()
    accfreq_arr = df_features["acc_dom_freq"].to_numpy()

    rows = []
    win_size = dataset.window_size
    for k, (file_idx, start) in enumerate(dataset._window_idx):
        end = start + win_size - 1
        t_end = float(t_session[end])
        sn = set_arr[end]
        if pd.isna(sn):
            continue
        # MNF/MDF lookup at the same end-of-window sample.
        ji = int(np.searchsorted(feat_t, t_end, side="right") - 1)
        ji = max(0, min(ji, len(feat_t) - 1))
        rows.append({
            "t_end_s": t_end,
            "set_number": int(sn),
            "exercise": ex_arr[end],
            "rpe_true": float(rpe_arr[end]) if pd.notna(rpe_arr[end]) else np.nan,
            "emg_mnf": float(mnf_arr[ji]),
            "emg_mdf": float(mdf_arr[ji]),
            "emg_dimitrov_rel": float(dim_arr[ji]),
            "acc_dom_freq": float(accfreq_arr[ji]),
            "pred_fatigue": float(preds[k]),
        })
    return pd.DataFrame(rows)


def plot_grid(table: pd.DataFrame, recording_id: str, subj: str,
               out_path: Path) -> Path:
    sets = sorted(table["set_number"].unique())
    n = len(sets)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 3.8 * n_rows),
                              squeeze=False)

    rpe_lo, rpe_hi = 0.5, 10.5
    # Common MNF/MDF y-range across sets — easier visual comparison.
    freq_lo = float(np.nanpercentile(
        np.r_[table["emg_mnf"].to_numpy(), table["emg_mdf"].to_numpy()], 2))
    freq_hi = float(np.nanpercentile(
        np.r_[table["emg_mnf"].to_numpy(), table["emg_mdf"].to_numpy()], 98))
    freq_lo, freq_hi = freq_lo - 5, freq_hi + 5
    dim_lo = float(np.nanpercentile(table["emg_dimitrov_rel"], 2))
    dim_hi = float(np.nanpercentile(table["emg_dimitrov_rel"], 98))
    dim_lo = max(0.0, dim_lo - 0.2)
    dim_hi = dim_hi + 0.2
    accf_lo = float(np.nanpercentile(table["acc_dom_freq"], 2))
    accf_hi = float(np.nanpercentile(table["acc_dom_freq"], 98))
    accf_lo = max(0.0, accf_lo - 0.05)
    accf_hi = accf_hi + 0.05

    for i, sn in enumerate(sets):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        sub = table[table["set_number"] == sn].sort_values("t_end_s")
        if sub.empty:
            ax.set_visible(False)
            continue

        ex = sub["exercise"].iloc[0]
        rpe_true = sub["rpe_true"].iloc[0]
        color = EXERCISE_COLORS.get(ex, "#888")

        t = sub["t_end_s"].to_numpy()
        # x relative to set start
        t = t - t[0]

        ax.plot(t, sub["emg_mnf"], color="#1f77b4", lw=1.4, marker="o", ms=3,
                label="MNF (Hz)")
        ax.plot(t, sub["emg_mdf"], color="#9467bd", lw=1.4, marker="s", ms=3,
                label="MDF (Hz)")
        ax.set_ylim(freq_lo, freq_hi)
        ax.set_ylabel("EMG freq (Hz)", fontsize=9)
        ax.set_xlabel("t in set (s)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(t, sub["pred_fatigue"], color="#C0392B", lw=2.0,
                 marker="D", ms=4, label="pred fatigue")
        if np.isfinite(rpe_true):
            ax2.axhline(rpe_true, color="#000", lw=1.0, ls="--", alpha=0.8,
                         label=f"true RPE={rpe_true:.0f}")
        ax2.set_ylim(rpe_lo, rpe_hi)
        ax2.set_ylabel("fatigue (1-10)", color="#C0392B", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#C0392B", labelsize=8)

        # Third axis: Dimitrov FInsm5 (baseline-normalized) — rises with fatigue
        # (Dimitrov et al. 2006). Offset 50 pt outward so it doesn't overlap ax2.
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("outward", 48))
        ax3.plot(t, sub["emg_dimitrov_rel"], color="#2ca02c", lw=1.4,
                 marker="^", ms=3, label="Dimitrov / baseline")
        ax3.set_ylim(dim_lo, dim_hi)
        ax3.set_ylabel("Dimitrov (rel)", color="#2ca02c", fontsize=9)
        ax3.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)

        # Fourth axis: ACC dominant frequency (Hz) = rep cadence — falls as
        # reps slow down (velocity-based-training analog; Sanchez-Medina &
        # Gonzalez-Badillo 2011). Offset further out so labels don't overlap.
        ax4 = ax.twinx()
        ax4.spines["right"].set_position(("outward", 100))
        ax4.plot(t, sub["acc_dom_freq"], color="#ff7f0e", lw=1.4,
                 marker="v", ms=3, label="ACC rep freq")
        ax4.set_ylim(accf_lo, accf_hi)
        ax4.set_ylabel("ACC dom freq (Hz)", color="#ff7f0e", fontsize=9)
        ax4.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=8)

        pred_mean = float(sub["pred_fatigue"].mean())
        err = (pred_mean - rpe_true) if np.isfinite(rpe_true) else np.nan
        ax.set_title(
            f"S{sn:02d} - {ex}  |  true={rpe_true:.0f}  pred={pred_mean:.1f}"
            f"  err={err:+.1f}",
            fontsize=10, color=color, fontweight="bold",
        )

        if i == 0:
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            lines3, labs3 = ax3.get_legend_handles_labels()
            lines4, labs4 = ax4.get_legend_handles_labels()
            ax.legend(lines1 + lines2 + lines3 + lines4,
                       labs1 + labs2 + labs3 + labs4,
                       loc="lower left", fontsize=7, framealpha=0.92)

    for j in range(n, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"recording {recording_id} ({subj}) - per-window fatigue prediction "
        f"vs EMG MNF/MDF (TCN v7, fatigue-only, LOSO held-out fold)",
        fontsize=13, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.85, hspace=0.45, right=0.97)
    despine(fig=fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True,
                     help="Recording id, e.g. 014 (must be in labeled_clean/)")
    ap.add_argument("--seed", type=int, default=42, choices=[42, 1337, 7])
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                     help="Phase2 model directory (default: v7 fatigue TCN)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rec_id = args.recording.zfill(3)
    fold, subj = resolve_fold(rec_id)
    print(f"[plot] recording_{rec_id} -> subject={subj}, LOSO fold={fold}")

    ckpt = args.run_dir / f"seed_{args.seed}" / f"fold_{fold}" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    hps = json.loads((args.run_dir.parent / "hps.json").read_text())
    dataset_meta = json.loads((args.run_dir.parent / "dataset_meta.json").read_text())
    train_cfg = json.loads((args.run_dir.parent / "train_config.json").read_text())
    print(f"[plot] checkpoint: {ckpt.relative_to(ROOT)}")
    print(f"[plot] hps: kernel={hps['tcn_kernel']} repr_dim={hps['repr_dim']} "
          f"dropout={hps['dropout']:.3f}")

    parquet = LABELED_CLEAN / f"recording_{rec_id}" / "aligned_features.parquet"
    if not parquet.exists():
        raise SystemExit(f"aligned_features missing: {parquet}")

    # Pre-fit encoders to match training-time class lists exactly.
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
        verbose=True,
    )
    if len(dataset) == 0:
        raise SystemExit("dataset has 0 active windows; nothing to predict")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[plot] device={device}, windows={len(dataset)}")

    model = load_tcn_checkpoint(ckpt, hps, dataset_meta)
    preds = predict_fatigue(model, dataset, device)

    df_aligned = dataset._dfs[0]
    df_features = pd.read_parquet(
        LABELED_CLEAN / f"recording_{rec_id}" / "window_features.parquet",
        columns=["t_session_s", "emg_mnf", "emg_mdf", "emg_dimitrov_rel",
                  "acc_dom_freq"],
    )
    table = build_window_table(dataset, df_aligned, df_features, preds)

    # CSV companion file so the numbers can be re-plotted without rerunning.
    out_dir = ROOT / "inspections" / f"recording_{rec_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"fatigue_predictions_seed{args.seed}.csv"
    table.to_csv(csv_path, index=False)

    # Per-set summary print
    summary = table.groupby("set_number").agg(
        exercise=("exercise", "first"),
        rpe_true=("rpe_true", "first"),
        pred_mean=("pred_fatigue", "mean"),
        pred_std=("pred_fatigue", "std"),
        mnf_med=("emg_mnf", "median"),
        mdf_med=("emg_mdf", "median"),
        n_windows=("pred_fatigue", "size"),
    )
    print(summary.to_string(float_format=lambda v: f"{v:6.2f}"))

    out_path = args.out or out_dir / f"fatigue_predictions_seed{args.seed}.png"
    out_path = plot_grid(table, rec_id, subj, out_path)
    print(f"[plot] saved: {out_path}")
    print(f"[plot] csv:   {csv_path}")


if __name__ == "__main__":
    main()
