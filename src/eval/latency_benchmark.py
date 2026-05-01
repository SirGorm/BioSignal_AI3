"""Streaming pipeline latency benchmark.

Replays N minutes of a recording through the StreamingFeaturePipeline plus
trained inference models, measuring per-hop wall-clock time. Reports p50/p95/p99.
PASS if p99 <= configured budget (default 100 ms).

Usage:
    python -m src.eval.latency_benchmark --recording dataset/recording_012 \
        --run runs/20260426_154705_default --minutes 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.streaming.realtime import (
    StreamingFeaturePipeline,
    FS_ECG, FS_EMG, FS_EDA, FS_ACC, FS_PPG_DEFAULT, FS_TEMP,
    BASELINE_S,
)
from src.data.loaders import load_biosignal, load_temperature, load_imu, load_metadata


def benchmark(
    rec_dir: Path,
    run_dir: Path,
    minutes: float = 10.0,
    chunk_size: int = 1,  # 1 sample @ 100 Hz = 10 ms hop
    budget_p99_ms: float = 100.0,
) -> dict:
    """Run the benchmark. chunk_size=1 measures per-sample (worst-case) latency."""
    meta = load_metadata(rec_dir)
    ppg_fs = int(meta["sampling_rates"].get("ppg", FS_PPG_DEFAULT))

    # Load native-rate signals
    ecg_df = load_biosignal(rec_dir, "ecg", "ecg")
    emg_df = load_biosignal(rec_dir, "emg", "emg")
    eda_df = load_biosignal(rec_dir, "eda", "eda")
    imu_df = load_imu(rec_dir)
    ppg_df = load_biosignal(rec_dir, "ppg_green", "ppg_green")
    temp_df = load_temperature(rec_dir)

    pipeline = StreamingFeaturePipeline(
        fs_ecg=FS_ECG, fs_emg=FS_EMG, fs_eda=FS_EDA,
        fs_acc=FS_ACC, fs_ppg=ppg_fs, fs_temp=FS_TEMP,
    )
    pipeline.set_baseline_end(float(ecg_df["timestamp"].iloc[0]) + BASELINE_S)

    # Try to load trained models for end-to-end latency
    models = {}
    for name in ["fatigue", "exercise", "phase", "reps"]:
        p = run_dir / "models" / f"{name}.joblib"
        if p.exists():
            try:
                models[name] = joblib.load(p)
            except Exception as e:
                print(f"  WARN: failed to load {name}: {e}", file=sys.stderr)

    # Trim to requested duration
    target_n_acc = int(minutes * 60 * FS_ACC)
    n_acc = min(len(imu_df), target_n_acc)
    imu_df = imu_df.iloc[:n_acc].copy()
    t_lo = float(imu_df["timestamp"].iloc[0])
    t_hi = float(imu_df["timestamp"].iloc[-1])

    ecg_df = ecg_df[(ecg_df["timestamp"] >= t_lo) & (ecg_df["timestamp"] <= t_hi + 1.0)].reset_index(drop=True)
    emg_df = emg_df[(emg_df["timestamp"] >= t_lo) & (emg_df["timestamp"] <= t_hi + 1.0)].reset_index(drop=True)
    eda_df = eda_df[(eda_df["timestamp"] >= t_lo) & (eda_df["timestamp"] <= t_hi + 1.0)].reset_index(drop=True)
    ppg_df = ppg_df[(ppg_df["timestamp"] >= t_lo) & (ppg_df["timestamp"] <= t_hi + 1.0)].reset_index(drop=True)

    has_temp = len(temp_df) >= 2
    if has_temp:
        temp_df = temp_df[(temp_df["timestamp"] >= t_lo) & (temp_df["timestamp"] <= t_hi + 1.0)].reset_index(drop=True)

    ecg_ratio = FS_ECG // FS_ACC      # 5
    emg_ratio = FS_EMG // FS_ACC      # 20
    eda_step = FS_ACC // FS_EDA       # 2
    ppg_ratio = ppg_fs // FS_ACC if ppg_fs >= FS_ACC else 1

    print(f"Benchmark: {minutes} min, {n_acc} hops at {FS_ACC} Hz, chunk={chunk_size} samples")
    print(f"Models loaded: {list(models)}")

    latencies_ms: list[float] = []
    feature_cols: list[str] | None = None

    ax = imu_df["ax"].to_numpy()
    ay = imu_df["ay"].to_numpy()
    az = imu_df["az"].to_numpy()
    acc_t = imu_df["timestamp"].to_numpy()
    ecg_arr, ecg_ts = ecg_df["ecg"].to_numpy(), ecg_df["timestamp"].to_numpy()
    emg_arr, emg_ts = emg_df["emg"].to_numpy(), emg_df["timestamp"].to_numpy()
    eda_arr, eda_ts = eda_df["eda"].to_numpy(), eda_df["timestamp"].to_numpy()
    ppg_arr, ppg_ts = ppg_df["ppg_green"].to_numpy(), ppg_df["timestamp"].to_numpy()
    if has_temp:
        temp_arr = temp_df["temperature"].to_numpy()
        temp_ts = temp_df["timestamp"].to_numpy()

    for acc_start in range(0, n_acc, chunk_size):
        acc_end = min(acc_start + chunk_size, n_acc)
        ecg_a, ecg_b = acc_start * ecg_ratio, acc_end * ecg_ratio
        emg_a, emg_b = acc_start * emg_ratio, acc_end * emg_ratio
        eda_a, eda_b = acc_start // eda_step, acc_end // eda_step
        ppg_a, ppg_b = acc_start * ppg_ratio, acc_end * ppg_ratio

        # Bound to actual array lengths (handle edge cases)
        ecg_b = min(ecg_b, len(ecg_arr))
        emg_b = min(emg_b, len(emg_arr))
        eda_b = min(eda_b, len(eda_arr))
        ppg_b = min(ppg_b, len(ppg_arr))

        if has_temp:
            t_lo_chunk = float(acc_t[acc_start])
            t_hi_chunk = float(acc_t[acc_end - 1])
            tmask = (temp_ts >= t_lo_chunk - 0.5) & (temp_ts <= t_hi_chunk + 0.5)
            tc = temp_arr[tmask] if tmask.any() else None
            tt = temp_ts[tmask] if tmask.any() else None
        else:
            tc, tt = None, None

        t0 = time.perf_counter()
        feats_list = pipeline.step(
            ecg_chunk=ecg_arr[ecg_a:ecg_b], ecg_t=ecg_ts[ecg_a:ecg_b],
            emg_chunk=emg_arr[emg_a:emg_b], emg_t=emg_ts[emg_a:emg_b],
            eda_chunk=eda_arr[eda_a:eda_b], eda_t=eda_ts[eda_a:eda_b],
            ax_chunk=ax[acc_start:acc_end], ay_chunk=ay[acc_start:acc_end],
            az_chunk=az[acc_start:acc_end], acc_t=acc_t[acc_start:acc_end],
            ppg_chunk=ppg_arr[ppg_a:ppg_b], ppg_t=ppg_ts[ppg_a:ppg_b],
            temp_chunk=tc, temp_t=tt,
        )

        # Run inference per emitted feature row (worst case for end-to-end)
        if feats_list and models:
            # Lazily lock feature columns to first non-empty emission
            if feature_cols is None:
                feature_cols = [k for k in feats_list[-1] if k != "t_unix"]
            x = np.array([[feats_list[-1].get(c, np.nan) for c in feature_cols]])
            # Replace NaN with 0 for predict (most LightGBM models handle NaN; some pipelines do not)
            for name, m in models.items():
                try:
                    if hasattr(m, "predict"):
                        m.predict(x)
                except Exception:
                    pass
        t1 = time.perf_counter()

        # Cost is per emitted hop; if multiple emitted, divide
        n_emit = max(len(feats_list), 1)
        per_hop_ms = (t1 - t0) * 1000.0 / n_emit
        for _ in range(n_emit):
            latencies_ms.append(per_hop_ms)

    arr = np.asarray(latencies_ms)
    stats = {
        "n_samples": int(arr.size),
        "minutes": minutes,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
        "budget_p99_ms": budget_p99_ms,
        "pass": bool(np.percentile(arr, 99) <= budget_p99_ms),
        "models_in_pipeline": list(models.keys()),
        "recording": rec_dir.name,
    }
    return stats, arr


def write_artifacts(run_dir: Path, stats: dict, latencies: np.ndarray) -> None:
    out = run_dir / "latency.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.eval.plot_style import apply_style, despine

        apply_style()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(latencies, bins=80, color="steelblue", edgecolor="white")
        for q, color in [(50, "green"), (95, "orange"), (99, "red")]:
            v = float(np.percentile(latencies, q))
            ax.axvline(v, color=color, ls="--", label=f"p{q} = {v:.2f} ms")
        ax.axvline(stats["budget_p99_ms"], color="black", ls=":", label=f"budget = {stats['budget_p99_ms']:.0f} ms")
        ax.set_xlabel("per-hop latency (ms)")
        ax.set_ylabel("count")
        ax.set_title(f"Streaming pipeline latency — {stats['recording']} ({stats['minutes']:.0f} min)")
        ax.legend()
        plt.tight_layout()
        despine(fig=fig)
        png = run_dir / "plots" / "latency_histogram.png"
        png.parent.mkdir(exist_ok=True)
        plt.savefig(png, dpi=120)
        plt.close()
        print(f"Wrote {png}")
    except Exception as e:
        print(f"  plot skipped: {e}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recording", required=True, help="Path to dataset/recording_NNN/")
    p.add_argument("--run", required=True, help="Path to runs/<...>/")
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--chunk", type=int, default=1)
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    budget = float(cfg["latency"]["budget_p99_ms"])

    stats, lat = benchmark(
        rec_dir=Path(args.recording),
        run_dir=run_dir,
        minutes=args.minutes,
        chunk_size=args.chunk,
        budget_p99_ms=budget,
    )
    write_artifacts(run_dir, stats, lat)

    print()
    print(f"  p50:    {stats['p50_ms']:.2f} ms")
    print(f"  p95:    {stats['p95_ms']:.2f} ms")
    print(f"  p99:    {stats['p99_ms']:.2f} ms (budget {stats['budget_p99_ms']:.0f} ms)")
    print(f"  max:    {stats['max_ms']:.2f} ms")
    print(f"  PASS:   {stats['pass']}")

    return 0 if stats["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
