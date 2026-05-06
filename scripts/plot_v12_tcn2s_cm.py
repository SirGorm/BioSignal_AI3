"""Confusion matrices for the v12 multi-raw-TCN @ 2s model (exercise + phase).

Reuses load_preds + plot_cm from plot_v13_singletask.py so visual style
stays consistent across thesis figures.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_v13_singletask import plot_cm  # noqa: E402

OUT = ROOT / "runs" / "comparison_v12" / "cm"
OUT.mkdir(parents=True, exist_ok=True)
RUN_DIR = ROOT / "runs" / "optuna_clean_v12eqw-w2s-multi-raw-tcn"

PH_CLASSES = ["concentric", "eccentric"]
EX_CLASSES = ["benchpress", "deadlift", "pullup", "squat"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    plot_cm(RUN_DIR, "exercise", EX_CLASSES,
            "Exercise CM — multi-raw-TCN @ 2 s (multi-task)",
            OUT / "cm_exercise_multi-raw-tcn_2s.png", cmap="Blues")
    plot_cm(RUN_DIR, "phase", PH_CLASSES,
            "Phase CM — multi-raw-TCN @ 2 s (multi-task)",
            OUT / "cm_phase_multi-raw-tcn_2s.png", cmap="Greens")


if __name__ == "__main__":
    main()
