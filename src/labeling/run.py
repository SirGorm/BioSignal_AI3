"""
Orchestrates offline labeling for one recording end-to-end.

Workflow per recording
----------------------
1. Load metadata.json → get PPG fs, kinect_sets count.
2. Load Participants.xlsx row for this recording → subject_id, exercises, RPE.
3. Parse markers.json → canonical set boundaries (start/end unix times + rep times).
4. Validate: markers set count == Participants.xlsx expected 12 sets.
   If markers have > 12 sets, call select_canonical_sets() to filter.
5. Load all biosignals → validate Unix timestamps.
6. Check EDA dynamic range per recording. Criterion (from inspections/recording_014_aligned):
   if EDA std < 1e-7 S OR EDA range < 5e-8 S → mark eda_status="unusable", set EDA
   column to NaN (Greco et al. 2016 — wrist EDA tonic level 1–10 µS; these recordings
   sit ≥40× below that floor, consistent with electrode-skin contact failure).
7. Check EMG baseline window: verify that markers["Set:1_Start"].unix_time - bio_t0 >= 90 s.
   Flag if baseline < 90 s.
8. Build 100 Hz unified grid from bio_t0 to bio_tend.
9. For each canonical set, load the corresponding joints JSON → compute
   primary joint angle time series.
10. Compute phase labels and joint-angle-derived rep counts.
11. Cross-validate rep counts: |joint_reps - marker_reps| <= 1; flag otherwise.
12. Build the aligned_features DataFrame via align.build_aligned_dataframe().
13. Write to data/labeled/<recording_id>/aligned_features.parquet.
14. Write quality_report.md.

Halt conditions (recorded; other recordings still processed)
------------------------------------------------------------
- Required CSV missing.
- Markers set count != 12 and cannot be reconciled.
- Biosignal timestamp < 1e9.
- Joint reps differ from marker reps by > 1 for any set (flagged, not halted).

References
----------
- Bulling A et al. (2014). "A tutorial on human activity recognition using
  body-worn inertial sensors." ACM Comput. Surv. 46(3), 33:1-33:33.
- De Luca CJ (1997). "The use of surface electromyography in biomechanics."
  J. Appl. Biomech. 13(2), 135-163.
- Greco A et al. (2016). "cvxEDA: a convex optimization approach to
  electrodermal activity processing." IEEE Trans. Biomed. Eng. 63(4), 797-804.
  (wrist EDA tonic-level reference 1–10 µS; electrode-contact failure mode.)
- Maeda Y et al. (2011). "Noninvasive skin temperature measurement of the
  wrist using a wearable device." Physiol. Meas. — [REF NEEDED: Maeda 2011
  exact citation for wrist temperature measurement].
- Task Force of the European Society of Cardiology (1996). "Heart rate
  variability: standards of measurement." Eur. Heart J. 17, 354-381.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.loaders import (
    load_biosignal,
    load_temperature,
    load_imu,
    load_metadata,
    load_all_biosignals,
)
from src.data.participants import load_participants, get_recording_info
from src.labeling.markers import parse_markers, select_canonical_sets
from src.labeling.joint_angles import (
    load_joint_angles_for_set,
    compute_angle_derivatives,
    label_phase,
    label_phase_from_acc,
    count_reps_from_angles,
    count_reps_from_acc,
)

# Exercises that get phase labels from wrist-IMU vertical velocity
# (`label_phase_from_acc`) instead of Kinect joint angle. Only bench is
# acc-driven: Kinect cannot see the elbow under the lifter's torso, so
# joint-angle based phase is unrecoverable for that exercise.
# Squat/deadlift/pullup all use Kinect joint angle (knee, hip, elbow
# respectively) — these are well-tracked when the lifter faces the camera.
_ACC_PHASE_EXERCISES = {"benchpress"}

# Exercises for which we accept an acc-based phase fallback when the
# Kinect joint-angle range is too small (Kinect mis-tracked or partially
# occluded the joint). Pullup is excluded — wrist barely translates on
# pullup, so acc velocity is near zero. For squat and deadlift, the
# wrist follows the body / bar vertically, so acc is a reliable fallback.
_ACC_PHASE_FALLBACK_EXERCISES = {"squat", "deadlift"}

# Per-recording manual exclusions of original marker set numbers.
# Recorded as feedback after visual QC — these sets had bad set
# boundaries or noisy marker placement that select_canonical_sets()
# alone could not detect. Markers listed here are removed BEFORE
# canonical selection runs. Recordings with > 12 - len(excluded)
# clean markers will still produce 12 canonical sets; otherwise the
# pipeline accepts fewer.
# Per-recording blacklists curated from QC plot review (post-labeling).
# Entries here apply AFTER labeling — they rewrite the parquet to mark
# specific (recording_id, canonical set_number) pairs as bad without
# re-running the labeler. See aligned_df post-processing in run().
#
# CANONICAL set numbers (1..N as they appear in the parquet's set_number
# column and in inspections/segmentation_qc/*/per_set/set_NN_*.png), NOT
# original marker numbers.
#
# PHASE_REPS: phase_label set to "unknown" and rep_density_hz set to 0
#   (rep_count_in_set set to NaN) for these sets — they are excluded from
#   phase-head training and reps-head training. Other heads (exercise,
#   fatigue) still see them.
# ALL_HEADS: in_active_set set to False — entire set excluded from every
#   training task. Use when the joint data and the marker sequence both
#   look broken (e.g. wrong recording attached to a set window).
_PHASE_REPS_BLACKLIST: set[tuple[str, int]] = {
    ("recording_003", 11),
    ("recording_006", 9),
    ("recording_007", 7), ("recording_007", 8), ("recording_007", 9),
    ("recording_008", 3),
    ("recording_008", 9), ("recording_008", 10), ("recording_008", 11),
    ("recording_008", 12),
    ("recording_009", 5), ("recording_009", 6),
    ("recording_011", 3),
    ("recording_011", 8), ("recording_011", 9),
    ("recording_011", 10), ("recording_011", 11),
    ("recording_013", 7),
    ("recording_014", 7), ("recording_014", 11), ("recording_014", 12),
}
_ALL_HEADS_BLACKLIST: set[tuple[str, int]] = {
    ("recording_009", 10), ("recording_009", 11), ("recording_009", 12),
}


_MANUAL_MARKER_EXCLUSIONS: dict[str, set[int]] = {
    # rec_008: orig markers 8 and 9 are both aborted pullup attempts
    # (2 reps/15s and 3 reps/15s respectively) preceding the real pullup
    # block at orig 10-13. Drop both → 11 canonical sets (1-7, 10-13).
    "recording_008": {8, 9},
    # rec_009: orig markers 10-14 are all bad — aborted attempts (10-11)
    # and wrong recordings (12-14) per QC review. Drop all → 9 canonical
    # sets (orig 1-9 only). This recording has no pullup data.
    "recording_009": {10, 11, 12, 13, 14},
}


def _align_to_canonical(
    full_list: list,
    canonical_sets,
) -> list:
    """Pick Participants.xlsx entries indexed by orig marker number.

    Participants.xlsx slot N (1-indexed) corresponds to original marker
    number N — not to canonical position. After manual exclusions or
    select_canonical_sets() drops aborted attempts, those two diverge,
    and a position-based lookup mislabels canonical sets that come from
    later orig markers. We use sm.set_num - 1 as the slot index instead.
    """
    out = []
    for sm in canonical_sets:
        slot = sm.set_num - 1
        if 0 <= slot < len(full_list):
            out.append(full_list[slot])
        else:
            out.append(None)
    return out


from src.labeling.align import (
    make_100hz_grid,
    build_set_info_array,
    build_aligned_dataframe,
)


# ---------------------------------------------------------------------------
# EDA quality check
# ---------------------------------------------------------------------------

_EDA_STD_THRESHOLD = 1e-7    # S — below this: sensor floor
_EDA_RANGE_THRESHOLD = 5e-8  # S — below this: no dynamic range


def _check_eda_quality(eda_df: pd.DataFrame) -> tuple[str, float, float]:
    """Return (status, std, range) for EDA signal.

    status is 'unusable' if std < 1e-7 S or range < 5e-8 S,
    otherwise 'ok'.

    Threshold rationale: Greco et al. (2016) report wrist EDA tonic level
    of 1–10 µS. Recordings with std < 0.1 µS and range < 0.05 µS are
    consistent with electrode-skin contact failure (sensor ADC noise floor).
    """
    vals = eda_df["eda"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return "unusable", 0.0, 0.0
    std = float(np.std(vals))
    rng = float(np.max(vals) - np.min(vals))
    if std < _EDA_STD_THRESHOLD or rng < _EDA_RANGE_THRESHOLD:
        return "unusable", std, rng
    return "ok", std, rng


# ---------------------------------------------------------------------------
# Quality report writer
# ---------------------------------------------------------------------------

def _write_quality_report(
    out_dir: Path,
    recording_id: str,
    subject_id: str,
    n_sets_markers: int,
    n_sets_participants: int,
    n_sets_metadata: int,
    canonical_sets,
    rep_diagnostics: list[dict],
    joint_coverage: dict[int, dict],
    temp_missing: bool,
    eda_status: str,
    eda_std: float,
    eda_range: float,
    emg_baseline_s: float,
    flags: list[str],
    exercises: list,
    rpe_list: list,
) -> None:
    """Write quality_report.md for one recording."""
    lines = []
    lines.append(f"# Quality Report — {recording_id} ({subject_id})")
    lines.append("")
    lines.append("## Set count validation")
    lines.append(f"- markers.json: **{n_sets_markers}** sets")
    lines.append(f"- Participants.xlsx: **{n_sets_participants}** sets (expected 12)")
    lines.append(f"- metadata.json total_kinect_sets: **{n_sets_metadata}**")
    lines.append(f"- Canonical sets used: **{len(canonical_sets)}**")
    lines.append("")

    lines.append("## Set durations and rep counts")
    lines.append("| Canonical set | Markers set_num | Exercise | Duration (s) | Marker reps | Joint reps | Flag |")
    lines.append("|---|---|---|---|---|---|---|")
    aligned_exercises = _align_to_canonical(exercises, canonical_sets)
    aligned_rpe = _align_to_canonical(rpe_list, canonical_sets)
    for pos, sm in enumerate(canonical_sets):
        exer = aligned_exercises[pos] if aligned_exercises[pos] is not None else "?"
        rpe = aligned_rpe[pos] if aligned_rpe[pos] is not None else "?"
        jcov = joint_coverage.get(sm.set_num, {})
        joint_reps = jcov.get("joint_reps", "N/A")
        marker_reps = sm.n_reps
        flag = ""
        if isinstance(joint_reps, int):
            diff = abs(joint_reps - marker_reps)
            if diff > 1:
                flag = f"REP_MISMATCH(diff={diff})"
        if marker_reps < 6 or marker_reps > 12:
            flag += f" REP_COUNT_UNUSUAL({marker_reps})"
        lines.append(
            f"| {pos + 1} | {sm.set_num} | {exer} | "
            f"{sm.duration_s:.1f} | {marker_reps} | {joint_reps} | {flag} |"
        )
    lines.append("")

    lines.append("## Joint angle coverage per set")
    lines.append("| Canonical set | Markers set_num | Joint file found | Frames | NaN angle % |")
    lines.append("|---|---|---|---|---|")
    for pos, sm in enumerate(canonical_sets):
        jcov = joint_coverage.get(sm.set_num, {})
        found = jcov.get("found", False)
        frames = jcov.get("n_frames", 0)
        nan_pct = jcov.get("nan_angle_pct", 100.0)
        lines.append(
            f"| {pos + 1} | {sm.set_num} | {found} | {frames} | {nan_pct:.1f}% |"
        )
    lines.append("")

    lines.append("## Modality status")
    lines.append(f"- temperature.csv: {'**MISSING/EMPTY** (NaN column emitted)' if temp_missing else 'present'}")
    eda_flag_str = (
        f"**UNUSABLE** (std={eda_std:.3e} S, range={eda_range:.3e} S — sensor floor; "
        f"threshold: std<1e-7 S OR range<5e-8 S; Greco et al. 2016). "
        f"EDA column set to all-NaN in parquet."
    ) if eda_status == "unusable" else f"ok (std={eda_std:.3e} S, range={eda_range:.3e} S)"
    lines.append(f"- EDA (`eda_status`): {eda_flag_str}")
    baseline_flag = "" if emg_baseline_s >= 90.0 else f" **WARNING: baseline only {emg_baseline_s:.1f} s < 90 s**"
    lines.append(f"- EMG/EDA baseline window: {emg_baseline_s:.1f} s before first set{baseline_flag}")
    lines.append("")

    if flags:
        lines.append("## Flags requiring manual review")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("## Flags")
        lines.append("None.")
        lines.append("")

    lines.append("## Methodological notes")
    lines.append(
        "- Set boundaries taken from markers.json (Set:N_Start / Set_N_End unix_time) "
        "rather than acc-magnitude segmentation. On rec_012, acc-magnitude "
        "over-segmented to 42 segments vs 12 true sets, confirming markers.json as "
        "the authoritative source (Bulling et al. 2014)."
    )
    lines.append(
        "- Joint frame Unix time anchored via: "
        "`t_frame = markers[Set:N_Start][unix_time] + frame_id / 30.0` "
        "(Azure Kinect DK ≈ 30 fps; internal timestamp_usec = 0 in all recordings)."
    )
    lines.append(
        "- Phase labeling: concentric/eccentric from sign of angular velocity "
        "(5 deg/s isometric threshold; De Luca 1997)."
    )
    lines.append(
        "- Bilateral joint angle averaged (left + right) to reduce single-side "
        "tracking failures (Fukuchi et al. 2018)."
    )
    lines.append(
        "- Temperature NaN-tolerance: empty temperature.csv returns an all-NaN "
        "column. Feature extractor must handle NaN input for temperature."
    )
    lines.append(
        "- EDA unusable criterion: std < 1e-7 S OR range < 5e-8 S, consistent with "
        "electrode-skin contact failure (Greco et al. 2016). All 9 recordings in "
        "dataset_aligned/ meet this criterion."
    )
    lines.append(
        "- EMG baseline: first 90-120 s of recording (verified by "
        "markers[Set:1_Start].unix_time - bio_t0 >= 90 s; CLAUDE.md). "
        "Per-subject normalization required due to inter-subject EDA baseline "
        "amplitude variation >30% (Greco et al. 2016)."
    )
    lines.append("")
    lines.append("## References")
    lines.append(
        "- Bulling A et al. (2014). A tutorial on human activity recognition using "
        "body-worn inertial sensors. ACM Comput. Surv. 46(3), 33:1-33:33."
    )
    lines.append(
        "- De Luca CJ (1997). The use of surface electromyography in biomechanics. "
        "J. Appl. Biomech. 13(2), 135-163."
    )
    lines.append(
        "- Fukuchi CA et al. (2018). A public dataset of overground and treadmill "
        "walking kinematics and kinetics in healthy individuals. PeerJ 6:e4640."
    )
    lines.append(
        "- Greco A et al. (2016). cvxEDA: a convex optimization approach to "
        "electrodermal activity processing. IEEE Trans. Biomed. Eng. 63(4), 797-804."
    )
    lines.append(
        "- Task Force of the European Society of Cardiology (1996). Heart rate "
        "variability: standards of measurement. Eur. Heart J. 17, 354-381."
    )
    lines.append(
        "- Maeda Y et al. (2011). [REF NEEDED: Maeda 2011 exact citation "
        "for wrist skin temperature]."
    )

    report_path = out_dir / "quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-recording labeling
# ---------------------------------------------------------------------------

def label_one_recording(
    rec_dir: Path,
    participants_data: dict[int, dict],
    out_base: Path,
    expected_n_sets: int = 12,
) -> dict:
    """Label one recording and write aligned_features.parquet + quality_report.md.

    Parameters
    ----------
    rec_dir:           Path to dataset_aligned/recording_NNN/.
    participants_data: Parsed Participants.xlsx (from load_participants()).
    out_base:          Root output directory (data/labeled/).
    expected_n_sets:   Canonical set count expected per recording.

    Returns
    -------
    dict with keys:
        'recording_id', 'status', 'subject_id', 'n_sets', 'flags',
        'error' (if status == 'error'), 'active_minutes', 'eda_status'
    """
    rec_id = rec_dir.name  # e.g. 'recording_012'
    result = {
        "recording_id": rec_id,
        "status": "ok",
        "subject_id": "unknown",
        "n_sets": 0,
        "flags": [],
        "error": None,
        "active_minutes": 0.0,
        "eda_status": "unknown",
    }

    try:
        # ---------------------------------------------------------------
        # Step 1: metadata
        # ---------------------------------------------------------------
        meta = load_metadata(rec_dir)
        recording_num = int(rec_id.split("_")[1])
        ppg_fs = meta.get("sampling_rates", {}).get("ppg", 100)
        n_sets_metadata = meta.get("total_kinect_sets", 0)
        bio_start_unix = meta.get("recording_start_unix_time",
                                   meta.get("data_start_unix_time", 0.0))

        # ---------------------------------------------------------------
        # Step 2: Participants.xlsx
        # ---------------------------------------------------------------
        rec_info = get_recording_info(participants_data, recording_num)
        if rec_info is None:
            result["flags"].append(
                f"Recording {recording_num} not found in Participants.xlsx. "
                "Cannot assign exercise or RPE labels."
            )
            exercises = [None] * 12
            rpe_list = [None] * 12
            subject_id = "unknown"
        else:
            subject_id = rec_info["subject_id"]
            exercises = rec_info["exercises"]
            rpe_list = rec_info["rpe"]

        result["subject_id"] = subject_id
        n_sets_participants = sum(1 for e in exercises if e is not None)

        # ---------------------------------------------------------------
        # Step 3: markers.json
        # ---------------------------------------------------------------
        all_sets = parse_markers(rec_dir / "markers.json")
        n_sets_markers = len(all_sets)

        # Apply manual marker exclusions (post-QC overrides) before any
        # automatic canonical selection.
        manual_excl = _MANUAL_MARKER_EXCLUSIONS.get(rec_id, set())
        if manual_excl:
            before = len(all_sets)
            all_sets = [s for s in all_sets if s.set_num not in manual_excl]
            removed = before - len(all_sets)
            result["flags"].append(
                f"Manually excluded {removed} marker set(s) "
                f"{sorted(manual_excl)} per QC review."
            )
            n_sets_markers = len(all_sets)

        # ---------------------------------------------------------------
        # Step 4: set count validation and canonical selection
        # ---------------------------------------------------------------
        # Always run the aborted-set filter — even after manual exclusions,
        # zero-rep / very-short attempts must be removed.
        canonical_sets = select_canonical_sets(
            all_sets,
            expected_n=expected_n_sets,
            min_reps=3,
            min_duration_s=10.0,
        )
        if len(canonical_sets) != n_sets_markers:
            result["flags"].append(
                f"markers.json had {n_sets_markers} sets after manual exclusion; "
                f"select_canonical_sets dropped {n_sets_markers - len(canonical_sets)} "
                f"aborted/short attempt(s) → {len(canonical_sets)} canonical sets."
            )
        if len(canonical_sets) < expected_n_sets:
            result["flags"].append(
                f"Recording has only {len(canonical_sets)} canonical sets "
                f"(< expected {expected_n_sets}); participants.xlsx slots "
                f"beyond this index will be ignored."
            )

        if len(canonical_sets) > expected_n_sets:
            # We over-filtered or under-filtered — something is wrong
            msg = (
                f"HALT: too many canonical sets. "
                f"markers={n_sets_markers}, after filtering={len(canonical_sets)}, "
                f"expected={expected_n_sets}."
            )
            result["status"] = "error"
            result["error"] = msg
            result["flags"].append(msg)
            _write_partial_quality_report(out_base, rec_id, subject_id, result["flags"])
            return result

        result["n_sets"] = len(canonical_sets)

        # ---------------------------------------------------------------
        # Step 5: load biosignals
        # ---------------------------------------------------------------
        ecg_df = load_biosignal(rec_dir, "ecg", "ecg")
        emg_df = load_biosignal(rec_dir, "emg", "emg")
        eda_df = load_biosignal(rec_dir, "eda", "eda")
        ppg_green_df = load_biosignal(rec_dir, "ppg_green", "ppg_green")
        imu_df = load_imu(rec_dir)
        temp_df = load_temperature(rec_dir)
        temp_missing = len(temp_df) == 0

        if temp_missing:
            result["flags"].append("temperature.csv is empty — temperature column set to NaN.")

        # ---------------------------------------------------------------
        # Step 6: EDA quality check
        # ---------------------------------------------------------------
        eda_status, eda_std, eda_range = _check_eda_quality(eda_df)
        result["eda_status"] = eda_status
        if eda_status == "unusable":
            result["flags"].append(
                f"EDA unusable: std={eda_std:.3e} S, range={eda_range:.3e} S "
                f"(sensor floor — electrode-skin contact failure; Greco et al. 2016). "
                f"EDA column set to all-NaN in parquet."
            )
            # NaN out the EDA values
            eda_df = eda_df.copy()
            eda_df["eda"] = np.nan

        # ---------------------------------------------------------------
        # Step 7: EMG baseline check
        # ---------------------------------------------------------------
        bio_t0 = float(ecg_df["timestamp"].iloc[0])
        bio_tend = float(ecg_df["timestamp"].iloc[-1])

        # Find Set:1_Start time from the first canonical set
        first_set_start = canonical_sets[0].start_unix
        emg_baseline_s = first_set_start - bio_t0
        if emg_baseline_s < 90.0:
            result["flags"].append(
                f"EMG baseline window only {emg_baseline_s:.1f} s "
                f"(expected >= 90 s per CLAUDE.md). "
                f"Per-subject normalization may be unreliable."
            )

        print(f"  [{rec_id}] bio_t0={bio_t0:.3f}, bio_tend={bio_tend:.3f}, "
              f"duration={bio_tend - bio_t0:.0f}s")
        print(f"  [{rec_id}] First 3 ECG timestamps: "
              f"{ecg_df['timestamp'].head(3).tolist()}")
        print(f"  [{rec_id}] EDA status: {eda_status} "
              f"(std={eda_std:.3e}, range={eda_range:.3e})")
        print(f"  [{rec_id}] EMG baseline: {emg_baseline_s:.1f}s")

        # ---------------------------------------------------------------
        # Step 8: 100 Hz grid
        # ---------------------------------------------------------------
        grid_t = make_100hz_grid(bio_t0, bio_tend)
        print(f"  [{rec_id}] Grid: {len(grid_t)} samples at 100 Hz "
              f"({len(grid_t) / 100 / 60:.1f} min)")

        # ---------------------------------------------------------------
        # Step 9 + 10: joint angles per set
        # ---------------------------------------------------------------
        joint_angle_dfs: dict[int, Optional[pd.DataFrame]] = {}
        joint_coverage: dict[int, dict] = {}
        rep_diagnostics: list[dict] = []

        # Slot-based alignment: orig marker N → Participants.xlsx slot N.
        # Differs from canonical position whenever sets are excluded.
        aligned_exercises = _align_to_canonical(exercises, canonical_sets)
        aligned_rpe = _align_to_canonical(rpe_list, canonical_sets)

        for pos_idx, sm in enumerate(canonical_sets):
            exer = aligned_exercises[pos_idx]
            if exer is None:
                exer = "unknown"

            # ---------- Bench: acc-based phase override ----------
            # Kinect cannot see the elbow under the lifter's torso on
            # bench, so primary_joint_angle_deg is mostly NaN. We build
            # a 100 Hz jdf directly from wrist IMU and derive phase
            # from the sign of vertical velocity (Karantonis et al. 2006).
            if exer in _ACC_PHASE_EXERCISES:
                imu_mask = (
                    (imu_df["timestamp"] >= sm.start_unix)
                    & (imu_df["timestamp"] <= sm.end_unix)
                )
                imu_set = imu_df.loc[imu_mask].sort_values("timestamp")
                if len(imu_set) >= 50:
                    ax_arr = imu_set["ax"].to_numpy(dtype=float)
                    ay_arr = imu_set["ay"].to_numpy(dtype=float)
                    az_arr = imu_set["az"].to_numpy(dtype=float)
                    ts_arr = imu_set["timestamp"].to_numpy(dtype=float)
                    acc_phase = label_phase_from_acc(
                        ax_arr, ay_arr, az_arr, ts_arr,
                        exer,
                        fs=100.0,
                        target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
                    )
                    # Acc-based rep timing — anchored to marker count.
                    # This replaces the previous "marker-based fill in
                    # build_set_info_array" path so bench reps now have
                    # precise timing from wrist vertical velocity peaks.
                    rep_series_acc = count_reps_from_acc(
                        ax_arr, ay_arr, az_arr, ts_arr,
                        fs=100.0,
                        target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
                    )
                    acc_rep_count = rep_series_acc.to_numpy(dtype=int)
                    acc_reps_max = int(acc_rep_count.max()) if len(acc_rep_count) else 0
                    jdf_acc = pd.DataFrame({
                        "t_unix": ts_arr,
                        "primary_joint_angle_deg": np.nan,
                        "joint_velocity_deg_s": np.nan,
                        "joint_accel_deg_s2": np.nan,
                        "phase_label": acc_phase,
                        "rep_count_in_set": acc_rep_count,
                    })
                    joint_angle_dfs[sm.set_num] = jdf_acc
                    joint_coverage[sm.set_num] = {
                        "found": True,
                        "n_frames": len(jdf_acc),
                        "nan_angle_pct": 100.0,  # by design (no joint angle)
                        "joint_reps": acc_reps_max,
                    }
                    rep_diagnostics.append({
                        "set_num": sm.set_num,
                        "canonical_pos": pos_idx + 1,
                        "marker_reps": sm.n_reps,
                        "joint_reps": acc_reps_max,
                        "diff": abs(acc_reps_max - sm.n_reps),
                    })
                    continue
                else:
                    result["flags"].append(
                        f"Set {sm.set_num} (bench, canonical {pos_idx + 1}): "
                        f"only {len(imu_set)} IMU samples in window — "
                        f"falling back to joint-angle path."
                    )

            jdf = load_joint_angles_for_set(
                rec_dir,
                sm.set_num,
                exer,
                sm.start_unix,
                set_end_unix=sm.end_unix,
                kinect_fps=30.0,
            )

            if jdf is None or len(jdf) == 0:
                joint_angle_dfs[sm.set_num] = None
                joint_coverage[sm.set_num] = {
                    "found": False,
                    "n_frames": 0,
                    "nan_angle_pct": 100.0,
                    "joint_reps": "N/A",
                }
                if sm.n_reps > 0:
                    result["flags"].append(
                        f"Set {sm.set_num} (canonical {pos_idx + 1}): "
                        f"joint file missing — phase/rep labels from markers only."
                    )
                continue

            # Filter to within set time window
            jdf = jdf[
                (jdf["t_unix"] >= sm.start_unix) &
                (jdf["t_unix"] <= sm.end_unix)
            ].copy()

            n_frames = len(jdf)
            nan_pct = float(jdf["primary_joint_angle_deg"].isna().mean() * 100)

            # Compute derivatives and phase
            if n_frames >= 10:
                vel, accel = compute_angle_derivatives(
                    jdf["primary_joint_angle_deg"],
                    jdf["t_unix"],
                    window=2,
                )
                jdf["joint_velocity_deg_s"] = vel.values
                jdf["joint_accel_deg_s2"] = accel.values

                phase_series = label_phase(
                    jdf["primary_joint_angle_deg"],
                    jdf["t_unix"],
                    exer,
                    target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
                )
                jdf["phase_label"] = phase_series.values

                # Fallback to acc-based phase when joint-angle path returns
                # mostly "unknown" — Kinect failed but the wrist still
                # translated vertically (squat/deadlift). For pullup, acc
                # is unreliable since the wrist barely moves.
                unknown_frac = float((jdf["phase_label"] == "unknown").mean())
                used_acc_fallback = False
                if (
                    exer in _ACC_PHASE_FALLBACK_EXERCISES
                    and unknown_frac > 0.5
                ):
                    imu_mask_fb = (
                        (imu_df["timestamp"] >= sm.start_unix)
                        & (imu_df["timestamp"] <= sm.end_unix)
                    )
                    imu_set_fb = imu_df.loc[imu_mask_fb].sort_values("timestamp")
                    if len(imu_set_fb) >= 50:
                        acc_phase_fb = label_phase_from_acc(
                            imu_set_fb["ax"].to_numpy(dtype=float),
                            imu_set_fb["ay"].to_numpy(dtype=float),
                            imu_set_fb["az"].to_numpy(dtype=float),
                            imu_set_fb["timestamp"].to_numpy(dtype=float),
                            exer,
                            fs=100.0,
                            target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
                        )
                        # Replace jdf with the IMU-rate frame so phase
                        # labels are dense across the set window.
                        jdf = pd.DataFrame({
                            "t_unix": imu_set_fb["timestamp"].to_numpy(dtype=float),
                            "primary_joint_angle_deg": np.nan,
                            "joint_velocity_deg_s": np.nan,
                            "joint_accel_deg_s2": np.nan,
                            "phase_label": acc_phase_fb,
                            "rep_count_in_set": 0,
                        })
                        used_acc_fallback = True
                        result["flags"].append(
                            f"Set {sm.set_num} (canonical {pos_idx + 1}, "
                            f"{exer}): joint-angle phase {unknown_frac:.0%} "
                            f"unknown — fell back to acc-based phase."
                        )

                # Joint-based rep count (skip if we fell back to acc).
                # Anchored to marker count: markers provide N (manual count
                # is reliable); joint angles provide precise per-rep timing.
                # Returns all-zeros if signal range is too small — caller
                # in align.py treats joint_reps == 0 as "fall back to markers".
                if used_acc_fallback:
                    joint_reps = "ACC"
                else:
                    rep_series = count_reps_from_angles(
                        jdf["primary_joint_angle_deg"],
                        jdf["t_unix"],
                        exer,
                        target_n_reps=sm.n_reps if sm.n_reps > 0 else None,
                    )
                    jdf["rep_count_in_set"] = rep_series.values
                    joint_reps = int(jdf["rep_count_in_set"].max())
            else:
                jdf["joint_velocity_deg_s"] = np.nan
                jdf["joint_accel_deg_s2"] = np.nan
                jdf["phase_label"] = np.nan
                jdf["rep_count_in_set"] = 0
                joint_reps = 0

            # Cross-validate rep count
            # Only flag when marker rep count is > 0 (some early recordings
            # like rec_001/rec_002 have no per-rep markers — comparison is
            # meaningless against zero)
            if isinstance(joint_reps, int):
                diff = abs(joint_reps - sm.n_reps)
            else:
                diff = 0  # acc fallback — rep count from markers, no comparison
            rep_diag = {
                "set_num": sm.set_num,
                "canonical_pos": pos_idx + 1,
                "marker_reps": sm.n_reps,
                "joint_reps": joint_reps,
                "diff": diff,
            }
            rep_diagnostics.append(rep_diag)

            if isinstance(joint_reps, int) and sm.n_reps > 0 and diff > 1:
                result["flags"].append(
                    f"Set {sm.set_num} (canonical {pos_idx + 1}): "
                    f"joint_reps={joint_reps} vs marker_reps={sm.n_reps} "
                    f"(diff={diff} > 1). Flag for manual review."
                )

            joint_coverage[sm.set_num] = {
                "found": True,
                "n_frames": n_frames,
                "nan_angle_pct": nan_pct,
                "joint_reps": joint_reps,
            }
            joint_angle_dfs[sm.set_num] = jdf

            if nan_pct > 1.0:
                result["flags"].append(
                    f"Set {sm.set_num} joint angle: {nan_pct:.1f}% NaN frames."
                )

        # ---------------------------------------------------------------
        # Build a combined joint angle DataFrame for the whole session
        # (only within active sets — NaN for rest periods)
        # ---------------------------------------------------------------
        session_joint_df = _build_session_joint_df(
            joint_angle_dfs, canonical_sets
        )

        # ---------------------------------------------------------------
        # Step 11: set-info array on 100 Hz grid
        # ---------------------------------------------------------------
        exercise_for_set = {
            sm.set_num: (aligned_exercises[pos_idx] or "unknown")
            for pos_idx, sm in enumerate(canonical_sets)
        }

        set_info = build_set_info_array(
            grid_t,
            canonical_sets,
            aligned_exercises,
            aligned_rpe,
            joint_angle_dfs,
            exercise_for_set,
        )

        # Compute total active time
        active_samples = int(set_info["in_active_set"].sum())
        active_minutes = active_samples / 100.0 / 60.0
        result["active_minutes"] = active_minutes

        # ---------------------------------------------------------------
        # Step 12: build aligned DataFrame
        # ---------------------------------------------------------------
        aligned_df = build_aligned_dataframe(
            grid_t=grid_t,
            bio_t0=bio_t0,
            ecg_df=ecg_df,
            emg_df=emg_df,
            eda_df=eda_df,         # already NaN'd if unusable
            ppg_green_df=ppg_green_df,
            imu_df=imu_df,
            temp_df=temp_df if not temp_missing else None,
            set_info=set_info,
            joint_angle_df=session_joint_df,
        )

        # Add identifying columns
        aligned_df.insert(0, "recording_id", rec_id)
        aligned_df.insert(0, "session_id", rec_id)  # alias for tests
        aligned_df.insert(0, "subject_id", subject_id)

        # Add EDA status as a constant column for downstream filtering
        aligned_df["eda_status"] = eda_status

        # ---------------------------------------------------------------
        # Apply curated per-set blacklists from QC review
        # ---------------------------------------------------------------
        phase_reps_blocked = []
        for (bl_rec, bl_set) in _PHASE_REPS_BLACKLIST:
            if bl_rec != rec_id:
                continue
            sel = (aligned_df["set_number"] == bl_set) & \
                  aligned_df["in_active_set"]
            if sel.any():
                aligned_df.loc[sel, "phase_label"] = "unknown"
                aligned_df.loc[sel, "rep_density_hz"] = 0.0
                aligned_df.loc[sel, "rep_count_in_set"] = np.nan
                aligned_df.loc[sel, "rep_index"] = np.nan
                phase_reps_blocked.append(bl_set)

        all_heads_blocked = []
        for (bl_rec, bl_set) in _ALL_HEADS_BLACKLIST:
            if bl_rec != rec_id:
                continue
            sel = aligned_df["set_number"] == bl_set
            if sel.any():
                aligned_df.loc[sel, "in_active_set"] = False
                aligned_df.loc[sel, "phase_label"] = "rest"
                aligned_df.loc[sel, "rep_density_hz"] = 0.0
                aligned_df.loc[sel, "rep_count_in_set"] = np.nan
                aligned_df.loc[sel, "rep_index"] = np.nan
                aligned_df.loc[sel, "rpe_for_this_set"] = np.nan
                aligned_df.loc[sel, "exercise"] = np.nan
                aligned_df.loc[sel, "set_number"] = np.nan
                all_heads_blocked.append(bl_set)

        if phase_reps_blocked:
            print(f"  [{rec_id}] PHASE+REPS blacklist applied to canonical "
                  f"sets {sorted(phase_reps_blocked)}")
            result["flags"].append(
                f"PHASE+REPS blacklist applied to canonical sets "
                f"{sorted(phase_reps_blocked)} (set in_active stays True; "
                f"phase_label=unknown, rep_density=0)"
            )
        if all_heads_blocked:
            print(f"  [{rec_id}] ALL-HEADS blacklist applied to canonical "
                  f"sets {sorted(all_heads_blocked)}")
            result["flags"].append(
                f"ALL-HEADS blacklist applied to canonical sets "
                f"{sorted(all_heads_blocked)} (in_active=False)"
            )

        # ---------------------------------------------------------------
        # Step 13: write parquet
        # ---------------------------------------------------------------
        out_dir = out_base / rec_id
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "aligned_features.parquet"
        aligned_df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"  [{rec_id}] Written {len(aligned_df)} rows to {parquet_path}")

        # ---------------------------------------------------------------
        # Step 14: quality report
        # ---------------------------------------------------------------
        _write_quality_report(
            out_dir=out_dir,
            recording_id=rec_id,
            subject_id=subject_id,
            n_sets_markers=n_sets_markers,
            n_sets_participants=n_sets_participants,
            n_sets_metadata=n_sets_metadata,
            canonical_sets=canonical_sets,
            rep_diagnostics=rep_diagnostics,
            joint_coverage=joint_coverage,
            temp_missing=temp_missing,
            eda_status=eda_status,
            eda_std=eda_std,
            eda_range=eda_range,
            emg_baseline_s=emg_baseline_s,
            flags=result["flags"],
            exercises=exercises,
            rpe_list=rpe_list,
        )
        print(f"  [{rec_id}] Quality report written.")

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["flags"].append(f"EXCEPTION: {exc}")
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Helper: build session-level joint angle DataFrame
# ---------------------------------------------------------------------------

def _build_session_joint_df(
    joint_angle_dfs: dict[int, Optional[pd.DataFrame]],
    canonical_sets,
) -> Optional[pd.DataFrame]:
    """Concatenate per-set joint DataFrames into one session-level DataFrame."""
    parts = []
    for sm in canonical_sets:
        jdf = joint_angle_dfs.get(sm.set_num)
        if jdf is not None and len(jdf) > 0:
            parts.append(jdf)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).sort_values("t_unix")


def _write_partial_quality_report(
    out_base: Path, rec_id: str, subject_id: str, flags: list[str]
) -> None:
    out_dir = out_base / rec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Quality Report — {rec_id} ({subject_id})",
        "",
        "## HALT — processing stopped early",
        "",
    ]
    for f in flags:
        lines.append(f"- {f}")
    (out_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
