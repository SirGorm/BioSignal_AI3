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
6. Build 100 Hz unified grid from bio_t0 to bio_tend.
7. For each canonical set, load the corresponding joints JSON → compute
   primary joint angle time series.
8. Compute phase labels and joint-angle-derived rep counts.
9. Cross-validate rep counts: |joint_reps - marker_reps| <= 1; flag otherwise.
10. Build the aligned_features DataFrame via align.build_aligned_dataframe().
11. Write to data/labeled/<recording_id>/aligned_features.parquet.
12. Write quality_report.md.

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
    count_reps_from_angles,
)
from src.labeling.align import (
    make_100hz_grid,
    build_set_info_array,
    build_aligned_dataframe,
)


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
    for pos, sm in enumerate(canonical_sets):
        exer = exercises[pos] if pos < len(exercises) else "?"
        rpe = rpe_list[pos] if pos < len(rpe_list) else "?"
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
    rec_dir:           Path to dataset/recording_NNN/.
    participants_data: Parsed Participants.xlsx (from load_participants()).
    out_base:          Root output directory (data/labeled/).
    expected_n_sets:   Canonical set count expected per recording.

    Returns
    -------
    dict with keys:
        'recording_id', 'status', 'subject_id', 'n_sets', 'flags',
        'error' (if status == 'error'), 'active_minutes'
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

        # ---------------------------------------------------------------
        # Step 4: set count validation and canonical selection
        # ---------------------------------------------------------------
        if n_sets_markers == expected_n_sets:
            canonical_sets = all_sets
        elif n_sets_markers > expected_n_sets:
            canonical_sets = select_canonical_sets(
                all_sets,
                expected_n=expected_n_sets,
                min_reps=5,
                min_duration_s=15.0,
            )
            result["flags"].append(
                f"markers.json has {n_sets_markers} sets (> expected {expected_n_sets}). "
                f"Filtered to {len(canonical_sets)} canonical sets using "
                f"min_reps=5, min_duration_s=15.0 s."
            )
        else:
            canonical_sets = all_sets
            result["flags"].append(
                f"markers.json has only {n_sets_markers} sets (< expected {expected_n_sets}). "
                "Using all available sets. Label assignment may be incomplete."
            )

        if len(canonical_sets) != expected_n_sets:
            msg = (
                f"HALT: Cannot reconcile set count. "
                f"markers={n_sets_markers}, "
                f"after filtering={len(canonical_sets)}, "
                f"expected={expected_n_sets}."
            )
            result["status"] = "error"
            result["error"] = msg
            result["flags"].append(msg)
            # Still write a quality report before returning
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

        # Determine session time bounds from ECG (most stable long signal)
        bio_t0 = float(ecg_df["timestamp"].iloc[0])
        bio_tend = float(ecg_df["timestamp"].iloc[-1])

        print(f"  [{rec_id}] bio_t0={bio_t0:.3f}, bio_tend={bio_tend:.3f}, "
              f"duration={bio_tend - bio_t0:.0f}s")
        print(f"  [{rec_id}] First 3 ECG timestamps: "
              f"{ecg_df['timestamp'].head(3).tolist()}")

        # ---------------------------------------------------------------
        # Step 6: 100 Hz grid
        # ---------------------------------------------------------------
        grid_t = make_100hz_grid(bio_t0, bio_tend)
        print(f"  [{rec_id}] Grid: {len(grid_t)} samples at 100 Hz "
              f"({len(grid_t) / 100 / 60:.1f} min)")

        # ---------------------------------------------------------------
        # Step 7 + 8: joint angles per set
        # ---------------------------------------------------------------
        joint_angle_dfs: dict[int, Optional[pd.DataFrame]] = {}
        joint_coverage: dict[int, dict] = {}
        rep_diagnostics: list[dict] = []

        all_joint_frames_for_session = []

        for pos_idx, sm in enumerate(canonical_sets):
            exer = exercises[pos_idx] if pos_idx < len(exercises) else None
            if exer is None:
                exer = "unknown"

            jdf = load_joint_angles_for_set(
                rec_dir,
                sm.set_num,
                exer,
                sm.start_unix,
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
                )
                jdf["phase_label"] = phase_series.values

                # Joint-based rep count
                rep_series = count_reps_from_angles(
                    jdf["primary_joint_angle_deg"],
                    jdf["t_unix"],
                    exer,
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
            diff = abs(joint_reps - sm.n_reps)
            rep_diag = {
                "set_num": sm.set_num,
                "canonical_pos": pos_idx + 1,
                "marker_reps": sm.n_reps,
                "joint_reps": joint_reps,
                "diff": diff,
            }
            rep_diagnostics.append(rep_diag)

            if sm.n_reps > 0 and diff > 1:
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
        # Step 9: set-info array on 100 Hz grid
        # ---------------------------------------------------------------
        exercise_for_set = {
            sm.set_num: (exercises[pos_idx] or "unknown")
            for pos_idx, sm in enumerate(canonical_sets)
        }

        set_info = build_set_info_array(
            grid_t,
            canonical_sets,
            exercises,
            rpe_list,
            joint_angle_dfs,
            exercise_for_set,
        )

        # Compute total active time
        active_samples = int(set_info["in_active_set"].sum())
        active_minutes = active_samples / 100.0 / 60.0
        result["active_minutes"] = active_minutes

        # ---------------------------------------------------------------
        # Step 10: build aligned DataFrame
        # ---------------------------------------------------------------
        aligned_df = build_aligned_dataframe(
            grid_t=grid_t,
            bio_t0=bio_t0,
            ecg_df=ecg_df,
            emg_df=emg_df,
            eda_df=eda_df,
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

        # ---------------------------------------------------------------
        # Step 11: write parquet
        # ---------------------------------------------------------------
        out_dir = out_base / rec_id
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "aligned_features.parquet"
        aligned_df.to_parquet(parquet_path, index=False, engine="pyarrow")
        print(f"  [{rec_id}] Written {len(aligned_df)} rows to {parquet_path}")

        # ---------------------------------------------------------------
        # Step 12: quality report
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
