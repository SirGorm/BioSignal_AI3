"""
CLI entry point for offline labeling pipeline.

Usage
-----
  python -m src.pipeline.label --all
  python -m src.pipeline.label --recording 012
  python -m src.pipeline.label --recording 001 --recording 002

Preconditions (checked at startup)
-----------------------------------
- inspections/ directory must contain at least one findings.md
  (data-inspection skill must have run first per CLAUDE.md).

Data source
-----------
- dataset_aligned/recording_NNN/ is the authoritative training source
  per CLAUDE.md. Biosignals are verbatim copies from dataset/; temperature
  is offset-corrected from dataset_memory/ (see alignment_offsets.json).
- Participants.xlsx is read from the original read-only location:
  C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/

Outputs (per recording)
-----------------------
  data/labeled/recording_<NNN>/aligned_features.parquet
  data/labeled/recording_<NNN>/quality_report.md

Summary
-------
  data/labeled/_summary.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root is two levels up from this file (src/pipeline/label.py → /)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Authoritative data source (per CLAUDE.md)
DATASET_ALIGNED_DIR = REPO_ROOT / "dataset_aligned"

# Participants.xlsx is read-only; lives outside repo
PARTICIPANTS_XLSX = Path(
    "C:/Users/skogl/Downloads/eirikgsk/BioSignal_AI/dataset/Participants/Participants.xlsx"
)


def _check_inspections() -> None:
    """Abort if no inspection findings.md are present."""
    inspections_dir = REPO_ROOT / "inspections"
    if not inspections_dir.exists():
        print(
            "ERROR: inspections/ directory not found. "
            "Run /inspect first before labeling.",
            file=sys.stderr,
        )
        sys.exit(1)

    findings = list(inspections_dir.rglob("findings.md"))
    if not findings:
        print(
            "ERROR: No inspections/*/findings.md found. "
            "Run /inspect on at least one recording before labeling.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Inspections found: {[str(f) for f in findings]}")


def _discover_recordings(dataset_dir: Path) -> list[Path]:
    """Return sorted list of recording directories."""
    return sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("recording_")]
    )


def _write_summary(
    out_base: Path,
    results: list[dict],
    participants_data: dict,
) -> None:
    """Write data/labeled/_summary.md."""
    lines = []
    lines.append("# Labeling Summary")
    lines.append("")

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] == "error"]
    flagged = [r for r in results if r["flags"] and r["status"] == "ok"]

    lines.append(f"- Recordings attempted: **{len(results)}**")
    lines.append(f"- Successfully labeled: **{len(ok)}**")
    lines.append(f"- Errors (halted): **{len(err)}**")
    lines.append(f"- Flagged for review: **{len(flagged)}**")
    lines.append("")

    # Active time per subject
    from collections import defaultdict
    subject_time: dict[str, float] = defaultdict(float)
    for r in ok:
        subject_time[r["subject_id"]] += r.get("active_minutes", 0.0)

    lines.append("## Active time per subject")
    lines.append("| Subject | Active minutes |")
    lines.append("|---|---|")
    for subj, mins in sorted(subject_time.items()):
        lines.append(f"| {subj} | {mins:.1f} |")
    lines.append("")
    total_active_h = sum(subject_time.values()) / 60.0
    lines.append(f"**Total active time: {total_active_h:.2f} hours**")
    lines.append("")

    # Recordings per subject
    subject_recordings: dict[str, list[str]] = defaultdict(list)
    for r in results:
        subject_recordings[r["subject_id"]].append(r["recording_id"])
    lines.append("## Recordings per subject")
    for subj, recs in sorted(subject_recordings.items()):
        lines.append(f"- **{subj}**: {', '.join(recs)}")
    lines.append("")

    # EDA status table
    lines.append("## EDA quality per recording")
    lines.append("| Recording | Subject | EDA status |")
    lines.append("|---|---|---|")
    for r in results:
        eda_st = r.get("eda_status", "unknown")
        lines.append(f"| {r['recording_id']} | {r['subject_id']} | {eda_st} |")
    lines.append("")

    unusable_eda = [r["recording_id"] for r in ok if r.get("eda_status") == "unusable"]
    if unusable_eda:
        lines.append("**EDA unusable (NaN in parquet):**")
        for rec in unusable_eda:
            lines.append(f"- {rec}")
        lines.append("")

    # Flagged recordings
    if flagged:
        lines.append("## Recordings flagged for manual review")
        for r in flagged:
            lines.append(f"### {r['recording_id']} ({r['subject_id']})")
            for f in r["flags"]:
                lines.append(f"- {f}")
        lines.append("")

    # Error recordings
    if err:
        lines.append("## Recordings with processing errors (halted)")
        for r in err:
            lines.append(f"### {r['recording_id']} ({r['subject_id']})")
            lines.append(f"- Error: {r['error']}")
            for f in r["flags"]:
                lines.append(f"- {f}")
        lines.append("")

    # Missing temperature
    temp_missing = [
        r["recording_id"] for r in ok
        if any("temperature.csv is empty" in f for f in r["flags"])
    ]
    if temp_missing:
        lines.append("## Recordings with missing temperature")
        for rec in temp_missing:
            lines.append(f"- {rec}")
        lines.append("")

    # Go/no-go recommendation
    lines.append("## Go/no-go for /train")
    n_ok = len(ok)
    n_err = len(err)
    if n_err == 0 and n_ok >= 9:
        lines.append(
            f"**GO** — all {n_ok} recordings labeled successfully. "
            f"Note: EDA features are NaN for {len(unusable_eda)} recordings "
            f"(sensor floor); downstream feature extractor must handle NaN EDA. "
            f"All other modalities (ECG, EMG, PPG-green, IMU, temperature) are present."
        )
    else:
        lines.append(
            f"**REVIEW NEEDED** — {n_err} recordings errored, {n_ok} OK. "
            f"Resolve errors before running /train."
        )
    lines.append("")

    summary_path = out_base / "_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline labeling pipeline for strength-RT recordings."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Label all recordings under dataset_aligned/recording_*"
    )
    group.add_argument(
        "--recording", action="append", metavar="NNN",
        help="Label one or more specific recordings (e.g. --recording 012)"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DATASET_ALIGNED_DIR,
        help="Source dataset directory (default: dataset_aligned/). "
             "Use dataset_clean/ to label the cleaned dataset."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / "labeled",
        help="Output directory for parquets (default: data/labeled/)."
    )

    args = parser.parse_args(argv)

    # --- precondition check ---
    _check_inspections()

    out_base = args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir

    if not PARTICIPANTS_XLSX.exists():
        print(
            f"ERROR: Participants.xlsx not found at {PARTICIPANTS_XLSX}",
            file=sys.stderr,
        )
        return 1

    if not source_dir.exists():
        print(
            f"ERROR: source dir not found at {source_dir}",
            file=sys.stderr,
        )
        return 1
    print(f"Source: {source_dir}\nOutput: {out_base}\n")

    # Lazy import (heavy) only after precondition check passes
    from src.data.participants import load_participants
    from src.labeling.run import label_one_recording

    print(f"Loading Participants.xlsx from {PARTICIPANTS_XLSX} ...")
    participants_data = load_participants(PARTICIPANTS_XLSX)
    print(f"  Loaded {len(participants_data)} recording entries.")

    # Determine which recordings to process
    if args.all:
        rec_dirs = _discover_recordings(source_dir)
        print(f"Found {len(rec_dirs)} recording directories in {source_dir}.")
    else:
        rec_dirs = []
        for num_str in args.recording:
            num_str = num_str.zfill(3)
            d = source_dir / f"recording_{num_str}"
            if not d.exists():
                print(f"WARNING: {d} does not exist — skipping.", file=sys.stderr)
            else:
                rec_dirs.append(d)

    if not rec_dirs:
        print("No recordings to process.", file=sys.stderr)
        return 1

    results = []
    for rec_dir in rec_dirs:
        print(f"\n=== Processing {rec_dir.name} ===")
        res = label_one_recording(
            rec_dir=rec_dir,
            participants_data=participants_data,
            out_base=out_base,
            expected_n_sets=12,
        )
        results.append(res)
        status = res["status"]
        n_flags = len(res["flags"])
        print(
            f"  [{rec_dir.name}] status={status}, "
            f"subject={res['subject_id']}, "
            f"sets={res['n_sets']}, "
            f"flags={n_flags}, "
            f"active_min={res.get('active_minutes', 0.0):.1f}, "
            f"eda={res.get('eda_status', 'unknown')}"
        )
        if res.get("error"):
            print(f"  [{rec_dir.name}] ERROR: {res['error']}")

    _write_summary(out_base, results, participants_data)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone. {n_ok} OK, {n_err} errors.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
