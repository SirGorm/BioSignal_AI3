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
        help="Label all recordings under dataset/recording_*"
    )
    group.add_argument(
        "--recording", action="append", metavar="NNN",
        help="Label one or more specific recordings (e.g. --recording 012)"
    )

    args = parser.parse_args(argv)

    # --- precondition check ---
    _check_inspections()

    dataset_dir = REPO_ROOT / "dataset"
    out_base = REPO_ROOT / "data" / "labeled"
    out_base.mkdir(parents=True, exist_ok=True)

    participants_path = dataset_dir / "Participants" / "Participants.xlsx"

    # Lazy import (heavy) only after precondition check passes
    from src.data.participants import load_participants
    from src.labeling.run import label_one_recording

    print(f"Loading Participants.xlsx from {participants_path} ...")
    participants_data = load_participants(participants_path)
    print(f"  Loaded {len(participants_data)} recording entries.")

    # Determine which recordings to process
    if args.all:
        rec_dirs = _discover_recordings(dataset_dir)
        print(f"Found {len(rec_dirs)} recording directories.")
    else:
        rec_dirs = []
        for num_str in args.recording:
            num_str = num_str.zfill(3)
            d = dataset_dir / f"recording_{num_str}"
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
            f"active_min={res.get('active_minutes', 0.0):.1f}"
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
