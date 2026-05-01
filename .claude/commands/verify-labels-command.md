---
description: Generate per-session label verification plots (PNG + HTML) so you can manually verify exercises, phases, and reps are correctly labeled. Run AFTER /label.
allowed-tools: Bash, Read, Write
argument-hint: [--subjects S001 S002]
---

# Verify Labels

Generates two files per session for manual visual verification:

1. **PNG (`<subject>_<session>_overview.png`)** — static overview, fast to scan.
   Joint angles with phase color overlay, vertical lines at rep onsets,
   exercise band at the bottom.

2. **HTML (`<subject>_<session>_interactive.html`)** — Plotly zoomable plot.
   Same content as PNG but you can zoom into specific sets and hover for
   exact values.

Plus one **`verification_summary.md`** at the top of the output directory
with auto-generated sanity flags per session.

## Preconditions

- `/label` completed (`data/labeled/<subject>/<session>/aligned_features.parquet`
  exists)
- `joint_*` columns present in aligned_features (joint angles are the
  primary signal here since this verifies labels derived FROM joint angles)
- Plotly installed for interactive HTML (`pip install plotly`); without
  it, HTML is a thin wrapper around the PNG

## Steps

1. **Check labeled data exists**:
   ```bash
   ls data/labeled/*/aligned_features.parquet | head
   ```
   If empty, halt and tell user to run `/label` first.

2. **Run visualizer**:
   ```bash
   python scripts/visualize_labels.py
   ```
   Or for specific subjects:
   ```bash
   python scripts/visualize_labels.py --subjects S001 S002
   ```

3. **Open the summary**: `runs/<ts>_label-verification/verification_summary.md`
   contains a table of all sessions, their warnings, and links to PNG/HTML.

## What to look for in the plots

**Phase boundaries (joint-angle subplot)**:
- Each rep should show a clean eccentric → bottom_pause → concentric → top_pause cycle
- Phase color changes should align with joint-angle direction changes
- If the phase color doesn't change at velocity zero-crossings, the phase
  detection threshold needs adjustment

**Rep onsets (vertical orange lines)**:
- One vertical line per detected rep
- Lines should appear at the start of concentric phases (joint angle
  beginning to extend)
- If lines are missing or doubled, rep detection threshold needs adjustment

**Exercise bands (bottom row)**:
- Colored bands cover the active sets
- Light grey covers rest periods
- Exercise names match the protocol (squat, deadlift, etc., per `configs/exercises.yaml`)
- Transitions between exercises should be at expected times

## Sanity flags auto-generated

The script produces these flags per session:

- ⚠ Active fraction outside 30–70% → check rest detection thresholds
- ⚠ Number of sets outside 6–12 → check exercise transitions
- ⚠ Reps per set outside 7–11 → check rep detection
- ⚠ RPE values outside 1–10 → check `participants.xlsx`
- ⚠ Joint-angle dropout >10% → motion-capture issue
- ⚠ Joint-angle range outside 0–180° → unit conversion issue
- ⚠ Time gaps >10× median → sensor disconnect

If a session has no warnings, it shows ✓ "No automated sanity flags raised".

## Output summary in chat (~10 lines)

```
Label verification complete: runs/<ts>_label-verification/

Inspected: 24 sessions
Sessions with warnings: 3
  - S007/session1: 2 warnings (active fraction low, set 3 has 5 reps)
  - S012/session1: 1 warning (joint-angle dropout 15%)
  - S019/session1: 1 warning (RPE = 11 on set 6)

For each session:
  - <subject>_<session>_overview.png   (fast scan)
  - <subject>_<session>_interactive.html (zoomable)

Open verification_summary.md to triage. Fix issues in /label and rerun.
```

## When to use this command

- **First time labeling**: always — catch threshold issues before training
- **After changing `configs/exercises.yaml`**: verify new thresholds
- **After adding new subjects**: spot-check before including in CV
- **Before paper submission**: include the summary as supplementary material

## Hard rules

- **Never edit `aligned_features.parquet` based on visual inspection alone.**
  If labels are wrong, fix `data-labeler` and rerun `/label`.
- **Never skip this step on first run.** Labeling thresholds calibrated on
  one subject often fail on another due to anthropometric variation.
- **Always check at least 3 random sessions manually**, even if no warnings
  are flagged.
