---
description: Run the offline labeling pipeline on all sessions (or a specific subject/session)
allowed-tools: Bash, Read, Write, Edit
argument-hint: [--subject SUBJECT_ID] [--session SESSION_ID] [--all]
---

# Label Sessions

Runs the offline labeling pipeline. **Requires `/inspect` to have been run on at least one session first** so sample rates and channel names are confirmed in CLAUDE.md.

## Preconditions

- `inspections/` directory exists with at least one `<subject>_<session>/findings.md`
- The user has applied recommendations from findings.md to `CLAUDE.md` and `configs/default.yaml`
- Recommended: open the most recent findings.md and confirm that no critical action items are unaddressed

## Steps

1. **Validate inputs** for each requested session: biosignals file exists, joint_angles.csv exists, participants.xlsx exists. Use the `data-loader` skill's `validate_session_files` function (which itself refuses to proceed if `inspections/` is empty).

2. **Invoke the data-labeler subagent** for each session. The subagent will:
   - Load biosignals, joint angles, participants spreadsheet
   - Run session segmentation (active vs rest) using `session-segmentation` skill
   - Cross-validate detected sets vs spreadsheet — halt and report on mismatch
   - Align joint-angle data to biosignal timestamps
   - Derive phase + rep labels from joint angles using `joint-angle-labeling` skill
   - Assign per-set RPE and exercise from spreadsheet
   - Write `aligned_features.parquet` and `quality_report.md` per session

3. **After all sessions**: aggregate quality reports into `data/labeled/_summary.md` showing:
   - Sessions processed
   - Total active time per subject
   - Distribution of detected reps per set
   - Sessions flagged for manual review

4. **Run validation tests**:
   - `pytest tests/test_label_alignment.py -v`
   - Halt and report if any test fails. Do NOT proceed to feature extraction with corrupt labels.

## Output

A short summary in chat:
```
Labeled N sessions across M subjects (X hours active).
Q sessions flagged for review (see data/labeled/_summary.md).
All alignment tests PASSED.
Next: run /train to extract features and train models.
```
