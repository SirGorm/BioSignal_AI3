---
description: Inspect a recording before doing anything else - load all signals, plot, verify Unix-time sync, generate findings report
allowed-tools: Bash, Read, Write, Edit
argument-hint: [subject_id] [session_id]
---

# Inspect Recording

**Run this FIRST on a new dataset.** Loads one recording, plots all signals, verifies timestamp synchronization, and produces a findings report you can use to update `CLAUDE.md` with actual sample rates and channel names.

## Steps

1. **Pick the recording.** Use the arguments if provided. If not, list available recordings under `data/raw/` and ask which to inspect. Recommend a mid-protocol session (not the first or last).

2. **Load the data-inspection skill** for the methodology.

3. **Run the full inspection procedure**:
   - List files in the session directory
   - Load biosignals at native sample rates (don't resample)
   - Load joint-angle data
   - Load `participants.xlsx` if available
   - Verify Unix timestamps (`> 1e9`) on both biosignal and joint files
   - Compute actual per-channel sample rates from timestamp deltas
   - Compute per-channel statistics (range, NaN %, clipping)
   - Generate plots:
     - signal_overview.png (all channels, full session)
     - signal_zoomed_set1.png (first detected active set)
     - ppg_channel_check.png (4 PPG wavelengths overlaid)
     - psd_<channel>.png per channel
     - timestamp_alignment.png (Unix time coverage)
     - sets_detected.png (acc-magnitude with set boundaries)
     - joint_coverage.png (joint availability vs detected sets)
   - Save stats.json
   - Cross-validate detected sets vs participants.xlsx

4. **Write findings.md** with:
   - Confirmed sample rates per modality
   - Confirmed channel names (especially PPG-green identification)
   - Sync verification (Unix offset, overlap duration)
   - Set count match/mismatch with spreadsheet
   - Quality flags (clipping, line noise, missing data)
   - Action items for `CLAUDE.md` and `configs/default.yaml`

5. **Save everything to `inspections/<subject_id>_<session_id>/`.**

6. **Print a summary** to chat (5-10 lines max) and ask the user to:
   - Open `findings.md`
   - Apply the recommended changes to `CLAUDE.md` and `configs/default.yaml`

## What NOT to do in this step

- **Don't preprocess.** No filtering, no resampling, no feature extraction. Inspection is read-only.
- **Don't write to `data/labeled/`.** This is exploration only.
- **Don't skip plotting** even if everything looks fine — the plots ARE the deliverable.
- **Don't proceed to labeling** until findings.md exists. If the user asks to label without inspecting first, ask whether they want to inspect first.

## Output

```
Inspection complete: <subject_id>/<session_id>
- Duration: 28.4 min
- Sample rates confirmed: ECG=500, EMG=1000, EDA=32, temp=4, acc=100, PPG=64
- Unix-time sync: offset 0.02 s, 28.4 min overlap — OK
- Sets detected: 9 (matches participants.xlsx ✓)
- Quality flags: 50 Hz line noise on ECG/EMG (notch filter mandatory)
- 1 action item for CLAUDE.md (PPG-green column is ppg_2, not ppg_green)

Open inspections/<subject>_<session>/findings.md for full report.
Apply recommended changes to CLAUDE.md before running /label.
```
