# Labeling Summary

- Recordings attempted: **10**
- Successfully labeled: **10**
- Errors (halted): **0**
- Flagged for review: **10**

## Active time per subject
| Subject | Active minutes |
|---|---|
| Hytten | 4.2 |
| Juile | 6.0 |
| Raghild | 6.3 |
| Tias | 5.1 |
| Vivian | 10.4 |
| kiyomi | 6.2 |
| lucas | 5.0 |
| lucas 2 | 6.6 |
| michael | 5.1 |
| sivert | 5.4 |

**Total active time: 1.00 hours**

## Recordings per subject
- **Hytten**: recording_006
- **Juile**: recording_013
- **Raghild**: recording_014
- **Tias**: recording_012
- **Vivian**: recording_003
- **kiyomi**: recording_010
- **lucas**: recording_009
- **lucas 2**: recording_011
- **michael**: recording_008
- **sivert**: recording_007

## EDA quality per recording
| Recording | Subject | EDA status |
|---|---|---|
| recording_003 | Vivian | unusable |
| recording_006 | Hytten | unusable |
| recording_007 | sivert | unusable |
| recording_008 | michael | unusable |
| recording_009 | lucas | unusable |
| recording_010 | kiyomi | unusable |
| recording_011 | lucas 2 | unusable |
| recording_012 | Tias | unusable |
| recording_013 | Juile | unusable |
| recording_014 | Raghild | unusable |

**EDA unusable (NaN in parquet):**
- recording_003
- recording_006
- recording_007
- recording_008
- recording_009
- recording_010
- recording_011
- recording_012
- recording_013
- recording_014

## Recordings flagged for manual review
### recording_003 (Vivian)
- EDA unusable: std=5.336e-09 S, range=5.175e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 6 joint angle: 2.6% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [11] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_006 (Hytten)
- markers.json had 12 sets after manual exclusion; select_canonical_sets dropped 1 aborted/short attempt(s) → 11 canonical sets.
- Recording has only 11 canonical sets (< expected 12); participants.xlsx slots beyond this index will be ignored.
- EDA unusable: std=4.282e-09 S, range=3.282e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- PHASE+REPS blacklist applied to canonical sets [9] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_007 (sivert)
- EDA unusable: std=7.215e-09 S, range=7.532e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 7 (canonical 7, deadlift): joint-angle phase 100% unknown — fell back to acc-based phase.
- Set 8 (canonical 8, deadlift): joint-angle phase 100% unknown — fell back to acc-based phase.
- Set 9 (canonical 9, deadlift): joint-angle phase 100% unknown — fell back to acc-based phase.
- PHASE+REPS blacklist applied to canonical sets [7, 8, 9] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_008 (michael)
- Manually excluded 1 marker set(s) [8, 9] per QC review.
- Recording has only 10 canonical sets (< expected 12); participants.xlsx slots beyond this index will be ignored.
- EDA unusable: std=6.556e-09 S, range=8.234e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 4 joint angle: 13.3% NaN frames.
- Set 5 joint angle: 8.6% NaN frames.
- Set 6 joint angle: 8.2% NaN frames.
- Set 11 joint angle: 10.1% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [3, 9, 10] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_009 (lucas)
- Manually excluded 3 marker set(s) [10, 11, 12, 13, 14] per QC review.
- Recording has only 9 canonical sets (< expected 12); participants.xlsx slots beyond this index will be ignored.
- EDA unusable: std=5.824e-09 S, range=9.934e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 7 joint angle: 3.4% NaN frames.
- Set 8 joint angle: 5.3% NaN frames.
- Set 9 joint angle: 6.3% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [5, 6] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_010 (kiyomi)
- EDA unusable: std=4.043e-09 S, range=4.191e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
### recording_011 (lucas 2)
- EDA unusable: std=5.286e-09 S, range=1.019e-07 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 10 joint angle: 2.0% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [3, 8, 9, 10, 11] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_012 (Tias)
- EDA unusable: std=6.129e-09 S, range=7.804e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 9 joint angle: 1.8% NaN frames.
### recording_013 (Juile)
- EDA unusable: std=3.411e-09 S, range=2.460e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- PHASE+REPS blacklist applied to canonical sets [7] (set in_active stays True; phase_label=unknown, rep_density=0)
### recording_014 (Raghild)
- EDA unusable: std=5.448e-09 S, range=5.035e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 7 (canonical 7, deadlift): joint-angle phase 100% unknown — fell back to acc-based phase.
- PHASE+REPS blacklist applied to canonical sets [7, 11, 12] (set in_active stays True; phase_label=unknown, rep_density=0)

## Go/no-go for /train
**GO** — all 10 recordings labeled successfully. Note: EDA features are NaN for 10 recordings (sensor floor); downstream feature extractor must handle NaN EDA. All other modalities (ECG, EMG, PPG-green, IMU, temperature) are present.
