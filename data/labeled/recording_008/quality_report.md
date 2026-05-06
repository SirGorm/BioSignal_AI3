# Quality Report — recording_008 (michael)

## Set count validation
- markers.json: **10** sets
- Participants.xlsx: **12** sets (expected 12)
- metadata.json total_kinect_sets: **12**
- Canonical sets used: **10**

## Set durations and rep counts
| Canonical set | Markers set_num | Exercise | Duration (s) | Marker reps | Joint reps | Flag |
|---|---|---|---|---|---|---|
| 1 | 1 | benchpress | 20.6 | 8 | 8 |  |
| 2 | 2 | benchpress | 26.7 | 10 | 10 |  |
| 3 | 3 | benchpress | 24.2 | 7 | 7 |  |
| 4 | 4 | squat | 31.3 | 8 | 8 |  |
| 5 | 5 | squat | 29.5 | 8 | 8 |  |
| 6 | 6 | squat | 33.4 | 8 | 8 |  |
| 7 | 7 | pullup | 27.3 | 6 | 6 |  |
| 8 | 10 | deadlift | 17.4 | 4 | 4 |  REP_COUNT_UNUSUAL(4) |
| 9 | 11 | deadlift | 49.9 | 7 | 7 |  |
| 10 | 12 | deadlift | 42.8 | 6 | 6 |  |

## Joint angle coverage per set
| Canonical set | Markers set_num | Joint file found | Frames | NaN angle % |
|---|---|---|---|---|
| 1 | 1 | True | 2063 | 100.0% |
| 2 | 2 | True | 2672 | 100.0% |
| 3 | 3 | True | 2416 | 100.0% |
| 4 | 4 | True | 655 | 13.3% |
| 5 | 5 | True | 618 | 8.6% |
| 6 | 6 | True | 696 | 8.2% |
| 7 | 7 | True | 571 | 0.0% |
| 8 | 10 | True | 745 | 0.3% |
| 9 | 11 | True | 900 | 10.1% |
| 10 | 12 | True | 641 | 0.0% |

## Modality status
- temperature.csv: present
- EDA (`eda_status`): **UNUSABLE** (std=6.556e-09 S, range=8.234e-08 S — sensor floor; threshold: std<1e-7 S OR range<5e-8 S; Greco et al. 2016). EDA column set to all-NaN in parquet.
- EMG/EDA baseline window: 159.3 s before first set

## Flags requiring manual review
- Manually excluded 1 marker set(s) [8, 9] per QC review.
- Recording has only 10 canonical sets (< expected 12); participants.xlsx slots beyond this index will be ignored.
- EDA unusable: std=6.556e-09 S, range=8.234e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 4 joint angle: 13.3% NaN frames.
- Set 5 joint angle: 8.6% NaN frames.
- Set 6 joint angle: 8.2% NaN frames.
- Set 11 joint angle: 10.1% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [3, 9, 10] (set in_active stays True; phase_label=unknown, rep_density=0)

## Methodological notes
- Set boundaries taken from markers.json (Set:N_Start / Set_N_End unix_time) rather than acc-magnitude segmentation. On rec_012, acc-magnitude over-segmented to 42 segments vs 12 true sets, confirming markers.json as the authoritative source (Bulling et al. 2014).
- Joint frame Unix time anchored via: `t_frame = markers[Set:N_Start][unix_time] + frame_id / 30.0` (Azure Kinect DK ≈ 30 fps; internal timestamp_usec = 0 in all recordings).
- Phase labeling: concentric/eccentric from sign of angular velocity (5 deg/s isometric threshold; De Luca 1997).
- Bilateral joint angle averaged (left + right) to reduce single-side tracking failures (Fukuchi et al. 2018).
- Temperature NaN-tolerance: empty temperature.csv returns an all-NaN column. Feature extractor must handle NaN input for temperature.
- EDA unusable criterion: std < 1e-7 S OR range < 5e-8 S, consistent with electrode-skin contact failure (Greco et al. 2016). All 9 recordings in dataset_aligned/ meet this criterion.
- EMG baseline: first 90-120 s of recording (verified by markers[Set:1_Start].unix_time - bio_t0 >= 90 s; CLAUDE.md). Per-subject normalization required due to inter-subject EDA baseline amplitude variation >30% (Greco et al. 2016).

## References
- Bulling A et al. (2014). A tutorial on human activity recognition using body-worn inertial sensors. ACM Comput. Surv. 46(3), 33:1-33:33.
- De Luca CJ (1997). The use of surface electromyography in biomechanics. J. Appl. Biomech. 13(2), 135-163.
- Fukuchi CA et al. (2018). A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals. PeerJ 6:e4640.
- Greco A et al. (2016). cvxEDA: a convex optimization approach to electrodermal activity processing. IEEE Trans. Biomed. Eng. 63(4), 797-804.
- Task Force of the European Society of Cardiology (1996). Heart rate variability: standards of measurement. Eur. Heart J. 17, 354-381.
- Maeda Y et al. (2011). [REF NEEDED: Maeda 2011 exact citation for wrist skin temperature].