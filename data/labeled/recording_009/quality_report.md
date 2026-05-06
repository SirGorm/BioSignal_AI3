# Quality Report — recording_009 (lucas)

## Set count validation
- markers.json: **9** sets
- Participants.xlsx: **12** sets (expected 12)
- metadata.json total_kinect_sets: **12**
- Canonical sets used: **9**

## Set durations and rep counts
| Canonical set | Markers set_num | Exercise | Duration (s) | Marker reps | Joint reps | Flag |
|---|---|---|---|---|---|---|
| 1 | 1 | deadlift | 50.5 | 8 | 8 |  |
| 2 | 2 | deadlift | 38.5 | 8 | 8 |  |
| 3 | 3 | deadlift | 39.3 | 8 | 8 |  |
| 4 | 4 | benchpress | 23.2 | 8 | 8 |  |
| 5 | 5 | benchpress | 23.5 | 8 | 8 |  |
| 6 | 6 | benchpress | 33.1 | 8 | 8 |  |
| 7 | 7 | squat | 29.5 | 8 | 8 |  |
| 8 | 8 | squat | 28.5 | 8 | 8 |  |
| 9 | 9 | squat | 33.3 | 8 | 8 |  |

## Joint angle coverage per set
| Canonical set | Markers set_num | Joint file found | Frames | NaN angle % |
|---|---|---|---|---|
| 1 | 1 | True | 1058 | 0.0% |
| 2 | 2 | True | 813 | 0.0% |
| 3 | 3 | True | 825 | 0.1% |
| 4 | 4 | True | 2320 | 100.0% |
| 5 | 5 | True | 2348 | 100.0% |
| 6 | 6 | True | 3305 | 100.0% |
| 7 | 7 | True | 618 | 3.4% |
| 8 | 8 | True | 599 | 5.3% |
| 9 | 9 | True | 702 | 6.3% |

## Modality status
- temperature.csv: present
- EDA (`eda_status`): **UNUSABLE** (std=5.824e-09 S, range=9.934e-08 S — sensor floor; threshold: std<1e-7 S OR range<5e-8 S; Greco et al. 2016). EDA column set to all-NaN in parquet.
- EMG/EDA baseline window: 151.9 s before first set

## Flags requiring manual review
- Manually excluded 3 marker set(s) [10, 11, 12, 13, 14] per QC review.
- Recording has only 9 canonical sets (< expected 12); participants.xlsx slots beyond this index will be ignored.
- EDA unusable: std=5.824e-09 S, range=9.934e-08 S (sensor floor — electrode-skin contact failure; Greco et al. 2016). EDA column set to all-NaN in parquet.
- Set 7 joint angle: 3.4% NaN frames.
- Set 8 joint angle: 5.3% NaN frames.
- Set 9 joint angle: 6.3% NaN frames.
- PHASE+REPS blacklist applied to canonical sets [5, 6] (set in_active stays True; phase_label=unknown, rep_density=0)

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