# Inspection report — recording_013

Participant: **Juile**
Recording UTC start: 2026-03-12 19:36:33 UTC
Duration: 37.7 min
Total sets (metadata.json): 12
Total sets (markers.json):  12
Total sets (Participants.xlsx): 12

## Modalities (measured fs from CSV timestamps)

| Modality | fs (measured) | mean | p1..p99 | NaN% |
|----------|--------------:|-----:|---------|-----:|
| ecg | 500.0 Hz | 2.61e-08 | -0.00313 .. 0.00318 | 0.00% |
| emg | 2000.1 Hz | 4.24e-09 | -0.000923 .. 0.000902 | 0.00% |
| eda | 50.0 Hz | 2.45e-07 | 2.36e-07 .. 2.55e-07 | 0.00% |
| temperature | empty/missing | — | — | — |
| ppg_green | 100.0 Hz | 6.09e+03 | 5.41e+03 .. 6.54e+03 | 0.00% |
| ax | 100.0 Hz | 3.42 | -11 .. 11.9 | 0.00% |
| ay | 100.0 Hz | -3.17 | -9.06 .. 6.98 | 0.00% |
| az | 100.0 Hz | 3.39 | -4.89 .. 10.3 | 0.00% |

Declared fs from metadata.json: `{'emg': 2000, 'ecg': 500, 'temperature': 1, 'eda': 50, 'ppg': 100, 'imu': 100}`

## Sets (markers.json + Participants.xlsx)

| Set | Exercise | RPE | Duration | Reps |
|----:|----------|----:|---------:|-----:|
| 1 | deadlift | 4 | 30.0 s | 8 |
| 2 | deadlift | 5 | 36.7 s | 10 |
| 3 | deadlift | 7 | 36.7 s | 10 |
| 4 | squat | 5 | 31.1 s | 8 |
| 5 | squat | 7 | 25.5 s | 8 |
| 6 | squat | 7 | 34.5 s | 10 |
| 7 | benchpress | 8 | 26.0 s | 7 |
| 8 | benchpress | 7 | 26.7 s | 10 |
| 9 | benchpress | 7 | 26.9 s | 10 |
| 10 | pullup | 7 | 26.8 s | 8 |
| 11 | pullup | 8 | 30.2 s | 8 |
| 12 | pullup | 9 | 28.3 s | 8 |

## Joint-skeleton file coverage (1 file per set)

| Set | Joint file | n_frames |
|----:|------------|---------:|
| 1 | recording_01_joints.json | 630 |
| 2 | recording_02_joints.json | 768 |
| 3 | recording_03_joints.json | 771 |
| 4 | recording_04_joints.json | 655 |
| 5 | recording_05_joints.json | 534 |
| 6 | recording_06_joints.json | 723 |
| 7 | recording_07_joints.json | 547 |
| 8 | recording_08_joints.json | 563 |
| 9 | recording_09_joints.json | 565 |
| 10 | recording_10_joints.json | 400 |
| 11 | recording_11_joints.json | 632 |
| 12 | recording_12_joints.json | 592 |

## Plots

- `signal_overview.png` — all channels with set windows shaded
- `signal_zoomed_set1.png` — set 1 zoom, red dashed = rep markers
- `ppg_channel_check.png` — 30 s rest window, all 4 PPG wavelengths
- `psd_<channel>.png` — per-channel PSD with 50/60 Hz lines marked
- `timestamp_alignment.png` — biosignal Unix-time coverage
- `sets_detected.png` — markers vs acc-magnitude segmentation
- `joint_coverage.png` — per-set Kinect skeleton frame counts
