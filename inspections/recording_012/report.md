# Inspection report — recording_012

Participant: **Tias**
Recording UTC start: 2026-03-12 18:47:29 UTC
Duration: 36.1 min
Total sets (metadata.json): 12
Total sets (markers.json):  12
Total sets (Participants.xlsx): 12

## Modalities (measured fs from CSV timestamps)

| Modality | fs (measured) | mean | p1..p99 | NaN% |
|----------|--------------:|-----:|---------|-----:|
| ecg | 500.0 Hz | -5.29e-08 | -0.00401 .. 0.00407 | 0.00% |
| emg | 2000.1 Hz | -2.97e-09 | -0.000364 .. 0.000347 | 0.00% |
| eda | 50.0 Hz | 2.74e-07 | 2.51e-07 .. 2.94e-07 | 0.00% |
| temperature | empty/missing | — | — | — |
| ppg_green | 100.0 Hz | 5.92e+03 | 4.91e+03 .. 6.41e+03 | 0.00% |
| ax | 100.0 Hz | 3.99 | -11.3 .. 12.9 | 0.00% |
| ay | 100.0 Hz | -4.05 | -10.1 .. 4.14 | 0.00% |
| az | 100.0 Hz | 2.82 | -5.5 .. 10.4 | 0.00% |

Declared fs from metadata.json: `{'emg': 2000, 'ecg': 500, 'temperature': 1, 'eda': 50, 'ppg': 100, 'imu': 100}`

## Sets (markers.json + Participants.xlsx)

| Set | Exercise | RPE | Duration | Reps |
|----:|----------|----:|---------:|-----:|
| 1 | pullup | 4 | 21.8 s | 8 |
| 2 | pullup | 6 | 25.6 s | 10 |
| 3 | pullup | 8 | 26.6 s | 10 |
| 4 | benchpress | 6 | 18.9 s | 8 |
| 5 | benchpress | 6 | 23.3 s | 10 |
| 6 | benchpress | 6 | 20.5 s | 10 |
| 7 | deadlift | 6 | 30.6 s | 8 |
| 8 | deadlift | 6 | 26.6 s | 8 |
| 9 | deadlift | 8 | 26.5 s | 8 |
| 10 | squat | 6 | 25.3 s | 8 |
| 11 | squat | 7 | 26.9 s | 8 |
| 12 | squat | 8 | 31.6 s | 9 |

## Joint-skeleton file coverage (1 file per set)

| Set | Joint file | n_frames |
|----:|------------|---------:|
| 1 | recording_01_joints.json | 460 |
| 2 | recording_02_joints.json | 541 |
| 3 | recording_03_joints.json | 559 |
| 4 | recording_04_joints.json | 398 |
| 5 | recording_05_joints.json | 487 |
| 6 | recording_06_joints.json | 431 |
| 7 | recording_07_joints.json | 643 |
| 8 | recording_08_joints.json | 559 |
| 9 | recording_09_joints.json | 555 |
| 10 | recording_10_joints.json | 526 |
| 11 | recording_11_joints.json | 566 |
| 12 | recording_12_joints.json | 662 |

## Plots

- `signal_overview.png` — all channels with set windows shaded
- `signal_zoomed_set1.png` — set 1 zoom, red dashed = rep markers
- `ppg_channel_check.png` — 30 s rest window, all 4 PPG wavelengths
- `psd_<channel>.png` — per-channel PSD with 50/60 Hz lines marked
- `timestamp_alignment.png` — biosignal Unix-time coverage
- `sets_detected.png` — markers vs acc-magnitude segmentation
- `joint_coverage.png` — per-set Kinect skeleton frame counts
