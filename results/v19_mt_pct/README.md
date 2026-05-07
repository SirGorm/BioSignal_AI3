# v19_mt_pct — Multi-task NN training results

Multitask trening (4 hoder samtidig: exercise, phase, fatigue, reps) på rå biosignal-vinduer fra 4 modaliteter (EMG, Acc, PPG-green, Temp). Subject-wise 5-fold CV, 3 seeds (42, 1337, 7), Optuna 50 trials per (arkitektur, vindusstørrelse).

## Beste modell per hode

Tabellen er aggregert over 3 seeds × 5 folds = 15 målinger fra phase 2 (final-fit med best HPs). Lavere er bedre for `val_total`, `mae`. Høyere er bedre for `f1_macro`, `pearson_r`.

| Hode / metrikk | Beste run | Verdi (mean ± std) |
|---|---|---|
| **Overall** (val_total, lavere bedre) | `v19_mt-pct-w1s-lstm_raw` | **-0.1453 ± 0.42** |
| **Exercise** (f1_macro) | `v19_mt-pct-w5s-cnn1d_raw` | **0.4944** |
| **Phase** (f1_macro) | `v19_mt-pct-w5s-cnn_lstm_raw` | **0.3834** |
| **Fatigue** (MAE, RPE 1–10) | `v19_mt-pct-w2s-lstm_raw` | **1.0226** |
| **Reps** (MAE) | `v19_mt-pct-w1s-lstm_raw` | **0.1066** |

> `w1s-lstm_raw` vinner både overall og reps. LSTM-baserte modeller dominerer fatigue+reps; CNN-arkitekturer dominerer exercise+phase.

## Full sammenligning (alle 12 runs)

| Run | val_total | exercise f1 | phase f1 | fatigue MAE | fatigue r | reps MAE |
|---|---|---|---|---|---|---|
| w1s-cnn1d_raw      | 1.4951  | 0.3538 | 0.3073 | 1.2564 | 0.1119 | 0.1215 |
| **w1s-lstm_raw**   | **-0.1453** | 0.3440 | 0.3092 | 1.0263 | 0.2777 | **0.1066** |
| w1s-cnn_lstm_raw   | 0.3525  | 0.2817 | 0.2841 | 1.0944 | 0.2064 | 0.1068 |
| w1s-tcn_raw        | 0.4418  | 0.3935 | 0.3279 | 1.3017 | 0.1922 | 0.1090 |
| w2s-cnn1d_raw      | 2.2911  | 0.3862 | 0.2770 | 1.2449 | 0.1529 | 0.2318 |
| **w2s-lstm_raw**   | 0.4647  | 0.3484 | 0.2896 | **1.0226** | 0.2384 | 0.2013 |
| w2s-cnn_lstm_raw   | 1.5155  | 0.3040 | 0.2682 | 1.1340 | 0.2174 | 0.1989 |
| w2s-tcn_raw        | 0.2119  | 0.3340 | 0.2745 | 1.0522 | 0.1519 | 0.2061 |
| **w5s-cnn1d_raw**  | 1.6116  | **0.4944** | 0.3325 | 1.2667 | 0.0912 | 0.4392 |
| w5s-lstm_raw       | 0.7770  | 0.3018 | 0.2334 | 1.0313 | 0.1841 | 0.4740 |
| **w5s-cnn_lstm_raw** | 2.1985 | 0.4756 | **0.3834** | 1.1004 | 0.0670 | 0.3933 |
| w5s-tcn_raw        | 0.7905  | 0.3298 | 0.2569 | 1.0664 | 0.1712 | 0.4841 |

## Innhold i denne mappen

```
results/v19_mt_pct/
├── README.md                    ← denne filen
├── runs/                        ← 4 best-per-task runs (uten phase1 .pt)
│   ├── v19_mt-pct-w1s-lstm_raw/      (overall + reps best)
│   ├── v19_mt-pct-w2s-lstm_raw/      (fatigue best)
│   ├── v19_mt-pct-w5s-cnn1d_raw/     (exercise best)
│   └── v19_mt-pct-w5s-cnn_lstm_raw/  (phase best)
└── logs/                        ← stdout-logger fra alle 12 runs
    └── {arch}-w{N}s.log
```

Per run-mappe:
- `config.json` — Optuna-oppsett (n_trials, epochs, seeds, ...)
- `best_hps.json` — beste hyperparametere fra phase 1 + Optuna-score + søketid
- `dataset_meta.json` — antall vinduer, subjects per fold, klasseboost
- `optuna.db` — full Optuna-database (alle 50 trials)
- `phase1_log.json` — per-trial scores
- `phase1/trial_NNN/{arch}/seed_42/fold_K/{history,metrics}.json` — kun JSON, ikke `.pt`
- `phase2/{arch}/cv_summary.json` — endelig CV-aggregat
- `phase2/{arch}/seed_*/fold_K/{checkpoint_best.pt, history.json, metrics.json, test_preds.pt}` — endelige modeller

`.pt`-filer fra phase 1 (Optuna trial-checkpoints) er ekskludert for å holde størrelse nede; kan re-genereres fra `best_hps.json` + samme seed.

## Oppsett (felles)

- 50 Optuna trials per (arkitektur × vindusstørrelse), `phase1_epochs=50`, `phase2_epochs=300`
- 3 seeds: 42, 1337, 7
- Subject-wise 5-fold CV (LOSO over deltaker-navn fra `Participants.xlsx`)
- Multi-task uncertainty-weighted loss (Kendall et al. 2018)
- Inputkanaler: EMG-envelope (100 Hz), Acc-magnitude, PPG-green, Temperature

## Hvor finner jeg "rådata"?

Phase 2 ligger nestet for fleksibilitet: `phase2/{arch}/cv_summary.json` har aggregerte metrikker; `phase2/{arch}/seed_*/fold_*/metrics.json` har per-fold detaljer; `test_preds.pt` har modellprediksjoner per fold for plotting/etterprosessering.
