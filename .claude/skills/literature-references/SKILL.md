---
name: literature-references
description: Use whenever you make a methodological decision in this project. Provides a curated bibliography of canonical references for every technique used (EMG fatigue features, HRV, EDA decomposition, PPG, velocity-based training, RPE, subject-wise CV, signal processing, ML methods). All agents MUST cite from this list when writing deliverables (model_card.md, findings.md, quality_report.md, code comments). DO NOT invent citations not on this list — if a needed reference is missing, flag it for the user to add.
---

# Literature References (curated, do not fabricate)

Every methodological choice in this project has a literature anchor. When agents write `model_card.md`, `findings.md`, `quality_report.md`, or non-trivial code comments explaining a choice, they cite from this list using the format `(Author Year)` inline and a full entry in a `## References` section.

**If you need a reference that is NOT in this list, do not invent one.** Add a TODO note in the deliverable like `[REF NEEDED: <description>]` and ask the user to provide the citation. Inventing references silently undermines the project.

## How to cite

Inline: `MNF declines during sustained contractions due to slowed motor unit conduction velocity (De Luca 1997).`

In `## References` at the end of the doc: full entry from this list, copied exactly.

## EMG and muscle fatigue

- **De Luca 1997** — De Luca, C. J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.
  *Foundational reference for surface EMG analysis. Use when discussing MNF/MDF as fatigue indicators or general EMG signal processing.*

- **Cifrek et al. 2009** — Cifrek, M., Medved, V., Tonković, S., & Ostojić, S. (2009). Surface EMG based muscle fatigue evaluation in biomechanics. *Clinical Biomechanics*, 24(4), 327–340.
  *Comprehensive review of EMG-based fatigue features and their use in dynamic contractions.*

- **Dimitrov et al. 2006** — Dimitrov, G. V., Arabadzhiev, T. I., Mileva, K. N., Bowtell, J. L., Crichton, N., & Dimitrova, N. A. (2006). Muscle fatigue during dynamic contractions assessed by new spectral indices. *Medicine and Science in Sports and Exercise*, 38(11), 1971–1979.
  *Source of the FInsm5 spectral fatigue index. Cite when computing the M(-1)/M(5) spectral moment ratio.*

- **Merletti & Parker 2004** — Merletti, R., & Parker, P. A. (Eds.). (2004). *Electromyography: Physiology, Engineering, and Non-Invasive Applications*. IEEE Press / Wiley.
  *Reference textbook for EMG signal processing, electrode placement, and crosstalk discussions.*

- **Phinyomark et al. 2012** — Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012). Feature reduction and selection for EMG signal classification. *Expert Systems with Applications*, 39(8), 7420–7431.
  *Cite when selecting or comparing time- and frequency-domain EMG features.*

## Heart rate variability (HRV)

- **Task Force 1996** — Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043–1065.
  *The canonical standards document. Cite for any HRV definitions (RMSSD, SDNN, pNN50, LF/HF).*

- **Shaffer & Ginsberg 2017** — Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258.
  *Modern HRV review, includes guidance on short-term recording windows and exercise contexts.*

- **Pan & Tompkins 1985** — Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236.
  *Cite when describing R-peak detection in real-time pipelines.*

## Electrodermal activity (EDA)

- **Boucsein 2012** — Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
  *Standard textbook reference for EDA measurement and interpretation.*

- **Greco et al. 2016** — Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE Transactions on Biomedical Engineering*, 63(4), 797–804.
  *Source of the cvxEDA tonic/phasic decomposition implemented in NeuroKit2.*

- **Posada-Quintero & Chon 2020** — Posada-Quintero, H. F., & Chon, K. H. (2020). Innovations in electrodermal activity data collection and signal processing: A systematic review. *Sensors*, 20(2), 479.
  *Use for EDA in exercise and motion-rich contexts (relevant for wrist-worn EDA during training).*

## Photoplethysmography (PPG)

- **Allen 2007** — Allen, J. (2007). Photoplethysmography and its application in clinical physiological measurement. *Physiological Measurement*, 28(3), R1.
  *Foundational PPG review; cite when discussing pulse-wave morphology or HR derivation.*

- **Tamura et al. 2014** — Tamura, T., Maeda, Y., Sekine, M., & Yoshida, M. (2014). Wearable photoplethysmographic sensors—past and present. *Electronics*, 3(2), 282–302.
  *Wearable-PPG context; relevant for wrist PPG and motion artifacts.*

- **Castaneda et al. 2018** — Castaneda, D., Esparza, A., Ghamari, M., Soltanpur, C., & Nazeran, H. (2018). A review on wearable photoplethysmography sensors and their potential future applications in health care. *International Journal of Biosensors & Bioelectronics*, 4(4), 195–202.
  *Multi-wavelength PPG; cite when justifying use of green wavelength specifically (best motion robustness for wrist sites).*

- **Maeda et al. 2011** — Maeda, Y., Sekine, M., & Tamura, T. (2011). Relationship between measurement site and motion artifacts in wearable reflected photoplethysmography. *Journal of Medical Systems*, 35(5), 969–976.
  *Cite when justifying motion-artifact handling on wrist-worn PPG.*

## Velocity-based training (VBT) and resistance-training fatigue

- **González-Badillo & Sánchez-Medina 2010** — González-Badillo, J. J., & Sánchez-Medina, L. (2010). Movement velocity as a measure of loading intensity in resistance training. *International Journal of Sports Medicine*, 31(05), 347–352.
  *Foundational VBT paper. Cite when using bar/wrist velocity as a load/effort indicator.*

- **Sánchez-Medina & González-Badillo 2011** — Sánchez-Medina, L., & González-Badillo, J. J. (2011). Velocity loss as an indicator of neuromuscular fatigue during resistance training. *Medicine and Science in Sports and Exercise*, 43(9), 1725–1734.
  *The reference for using within-set velocity loss as a fatigue marker. Anchors the 20–30% MPV-loss thresholds used in VBT prescription.*

- **Weakley et al. 2021** — Weakley, J., Mann, B., Banyard, H., McLaren, S., Scott, T., & Garcia-Ramos, A. (2021). Velocity-based training: From theory to application. *Strength & Conditioning Journal*, 43(2), 31–49.
  *Modern practical review of VBT; useful for justifying real-time velocity feedback systems.*

- **Pareja-Blanco et al. 2017** — Pareja-Blanco, F., Rodríguez-Rosell, D., Sánchez-Medina, L., et al. (2017). Effects of velocity loss during resistance training on athletic performance, strength gains and muscle adaptations. *Scandinavian Journal of Medicine & Science in Sports*, 27(7), 724–735.
  *Cite when discussing training load prescription based on velocity loss.*

## Rate of Perceived Exertion (RPE)

- **Borg 1982** — Borg, G. A. (1982). Psychophysical bases of perceived exertion. *Medicine and Science in Sports and Exercise*, 14(5), 377–381.
  *Foundational RPE paper (original 6–20 Borg scale).*

- **Robertson et al. 2003** — Robertson, R. J., Goss, F. L., Rutkowski, J., Lenz, B., Dixon, C., Timmer, J., Frazee, K., Dube, J., & Andreacci, J. (2003). Concurrent validation of the OMNI perceived exertion scale for resistance exercise. *Medicine and Science in Sports and Exercise*, 35(2), 333–341.
  *Validates the 0/1–10 OMNI-RES scale used for resistance training. Cite when project uses 1–10 RPE in a strength context.*

- **Day et al. 2004** — Day, M. L., McGuigan, M. R., Brice, G., & Foster, C. (2004). Monitoring exercise intensity during resistance training using the session RPE scale. *Journal of Strength and Conditioning Research*, 18(2), 353–358.
  *Cite when using session-level RPE for training-load monitoring.*

- **Helms et al. 2016** — Helms, E. R., Cronin, J., Storey, A., & Zourdos, M. C. (2016). Application of the repetitions in reserve-based rating of perceived exertion scale for resistance training. *Strength & Conditioning Journal*, 38(4), 42–49.
  *Reps-in-reserve (RIR) RPE scale; useful when discussing how subjects estimate RPE near task failure.*

## Activity recognition and set detection from accelerometry

- **Bonomi et al. 2009** — Bonomi, A. G., Goris, A. H., Yin, B., & Westerterp, K. R. (2009). Detection of type, duration, and intensity of physical activity using an accelerometer. *Medicine & Science in Sports & Exercise*, 41(9), 1770–1777.
  *Cite when discussing acc-based activity detection or threshold selection.*

- **Mannini & Sabatini 2010** — Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying human physical activity from on-body accelerometers. *Sensors*, 10(2), 1154–1175.
  *Cite for window-based feature extraction from acc and standard activity-recognition pipelines.*

- **Khan et al. 2010** — Khan, A. M., Lee, Y. K., Lee, S. Y., & Kim, T. S. (2010). A triaxial accelerometer-based physical-activity recognition via augmented-signal features and a hierarchical recognizer. *IEEE Transactions on Information Technology in Biomedicine*, 14(5), 1166–1172.
  *Cite when using magnitude features or hierarchical classification for activities.*

## Cross-validation and physiological ML

- **Saeb et al. 2017** — Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019.
  *Cite when motivating subject-wise (LOSO/GroupKFold) cross-validation. Demonstrates that record-wise CV inflates accuracy.*

- **Little et al. 2017** — Little, M. A., Varoquaux, G., Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). Using and understanding cross-validation strategies. *GigaScience*, 6(5), gix020.
  *Companion to Saeb et al. with practical CV guidance for biomedical data.*

- **Esterman et al. 2010** — Esterman, M., Tamber-Rosenau, B. J., Chiu, Y. C., & Yantis, S. (2010). Avoiding non-independence in fMRI data analysis. *NeuroImage*, 50(2), 572–576.
  *Earlier articulation of the leakage problem from neuroimaging; useful when explaining why per-window splits fail.*

## Signal processing fundamentals

- **Welch 1967** — Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
  *Cite for any PSD computation (Welch's method is what scipy.signal.welch implements).*

- **Oppenheim & Schafer 2010** — Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
  *Reference textbook for IIR/FIR filtering, causal filtering, and persisted state.*

- **Welford 1962** — Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and products. *Technometrics*, 4(3), 419–420.
  *Cite for online running mean/variance (used in streaming z-score normalization).*

## Tools and frameworks

- **Makowski et al. 2021** — Makowski, D., Pham, T., Lau, Z. J., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689–1696.
  *Cite when using NeuroKit2 for ECG/EDA/PPG/EMG processing.*

- **Ke et al. 2017** — Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
  *Cite for choice of LightGBM as primary model.*

- **Lundberg & Lee 2017** — Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
  *SHAP. Cite for feature importance and interpretability.*

- **Akiba et al. 2019** — Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD '19: Proceedings of the 25th ACM SIGKDD*.
  *Cite for hyperparameter tuning.*

- **Virtanen et al. 2020** — Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
  *Cite scipy.signal usage.*

## Multi-task learning

- **Caruana 1997** — Caruana, R. (1997). Multitask learning. *Machine Learning*, 28(1), 41–75.
  *Foundational MTL paper. Cite when discussing whether to share parameters across the four tasks.*

- **Ruder 2017** — Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv preprint arXiv:1706.05098*.
  *Modern MTL survey; useful when justifying single-task vs multi-task choices in low-data regimes.*

- **Kendall et al. 2018** — Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7482–7491.
  *Cite when using uncertainty-weighted loss combination for unbalanced task losses.*

## Neural architectures for time-series and biosignals

- **Hochreiter & Schmidhuber 1997** — Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
  *Foundational LSTM paper. Cite for any LSTM-based architecture.*

- **Schuster & Paliwal 1997** — Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.
  *Cite for BiLSTM. Note: BiLSTM is non-causal and cannot be deployed in strict real-time pipelines.*

- **Yang et al. 2015** — Yang, J., Nguyen, M. N., San, P. P., Li, X. L., & Krishnaswamy, S. (2015). Deep convolutional neural networks on multichannel time series for human activity recognition. *Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI)*, 25, 3995–4001.
  *Cite for 1D-CNN applied to multichannel sensor data.*

- **Ordóñez & Roggen 2016** — Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition. *Sensors*, 16(1), 115.
  *DeepConvLSTM — canonical CNN-LSTM for multimodal wearable HAR. Direct analogue to this project's setup.*

- **Bai et al. 2018** — Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.
  *The TCN paper. Cite for any temporal convolutional network with dilated causal convolutions.*

- **Karpathy et al. 2014** — Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., & Fei-Fei, L. (2014). Large-scale video classification with convolutional neural networks. *CVPR*, 1725–1732.
  *Origin of CNN-LSTM hybrid architectures (originally for video; applies to 1D signals).*

## Multimodal fusion

- **Baltrušaitis et al. 2019** — Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423–443.
  *Canonical taxonomy of multimodal ML. Cite when comparing early/late/hybrid fusion strategies.*

- **Ramachandram & Taylor 2017** — Ramachandram, D., & Taylor, G. W. (2017). Deep multimodal learning: A survey on recent advances and trends. *IEEE Signal Processing Magazine*, 34(6), 96–108.
  *Practical survey of deep multimodal architectures.*

## Deep learning training and regularization

- **Loshchilov & Hutter 2019** — Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.
  *AdamW. Cite as the recommended optimizer for transformer/CNN training.*

- **Loshchilov & Hutter 2017** — Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR*.
  *Cosine annealing learning rate schedule with warm restarts.*

- **Smith 2018** — Smith, L. N. (2018). A disciplined approach to neural network hyper-parameters: Part 1 — learning rate, batch size, momentum, and weight decay. *arXiv preprint arXiv:1803.09820*.
  *Practical guide to NN hyperparameter selection.*

- **Goodfellow et al. 2016** — Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
  *Reference textbook. Cite for fundamental concepts (regularization, optimization, gradient clipping).*

## When to use which group of references in this project

| Topic | Primary references |
|-------|-------------------|
| EMG fatigue features (MNF, MDF, Dimitrov) | De Luca 1997, Cifrek et al. 2009, Dimitrov et al. 2006 |
| HRV from ECG | Task Force 1996, Shaffer & Ginsberg 2017 |
| EDA decomposition | Greco et al. 2016, Boucsein 2012 |
| PPG (green wavelength) | Allen 2007, Tamura et al. 2014, Maeda et al. 2011 |
| Rep counting and phase via velocity | González-Badillo & Sánchez-Medina 2010, Sánchez-Medina & González-Badillo 2011 |
| Velocity loss as fatigue | Sánchez-Medina & González-Badillo 2011, Pareja-Blanco et al. 2017 |
| RPE 1–10 in resistance training | Borg 1982, Robertson et al. 2003, Helms et al. 2016 |
| Active set detection from acc | Bonomi et al. 2009, Mannini & Sabatini 2010 |
| Subject-wise CV | Saeb et al. 2017, Little et al. 2017 |
| Causal/online filtering | Oppenheim & Schafer 2010 |
| Welch PSD | Welch 1967 |
| LightGBM | Ke et al. 2017 |
| SHAP | Lundberg & Lee 2017 |
| Optuna | Akiba et al. 2019 |
| NeuroKit2 | Makowski et al. 2021 |
| 1D-CNN for biosignals | Yang et al. 2015 |
| LSTM, BiLSTM | Hochreiter & Schmidhuber 1997, Schuster & Paliwal 1997 |
| CNN-LSTM (DeepConvLSTM) | Ordóñez & Roggen 2016, Karpathy et al. 2014 |
| TCN (temporal convolutional networks) | Bai et al. 2018 |
| Multimodal fusion | Baltrušaitis et al. 2019, Ramachandram & Taylor 2017 |
| Multi-task uncertainty weighting | Kendall et al. 2018 |
| AdamW optimizer | Loshchilov & Hutter 2019 |
| Cosine LR schedule | Loshchilov & Hutter 2017 |
| NN hyperparameter tuning | Smith 2018, Akiba et al. 2019 |

## Output template — `## References` section

Every deliverable (model_card.md, findings.md, quality_report.md) ends with:

```markdown
## References

- De Luca, C. J. (1997). The use of surface electromyography in biomechanics. *Journal of Applied Biomechanics*, 13(2), 135–163.
- Sánchez-Medina, L., & González-Badillo, J. J. (2011). Velocity loss as an indicator of neuromuscular fatigue during resistance training. *Medicine and Science in Sports and Exercise*, 43(9), 1725–1734.
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019.
- ...
```

Only include refs actually cited in the body. Order alphabetically by first author.

## Hard rules

- **Never invent references.** If you need one not on this list, write `[REF NEEDED: <topic>]` and ask the user.
- **Always cite when stating empirical claims.** "MNF declines during fatigue" needs a citation. "We use a 100 ms hop" doesn't (it's a project choice, not a claim about the world).
- **Always include `## References` in any deliverable that makes methodological claims.** model_card.md, findings.md, quality_report.md — all required.
- **Order alphabetically by first author** in the References section.
- **Quote the entry exactly** from this list — don't reformat or abbreviate.
