# Strength-RT — Komplett Claude Code Pakke

Sanntids estimering av fatigue (RPE), øvelse-klassifisering, fase og rep-telling fra 6 wearable biosignal-modaliteter, med Unix-time synkronisering mellom biosignaler og joint-angle data.

## Workflow (kjør i denne rekkefølgen)

```
1.  /inspect S001 1                 →  Se på faktiske data, lag findings.md (med refs)
2.  Oppdater CLAUDE.md fra findings (sample rates, kolonnenavn, etc.)
3.  /label --all                    →  Generer aligned_features.parquet per session
4.  /train baseline-lgbm            →  Tren 4 modeller, benchmark latency, model_card.md
5.  Iterer
```

Alle agenter siterer litteratur i sine deliverables (model_card.md, findings.md, quality_report.md). Sitater må komme fra `literature-references` skillet — som inneholder en kuratert bibliografi av kanoniske referanser (Dimitrov 2006 for FInsm5, Sánchez-Medina & González-Badillo 2011 for velocity-loss-fatigue, Saeb 2017 for subject-wise CV, etc.). Hooken `verify-references.sh` blokkerer skriving av deliverables som mangler `## References`-seksjon.

**Steg 1 er ufravikelig.** Inspeksjons-fasen ser på en faktisk recording, kjører gjennom alle signaler, lager plots og verifiserer Unix-time alignment. Resultatet er en `findings.md` som forteller deg hva som faktisk er i dataene dine — sample rates, hvilken PPG-kolonne som er grønn, om det er linjestøy som krever notch-filter, om sets matcher spreadsheet, osv. Resten av pipelinen leser fra dette i stedet for å gjette.

## Hva pakken gir deg

3 subagents, 10 skills, 3 slash-commands, 5 hooks, 3 tester, og config-stubs. Alt designet rundt din spesifikke setup (ECG/EMG/EDA/temp/acc/PPG-grønn med Unix-time stempler, joint-data som ground truth for fase/reps, RPE per sett i `participants.xlsx`).

```
strength-rt-v2/
├── CLAUDE.md                                 # Prosjektkontekst
├── configs/
│   ├── default.yaml                          # Hyperparam, vinduer, filtre
│   └── exercises.yaml                        # Per-øvelse: ROM, threshold, freq
├── tests/
│   ├── test_no_leakage.py
│   ├── test_no_joint_in_streaming.py
│   └── test_label_alignment.py
└── .claude/
    ├── settings.json
    ├── agents/
    │   ├── data-labeler.md                   # Krever inspeksjon kjørt først
    │   ├── biosignal-feature-extractor.md
    │   └── ml-expert.md
    ├── skills/
    │   ├── data-inspection/                  # Utforsk én recording grundig
    │   ├── data-loader/                      # Unix-time anchored
    │   ├── session-segmentation/
    │   ├── joint-angle-labeling/
    │   ├── multimodal-features/
    │   ├── emg-fatigue-features/
    │   ├── motion-rep-detection/
    │   ├── rpe-fatigue-modeling/
    │   ├── real-time-pipeline/
    │   ├── multi-task-evaluation/
    │   └── literature-references/            # NY: kuratert bibliografi
    ├── commands/
    │   ├── inspect.md                        # /inspect <subj> <sess>
    │   ├── label.md
    │   └── train.md
    └── hooks/
        ├── format-and-check.sh
        ├── check-no-filtfilt.sh
        ├── check-no-joint-in-streaming.sh
        ├── verify-labeled-data.sh
        ├── verify-references.sh              # NY: krever ## References i deliverables
        └── block-dangerous.sh
```

## Hvordan inspeksjonen fungerer

Når du kjører `/inspect S001 1`:

1. Lister filene i `data/raw/S001/1/`
2. Laster biosignal-fila ved native sample rate (ingen resampling enda)
3. Verifiserer at timestamps er Unix epoch (`>1e9`)
4. Beregner faktisk fs per kanal fra timestamp-deltas (ikke stoler på config)
5. Genererer per-kanal statistikk (range, NaN%, clipping)
6. Plotter:
   - `signal_overview.png` — alle kanaler over hele sesjonen
   - `signal_zoomed_set1.png` — første aktive sett, alle kanaler
   - `ppg_channel_check.png` — alle 4 PPG-wavelengths overlagt (slik du kan se hvilken som er grønn)
   - `psd_<channel>.png` — power spectrum per kanal (50/60 Hz line noise synlig)
   - `timestamp_alignment.png` — Unix-time coverage for biosignal vs joint
   - `sets_detected.png` — acc-magnitude med segment-grenser
   - `joint_coverage.png` — joint-data tilstedeværelse vs detekterte sett
7. Sammenligner detekterte sett med `participants.xlsx`
8. Skriver `findings.md` med konkrete action items for `CLAUDE.md` og `configs/default.yaml`

Alt lagres under `inspections/<subject>_<session>/`. Det blir audit-trail for resten av prosjektet.

## Synkronisering: Unix epoch som autoritativ klokke

Den største forenklingen i denne pakken: **alle filer bruker Unix epoch timestamps**. Det betyr:

- Biosignaler starter f.eks. ved `1714123456.789` (2024-04-26 10:30:56 UTC)
- Joint-data starter ved `1714123456.812` (23 ms senere)
- `participants.xlsx` har `set_start_unix` og `set_end_unix` på samme klokke

Synkronisering er da bare interpolasjon basert på Unix time:

```python
from scipy.interpolate import interp1d
f = interp1d(joint_df['t_unix'], joint_df['knee_angle'],
             bounds_error=False, fill_value=np.nan)
bio_df['knee_angle'] = f(bio_df['t_unix'])
```

Ingen DTW, ingen cross-correlation, ingen sync-pulse. Hvis Unix time faller utenfor joint-coverage (under hvile), blir verdien NaN — det er nøyaktig riktig oppførsel.

Alle loadere validerer `t_unix > 1e9`. Hvis du får en fil med sesjons-relative timestamps (start på 0), feiler loaderen høyt og krever forklaring.

## Slik tar du det i bruk

1. **Pakk ut over prosjektroten**:
   ```bash
   unzip strength-rt-v2-claude-pakke.zip
   cd <ditt-prosjekt>
   chmod +x .claude/hooks/*.sh
   ```

2. **Restart Claude Code-sesjonen** så agents/skills/hooks blir lest.

3. **Sjekk at alt er på plass**:
   - `/agents` skal vise 3 agents
   - `/hooks` skal vise 5 hooks
   - Det skal være 10 mapper under `.claude/skills/`

4. **Kjør første inspeksjon**:
   ```
   /inspect S001 1
   ```
   Eller, hvis du vil at hovedagenten skal velge: bare skriv "Inspiser en typisk session så vi vet hva vi jobber med."

5. **Les `inspections/S001_1/findings.md`** og oppdater:
   - Sample rate-tabellen i `CLAUDE.md`
   - PPG-grønn-kolonnen i `data-loader`-skillet (hvis ikke `ppg_green`)
   - Filterparameter i `configs/default.yaml` om nødvendig

6. **Kjør labeling og trening**:
   ```
   /label --all
   /train baseline-lgbm
   ```

## Designvalg som er viktige å forstå

1. **Inspect-først.** Sample rates, kolonnenavn og data-kvalitet bekreftes på faktiske data, ikke antas fra dokumentasjon. Hooks og validatorer refererer til `inspections/` som source of truth.

2. **Unix time som delt klokke.** All sync er via absolutt tid. Ingen fil i prosjektet skal ha sesjons-relative timestamps. Loadere håndhever dette.

3. **Joint-angle data ALDRI i streaming.** Hooken `check-no-joint-in-streaming.sh` AST-skanner alle filer i `src/streaming/` og blokkerer skriving som refererer joint-data. Den deployerte modellen ser bare biosignaler.

4. **Causal-only i streaming.** `check-no-filtfilt.sh` blokkerer non-causale operasjoner i streaming-koden. Bruk persisted-state alternativer (`sosfilt` med `zi`).

5. **State-machine før ML for reps og fase.** Acc-magnitude state-machinen er ofte god nok for styrketrening. ML brukes bare når state-machine F1 < 0.85 vs joint-angle ground truth.

6. **LightGBM før deep learning.** Med ~24 subjects × 9 sets = ~216 RPE-rader er du i low-data regime. Deep learning hjelper sjeldent her. Hvis modellen ikke virker med LightGBM, har du data-problem.

7. **Litteratur-siteringer er obligatoriske.** Hver metodologisk avgjørelse i model_card.md, findings.md og quality_report.md må siteres. Bibliografien er forhåndskuratert i `literature-references` skillet for å forhindre hallusinerte referanser. `verify-references.sh` blokkerer deliverables som mangler `## References`-seksjon. Hvis en nødvendig referanse ikke er i bibliografien, skriv `[REF NEEDED: <topic>]` og spør brukeren — ikke finn på.

## Når du sitter fast

| Symptom | Sannsynlig årsak | Hva sjekke |
|---------|------------------|------------|
| Loaderen klager på timestamps | Filer har sesjons-relative timestamps | Sjekk recorder-konfigurasjonen — bør gi Unix epoch |
| `/inspect` finner ikke PPG-grønn | Kolonnenavn matcher ikke aliases | Se ppg_channel_check.png, oppdater `PPG_GREEN_ALIAS` i data-loader skill |
| Detekterte sett != spreadsheet | Threshold for høy/lav, eller spreadsheet-tider feil | Inspeksjons sets_detected.png + sammenlign med set_start_unix/set_end_unix |
| Joint-coverage er sparsom utenfor sett | Forventet — joint data finnes kun under aktive sett | OK |
| Joint-coverage mangler I et sett | Tracking failure under det settet | Flagg settet, ekskluder fra rep/phase ground truth |
| Fatigue MAE > 1.5 | Data-problem | Plot MNF over tid på fatiguing set; må trende ned |
| Latency p99 > 100 ms | Tung feature i streaming | Fjern fra feature-importance-bunnen |
| Rep-counter feiler på pushups | Threshold for høy | Senk `acc_peak_threshold_g` for pushup i exercises.yaml |

## Hvorfor 3 agenter

`data-labeler` (offline preprocessing), `biosignal-feature-extractor` (offline + online parity), `ml-expert` (4 oppgaver). Hver har klart adskilt ansvar og leverer en strukturert handoff til neste. Færre subagents = bedre helhetsoversikt for hovedagenten; flere = unødvendig context-gatekeeping.

## Neste konkrete steg

1. Pakk ut, kjør `/inspect` på en typisk session
2. Les `findings.md`, oppdater `CLAUDE.md`
3. `/label` på første subject for å verifisere flyt
4. `/train baseline-lgbm` for å etablere tallene som må slås
5. Iterer features og data-kvalitet basert på model_card.md
