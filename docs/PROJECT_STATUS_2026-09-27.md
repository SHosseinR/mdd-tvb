# MDD–TVB: where the project stands

**Date:** 2026-09-27 · **Branch:** `claude/audit-linear-regime` (all work committed and pushed; `main` untouched)
· **Detailed reports:** `INDEPENDENT_AUDIT_2026-09-24.md`, `NEXT_STEPS_REPORT_2026-09-25.md`,
`BETA_AND_NEW_DATASETS_2026-09-27.md`. A glossary of the technical terms is at the end (§9).

---

## 1. Summary in plain words

The project set out to fit a whole-brain model (The Virtual Brain, Jansen–Rit neural masses on a Schaefer-200
connectome) to each person's resting EEG, compare depressed with healthy people through the fitted parameters, and
later use the fitted models to design TMS treatment.

**The modelling machinery now works and is validated:**
- a fast, exact, stable-regime model of the EEG cross-spectrum;
- a proper statistical likelihood;
- parameters that are identifiable, reliable and honestly uncertain;
- tested on data never used for development, including three other laboratories' EEG systems.

**The scientific question it was pointed at does not have a positive answer in the available data:**
- The Healthy-vs-MDD contrast in TDBRAIN is mostly a recording-batch artefact.
- Across four independent sources, no EEG feature or model parameter separates depressed from healthy people
  consistently.
- Nothing measured from resting EEG predicts who responds to rTMS; only age and baseline severity do.

**The model itself still has three structural gaps:** it produces no individual beta rhythm, no frontal-midline theta,
and no non-sinusoidal alpha. More parameter tuning will not fix these; they need a new model component.

**What is needed next is mainly data, not code:**
- a larger depression dataset with properly matched controls (EMBARC, via the NIMH Data Archive);
- ideally TMS-evoked EEG from depressed patients (on request from published studies).

---

## 2. What the project is

| Component | What it is |
|---|---|
| Structural model | Schaefer-200 parcels, connection strengths and tract lengths from PyTepFit; fixed for everyone |
| Neural model | two Jansen–Rit populations per parcel (an "alpha" and a "fast" generator), delayed coupling, noise input |
| Forward model | how brain activity appears at the 26 scalp electrodes: TVB's analytic formula, or a head-model BEM |
| Data | TDBRAIN eyes-closed resting EEG, 26 channels: 176 Healthy, 151 MDD-indication, 164 rTMS-treated MDD; plus other TDBRAIN groups and three external datasets |
| Fitting | find, for each person, the model parameters whose predicted EEG cross-spectrum (2–40 Hz) matches the first half of the recording |
| Evaluation | predict the **second half** of the recording, never seen by the fit, and compare with simply predicting the population average |

---

## 3. What has been done, in order

### Phase A — original work (earlier sessions, milestones M1–M5.2)
Built the TVB model, the EEG forward model and a GPU simulator. M5.1 fitted each subject by choosing among 2,048
pre-simulated parameter sets ("bank"). It passed 9/11 of its acceptance gates, but failed alpha topography and the
lagged-connectivity group effect. M5.2 tried a better head model (BEM), which appeared worse.

### Phase B — independent audit (2026-09-24)
The audit found five causes behind the poor fit:

1. **The fitted model was a clock, not resting EEG.** In 318/327 subjects the best model oscillated by itself with
   the noise switched off (a "limit cycle"), which produced unrealistically regular EEG.
2. **The model class could not produce lagged (phase-shifted) connectivity**, which real EEG has.
3. **The objective had silently become alpha-only.** No theta or beta features survived its selection step, and the
   zero-lag spatial pattern was never scored.
4. **2,048 bank candidates for 15 parameters was far too coarse.**
5. **TVB's "sphere" lead field is an infinite-medium formula without a skull.**

**Built:** an exact analytic spectrum of the same model in its stable, noise-driven regime (validated against
simulation, r = 0.9997), continuous per-subject fitting on GPU, and a shared delayed alpha drive that restores lagged
connectivity. This became **M5.3**, which beat M5.1 on its own metric (total 0.745 → 0.688, alpha topography 0.912 →
0.510).

### Phase C — next steps (2026-09-25)

| Step | Result |
|---|---|
| **Froze M5.3** and **pre-registered an external test** | M5.3 reproduced exactly from the frozen settings |
| External test on **164 never-used rTMS patients** | M5.3 beat M5.1 as predicted: 0.806 → 0.698, paired −0.069 [−0.083, −0.048]; alpha topography 0.909 → 0.547 |
| **Re-cleaned the EEG** from raw files (re-implementation of the TDBRAIN authors' pipeline: eye-artefact regression, muscle/jump/blink detection, bridged-electrode repair) | split-half reliability up (theta maps 0.71 → 0.89, theta coherence 0.55 → 0.82); frontal eye artefact halved; 252/262 subjects kept |
| **Scored the models on plain, interpretable quantities** (spectrum, band maps, coherence) instead of the M5.1 feature space | **M5.1 and M5.3 predicted individual spectra and spatial coherence worse than the population average** (coherence 4–12× worse); the earlier "good fit" was a property of the old objective |
| **New objective (M5.4):** a proper statistical likelihood of the whole cross-spectrum (complex-Wishart/Whittle), muscle-prone channels excluded above 20 Hz, non-identifiable parameters fixed or tied, Laplace posteriors | final form: BEM head model, 10 spatial modes, plus the empirical population spectrum as a background term. See §4 |
| **rTMS response prediction** (pre-registered) | only clinical variables predict (AUC 0.67); EEG markers and model parameters do not (§4.5) |
| **Specificity check** of the one "robust" group difference | it is an acquisition-batch artefact (§4.4) |

### Phase D — latest (2026-09-26/27)
- Figures of the fitted signals (spectra, scalp maps, coherence, EEG traces; added by another session).
- A beta investigation: two model changes and one data analysis (§4.2).
- Three batch-free external datasets downloaded, cleaned identically and analysed with the final model (§4.4, §4.6).

---

## 4. What we now know

### 4.1 Data
- **TDBRAIN has an acquisition-batch confound.** Nearly all Healthy volunteers were recorded in an early batch
  (subject ID < 88000000), and every clinical group later. Features and parameters step at that boundary regardless
  of diagnosis (details in §4.4).
- **Muscle activity** contaminates 20–40 Hz at lateral channels (Fp1/Fp2/F7/F8/T7/T8) in most recordings. The new
  cleaning flags it, and the likelihood ignores those channels above 20 Hz.
- **Group effects at this sample size replicate only partly.** Independent random halves of the TDBRAIN subjects
  agree at r ≈ 0.5–0.67 for band topographies and r ≈ 0.07 for the full channel × frequency pattern. Gates built on
  the latter were unachievable.
- **Longer recordings (5 min) help a lot.** The same-subject noise ceiling drops from 0.69 (TDBRAIN, 2 min) to
  0.29–0.49.

### 4.2 The model (M5.4 final)

| Unseen half, error / population average (lower is better, < 1 beats the average) | M5.1 | M5.3 | **M5.4** | own first half (ceiling) |
|---|---|---|---|---|
| whole cross-spectrum (Wishart deviance) | — | — | **0.84** | 0.69 |
| log spectrum, all channels | 1.64 | 1.81 | **0.77** | 0.20 |
| alpha / beta topography | 1.27 / 1.50 | 1.39 / 1.47 | **0.98 / 0.84** | 0.16 / 0.15 |
| zero-lag coherence (θ/α/β) | 11 / 5.7 / 12 | 8.0 / 4.4 / 8.7 | **1.19 / 1.12 / 1.04** | 0.39 / 0.21 / 0.29 |
| theta topography | 2.17 | 2.11 | 1.16 | 0.23 |
| lagged coherency (α) | 1.29 | 1.09 | 1.02 | 0.46 |
| alpha peak frequency error (Hz; average 0.66) | 0.50 | **0.31** | 0.51 | 0.15 |

(252–262 TDBRAIN development subjects. M5.1 is scored on the original cleaning; M5.3, M5.4 and the ceiling are
scored on the re-cleaned data. The whole-cross-spectrum deviance exists only for the likelihood model.)

**What M5.4 does well:**
- It reproduces average spectra in every region (shape r 0.992–0.997) and spatial coherence (RMSE 0.08 vs 0.32
  for M5.3).
- It generates irregular, realistic EEG traces.
- The individually fitted neural parameters matter: freezing them at population values worsens the unseen fit for
  99.6 % of subjects.
- It transfers unchanged to other EEG systems (§4.6).

**What it does not do:**
1. **No individual beta.**
   - About half of posterior beta is a *harmonic* of non-sinusoidal alpha (significant alpha–beta phase coupling in
     54 % of subjects, beta peak at 2 × alpha frequency in 48 %). No linear model can produce this.
   - Central beta is an independent rhythm. Letting the model's beta generator vary per subject (globally, or only in
     sensorimotor cortex) changed nothing: posterior = prior, because the Jansen–Rit generator only makes broad bumps
     that background terms can imitate. Central beta is predicted at exactly the population-average level.
2. **No frontal-midline theta.** Theta topography is worse than the population average in all four datasets.
3. **Sharp alpha peaks come out too low.** The median predicted peak height is 74 % of measured, and 32 % of
   subjects are below half. The stable linear regime yields broad resonances.
4. **Much of the predicted power is borrowed** from the population template (median 62 %).
5. **The mean-drive parameter sits at its upper bound**, a sign the model wants something outside its certified range.
6. **Lagged connectivity is no longer carried by the shared-drive mechanism;** the population background absorbs it.

### 4.3 Inference quality (M5.4)

| Check | Result |
|---|---|
| Synthetic recovery: refit 120 simulated subjects | all 24 parameters recovered (median r = 0.88; weakest: mean drive 0.62) |
| 95 % posterior intervals contain the truth | 90 % (close to nominal) |
| Split-half calibration on real data | posteriors honest for the gains (variance of z ≈ 1); inflation applied where needed |
| Test–retest between recording halves | ICC 0.76–0.94 for every parameter |
| Head model | BEM beats TVB's formula for 82 % of subjects: the open question from the audit is settled |

### 4.4 The Healthy-vs-MDD signal
- **In TDBRAIN, the only robust parameter difference is an artefact.** Lower left dorsal-attention input in MDD
  (−0.51 SD, Holm p = 0.001) appeared in every model version. It is equally present in late-batch OCD, insomnia and
  ADHD patients, and absent in early-batch memory-complaint patients. Batch explains it (+0.42 SD, p = 0.001);
  depression does not (−0.04, p = 0.63).
- **At the data level, most TDBRAIN Healthy–MDD EEG differences are batch.** Alpha/beta topography and zero-lag
  coherence reproduce the batch difference (r = 0.72–0.90). This was estimated without any depressed subject, and
  none of it replicates in the external datasets.
- **One difference is real but not depression-specific: lagged alpha coherency.**
  - It separates *patients* from healthy people in TDBRAIN within the same batch, and in two independent clinical MDD
    datasets (r ≈ 0.7).
  - Depressed patients do not differ from other patients.
  - It is absent in medication-free students with high BDI or current MDD.
  - Medication or chronic illness are plausible explanations that cannot be tested here.
- **No model parameter shows a consistent depression effect** across TDBRAIN (batch-matched), MODMA, Mumtaz and
  ds003478. The best candidate is heterogeneous: +1.23, +0.48, −0.01, +0.04 SD; I² = 0.82.

### 4.5 rTMS treatment response (TDBRAIN, 163 patients, pre-registered)

| Predictors | Cross-validated AUC | Permutation p |
|---|---|---|
| Clinical (age, sex, baseline BDI, protocol) | **0.67** | 0.002 |
| 8 standard EEG markers | 0.50 | 0.50 |
| Model parameters (M5.3 or M5.4) | 0.47 | ≈ 0.8 |
| Clinical + EEG + model | 0.59 | adding the model does not help (p = 0.62) |

Older age and higher baseline severity predict lower response. This is consistent with the literature's
non-replication of EEG predictors of rTMS response.

### 4.6 External datasets (single system each, so no batch confound)

| Dataset | Subjects used | Model fit on unseen halves (ceiling) | Beats the average |
|---|---|---|---|
| MODMA (Lanzhou, 128-channel EGI) | 23 MDD, 29 HC | 0.86 (0.33) | 100 % |
| Mumtaz 2016 (19-channel) | 30 MDD, 28 HC | 0.86 (0.29) | 88 % |
| OpenNeuro ds003478 (Neuroscan 64) | 44 high-BDI, 71 low-BDI students | 0.87 (0.49) | 95 % |

The model generalises, but with more data per subject its structural error dominates: it captures 20–35 % of the
achievable improvement, against about 50 % in TDBRAIN. Mumtaz's two groups differ in overall power by about 2×,
which suggests they were recorded differently. Its large group differences should not be trusted on their own.

---

## 5. Problems, ranked

| # | Problem | Evidence | Status / fix |
|---|---|---|---|
| 1 | **No valid disease contrast in TDBRAIN** | batch step explains the group differences (§4.4) | use batch-matched comparisons or external data; needs **larger matched data** (EMBARC) |
| 2 | **Samples too small for the effects that exist** | external effects ≈ 0.2–0.3 SD; about 110 MDD pooled | EMBARC (≈ 300 MDD + controls, same sites) |
| 3 | **Model lacks narrow rhythms other than alpha** (beta, frontal theta) and non-sinusoidal alpha | two beta experiments: posterior = prior; theta worse than average everywhere | new component: resonators with free peak width (damped oscillators or a corticothalamic loop) plus a waveform term |
| 4 | **Heavy reliance on the population template** (62 % of power) | decomposition, ablation | shrinks only if (3) is solved |
| 5 | **Mean drive at its bound** | population fit and most subjects | re-examine the certified range or the excitability parameterisation |
| 6 | **Resting EEG does not constrain treatment response** | rTMS AUC 0.47–0.50 for all EEG/model features | stimulation-evoked data (TMS-EEG) would constrain the relevant dynamics; public TMS-EEG is healthy-only |
| 7 | Only eyes-closed data used | EO cleaned and on disk, used only for alpha reactivity | a joint EC/EO fit is a strong untested constraint |

---

## 6. What was investigated and ruled out

| Hypothesis | Test | Outcome |
|---|---|---|
| Delays alone create lagged connectivity | analytic model, stable regime | no (magnitude ~0.0003); a shared delayed drive does |
| The frozen audit model only works on development data | pre-registered external test (164) | ruled out: it replicated |
| Cleaning artefacts drive the poor fit | full re-cleaning from raw | cleaning helps reliability, but the model failures remain |
| The feature objective measured real fit | direct spectral scoring | ruled out: it hid 4–12× coherence errors |
| Neural parameters don't matter under the likelihood | freeze-neural ablation | ruled out: they improve fit for 99.6 % of subjects |
| TVB lead field is as good as BEM | likelihood comparison | ruled out: BEM better for 82 % |
| Left dorsal-attention deficit is a depression marker | specificity (4 clinical groups) and early-batch SMC | ruled out: acquisition batch |
| TDBRAIN Healthy–MDD EEG differences are depression effects | batch effect estimated without depressed subjects; external datasets | mostly ruled out: batch; lagged alpha is clinical but not depression-specific |
| Freeing the beta generator fixes beta | refit and recovery | ruled out: posterior = prior |
| A sensorimotor-specific beta generator fixes beta | refit and recovery | ruled out: no change on real data |
| Model or EEG features predict rTMS response | pre-registered nested CV, permutation | ruled out at this sample size |
| Any model parameter separates MDD across datasets | 3 external datasets + TDBRAIN batch-matched | none consistent |

---

## 7. Where things are

**Code (main entry points):**
- `src/mdd_tvb/linear_spectral.py`, `linear_jax.py`: the analytic model.
- `src/mdd_tvb/whittle.py`: the likelihood.
- `src/mdd_tvb/preprocess_v2.py`: EEG cleaning.
- `src/mdd_tvb/new_datasets.py`: readers for the external datasets.
- `scripts/m54/`: fitting (`m54_fit.py`), population fit, synthetic recovery, group analysis, evaluation, figures,
  Kaggle job generator.
- `scripts/datasets/`: download, cleaning and cross-dataset analyses.
- `scripts/rtms/predict_response.py`: response prediction.
- Tests: 42 passing.

**Data:**
- TDBRAIN at `D:/university/projects/graph-opt/tbdbrain/`.
- MODMA at `D:/university/projects/graph-opt/mdd-dataset-2/`.
- ds003478 and Mumtaz at `D:/university/projects/mdd-new-datasets/`.
- Cleaned spectra at `outputs/preproc_v2/<cohort>_<task>/`.

**GPU:** Kaggle jobs `mddtvb-*` under your account. Private datasets: `mdd-tvb-linear-bundle`, `mdd-tvb-linear-code`,
`mdd-tvb-external`.

**Pre-registrations:** `docs/M53_FROZEN_MODEL.md`, `docs/RTMS_PREDICTION_PLAN.md`.

---

## 8. Decisions and next steps

1. **Data access (only you can start these):**
   - EMBARC (NIMH Data Archive, collection 2199, https://nda.nih.gov/edit_collection.html?id=2199): about 300 MDD
     patients and healthy controls from the same sites, resting EEG eyes open and closed, sertraline/placebo outcome.
     It needs an NIH login and an institutional Data Use Certification. It is the right dataset to test a depression
     signal with adequate power and matched acquisition.
   - Requests to authors of MDD TMS-EEG studies (e.g. 133 MDD vs 76 controls; 53 MDD before/after rTMS): the only
     route to stimulation responses in depression.
2. **Model work, to be done if the goal still needs better resting-EEG fits:**
   - resonators with a free peak width for beta and frontal-midline theta, or a corticothalamic loop;
   - a waveform (harmonic) term for posterior beta;
   - then re-test the population-template share and the mean-drive bound.
3. **Not recommended now:** further tuning on TDBRAIN Healthy vs MDD, or any stimulation-protocol optimisation.
   Neither has a valid target yet.

---

## 9. Glossary

- **Cross-spectrum (CSD):** for every frequency, the power at each electrode and the shared activity between every
  pair of electrodes. It is the full second-order description of resting EEG.
- **Unseen half / ratio to the population average:** each recording is split in time. The model is fitted to the first
  half and judged on the second. Scores are errors divided by the error of simply predicting the average of other
  people. Below 1 means the individual model is better than the average.
- **Ceiling:** the score obtained by using the subject's own first half as the prediction. No model can be expected to
  do better.
- **Whittle / Wishart deviance:** the statistically correct measure of how well a predicted cross-spectrum explains
  measured data (a likelihood), rather than an ad-hoc feature distance.
- **Zero-lag vs lagged coherence:** zero-lag is shared activity with no time shift (largely volume conduction through
  the head). Lagged is shared activity with a time shift (genuine interaction between areas).
- **Lead field / BEM:** the mapping from brain sources to scalp electrodes; the BEM is a realistic head model with
  skull and scalp.
- **Limit cycle vs stable regime:** a model that oscillates on its own (clock-like) vs one whose rhythms are resonances
  excited by noise. Real resting EEG behaves like the latter.
- **Identifiable / recovery:** simulated subjects with known parameters are refitted; a parameter is identifiable if its
  estimate tracks the truth (r ≥ 0.5).
- **Posterior = prior:** the data did not change our belief about a parameter; the recordings carry no information
  about it.
- **Batch confound:** groups recorded at different times or with different equipment differ for technical reasons,
  which masquerade as group differences.
- **Holm correction / I²:** correction for testing many parameters; I² is the share of variation between datasets
  that is real heterogeneity rather than chance.
