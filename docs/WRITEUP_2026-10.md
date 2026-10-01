# Individual whole-brain models of resting EEG in depression: a validated likelihood approach, and what it can and cannot tell us

**Draft write-up, 2026-10** · Branch `claude/audit-linear-regime` of the MDD–TVB repository. This document
consolidates the project; the detailed reports are listed in Appendix A.

---

## Abstract

**Background.** Connectome-based neural-mass models (for example The Virtual Brain) promise individual,
mechanistic descriptions of brain activity. Fitted to resting EEG, they could yield biomarkers for psychiatric
disorders such as major depressive disorder (MDD). Most published fits, however:
- score a few spectral features;
- are rarely tested on unseen data;
- are rarely checked against acquisition confounds.

**Methods.** We fitted a delayed network of Jansen–Rit neural masses on a Schaefer-200 connectome to each person's
26-channel resting EEG.
- **Model regime:** we first showed that the original simulation-based fits sat in a self-oscillating (limit-cycle)
  regime. We replaced them with an exact analytic cross-spectrum of the stable, noise-driven regime.
- **Likelihood:** a complex-Wishart (Whittle) likelihood of the full 2–40 Hz cross-spectrum, with an explicit
  observation model.
- **Inference:** Laplace posteriors, checked by synthetic recovery, split-half calibration and test–retest
  reliability.
- **Test design:** fits use the first half of each recording; scoring uses the unseen second half, against a
  population-average baseline.
- **Data:** the TDBRAIN database (development cohort and a pre-registered external cohort of 164 rTMS patients) and
  three batch-free depression datasets from other laboratories (MODMA, Mumtaz 2016, OpenNeuro ds003478).
- **Extensions:** a joint eyes-open/eyes-closed fit, corticothalamic loops (private and shared thalamic nuclei), and
  prediction of event-related potentials (OpenNeuro ds003474) from each person's resting fit.

**Results.**
- **Fit and inference.** The final model (M5.4) predicts unseen data better than the population average in 99% of
  TDBRAIN subjects (median deviance ratio 0.84; same-subject ceiling 0.69). It transfers unchanged to three other EEG
  systems (0.86–0.87). Its parameters are recoverable (median r = 0.88), calibrated, and reliable (ICC 0.76–0.94).
- **Depression.** The Healthy-vs-MDD contrast in TDBRAIN is dominated by an acquisition-batch confound. No model
  parameter separated depressed from control participants consistently across datasets. Nothing from resting EEG
  predicted rTMS response; only age and severity did (AUC 0.67).
- **Eyes-open change.** A within-person contrast, the change in visual-network drive when the eyes open:
  - removes much of the batch effect;
  - differs with depressive symptoms in the batch-free data (pooled −0.36 SD, p = 0.013);
  - is not depression-specific among TDBRAIN patients.
- **Thalamic loop.** A per-person corticothalamic delay tracked individual alpha frequency in all seven cohorts and
  increased with age. Linear loops could not produce beta peaks.
- **Shared nuclei.** Letting groups of regions share thalamic nuclei (one per network × hemisphere, plus a diffuse
  matrix population) kept the fit gain: 79% of subjects better than M5.4. It also restored the spatial alpha
  coherence that private loops lost (64% of subjects better than private loops, p = 2×10⁻¹⁰).
- **Evoked responses.** A person's resting network predicted their held-out ERPs no better than anyone else's.

**Conclusions.**
- Likelihood-based, out-of-sample-validated fitting of whole-brain models to individual resting EEG is feasible and
  transferable.
- The parameters carry biological information (age, alpha frequency).
- In the available data they do not carry a replicable depression signal.
- Acquisition confounds and small effects, not model fitting, are the binding constraints.

---

## 1. Introduction

Whole-brain models couple neural masses through a structural connectome and map their activity to sensors through a
forward model. Groups have fitted such models, or simpler neural-field and neural-mass models, to resting EEG/MEG to
study ageing, anaesthesia, sleep and neurodegeneration (Robinson 2001; van Albada 2010; Hartoyo 2020; Ranasinghe 2022;
Griffiths 2020; Jafarian 2024).

Applications to depression are few and small. EEG markers of depression and treatment response generally fail to
replicate (Widge 2019; van der Vinne 2017).

This project began as a TVB pipeline that fitted each TDBRAIN subject to compare MDD with healthy controls and,
later, to plan TMS. An independent audit found that the early fits looked good for the wrong reasons. The project was
rebuilt around four principles:
1. a model in the regime resting EEG actually occupies;
2. a proper likelihood of the full cross-spectrum;
3. validation on unseen data and other datasets;
4. explicit checks for acquisition confounds.

This write-up reports the resulting method and its findings, including the negative ones.

## 2. Data

| Dataset | System | Use | Subjects (usable) |
|---|---|---|---|
| TDBRAIN development (van Dijk 2022) | 26-ch, 2-min EC/EO | development, Healthy vs MDD | 252 EC (Healthy, MDD), 213 with EO |
| TDBRAIN rTMS (pre-registered external) | same | external test, treatment response | 162–164 |
| TDBRAIN late-era clinical controls (OCD, insomnia, tinnitus, ADHD) | same | batch / specificity | 187 |
| TDBRAIN SMC (early era) | same | batch / specificity | 85 |
| MODMA (Cai 2022) | 128-ch EGI → 26 positions, 5-min EC | batch-free replication | 23 MDD, 29 HC |
| Mumtaz 2016 (figshare) | 19-ch, EC/EO | batch-free replication | 30 MDD, 28 HC |
| OpenNeuro ds003478 (Cavanagh) | 64-ch Neuroscan, alternating EO/EC | batch-free replication (BDI) | 44 high-, 71 low-BDI students |
| OpenNeuro ds003474 (same people) | probabilistic-selection task | evoked responses | 106 with all conditions |

**Preprocessing (v2)** re-implements the TDBRAIN authors' pipeline:
- EOG regression below 7 Hz;
- detectors for muscle, jumps, kurtosis, swings and blinks;
- detection of bridged electrodes and neighbour repair;
- tonic-muscle channels flagged rather than repaired.

It is applied identically to every dataset (mapped onto the 26 TDBRAIN positions). Cross-spectra use 4-s Hann
epochs (1-Hz bins, 2–40 Hz). The first and second halves of each recording share no samples.

## 3. Methods

### 3.1 Model

- **Regions:** 200 Schaefer parcels (7-network labels).
- **Connectome:** group-template streamline counts and tract lengths (log-transformed, row-normalised).
- **Neural masses:** each region has two Jansen–Rit populations, an "alpha" and a "fast" generator.
- **Coupling:** long-range, delayed; conduction speed is a population parameter.
- **Drive:** per-region noise, with gains per network × hemisphere; a shared delayed alpha drive; an aperiodic source
  background; diagonal observation noise.
- **Population background:** the empirical population cross-spectrum of the training subjects, entering with a fitted
  share. This makes the model nest the population-average baseline.

The spectrum is computed analytically in the **stable linear regime**:
- an equilibrium found by Newton iteration with implicit differentiation;
- closed-form node transfer functions;
- a network solve per frequency;
- a forward model through a template BEM lead field.

The analytic spectrum matches the stochastic simulator (log-spectrum r = 0.9997) at about 50× lower cost. Stability
is certified per state:
- node eigenvalues;
- distance from the Jansen–Rit fold;
- a small-gain bound, or the exact argument-principle test where needed (§3.5).

### 3.2 Likelihood

The complex-Wishart (Whittle) likelihood of the binned cross-spectrum:
- uses ν = 2.2 × epochs degrees of freedom (calibrated by simulation);
- is evaluated in observed coordinates, so that average reference, channel repair and missing channels are modelled
  exactly;
- excludes muscle-prone channels above 20 Hz (marginalised);
- uses the 10 principal spatial modes of the population spectrum;
- profiles out the overall scale per subject.

### 3.3 Inference and validation

- **Fitting:** a bank of 2,000 certified states is screened. Spatial and nuisance parameters are fitted first, then
  all free parameters are refined jointly (Adam). The best *certified* iterate is kept, and a Laplace posterior comes
  from the expected Fisher information.
- **Fixed parameters:** four non-identifiable globals (conduction speed, fast-generator ratio and fraction, noise time
  constant) are fixed at the population fit.
- **Scoring:** Wishart deviance on the unseen second half, divided by that of the population average (nested folds;
  training subjects only). The *ceiling* is the subject's own first half used as the prediction.
- **Interpretable blocks:** spectrum, band topographies, zero-lag and lagged coherence, alpha peak frequency, beta
  shape.
- **Recovery:** 120 synthetic subjects with known parameters, plus posterior coverage.
- **Calibration and reliability:** split-half calibration (variance of standardised differences) and test–retest ICC
  between halves.
- **Pre-registrations:** the frozen M5.3 external test, and rTMS response prediction.

### 3.4 Group analyses

- **Main test:** calibrated random-effects comparison (REML), with age and sex as covariates and Holm correction.
- **TDBRAIN batch decomposition:** early acquisition era (ID < 88000000) + clinical status + depression + age + sex,
  across all TDBRAIN cohorts.
- **Batch-matched contrast:** depressed vs other late-era patients.
- **Across datasets:** fixed-effect meta-analysis over the batch-free datasets.

### 3.5 Extensions

- **Joint eyes-closed + eyes-open fit.**
  - One parameter set explains both recordings.
  - A named subset may change with eyes open; recording nuisances are per condition.
  - One overall scale is shared, so the eyes-open power change must be explained by the model.
- **Corticothalamic loop.** A linearised relay + reticular circuit:
  `θ(ω) = g e^(−iωt0) L(1 − γL)/(1 + κL²)`.
  - The mean feedback is absorbed into the mean drive.
  - *Private* variants (T1/T2) give each region its own loop: a diagonal term in the network kernel.
    - T1: loop gains free per subject.
    - T2: gains and delay free, cortical time constants fixed.
  - *Shared* variants (S1/S2) use one core nucleus per network × hemisphere plus one diffuse matrix population:
    `P = (1 − m) P_core + m P_matrix`, with matrix share m.
  - Certification for private loops: a per-region scalar argument principle plus small gain over 0–150 Hz.
  - Certification for shared nuclei: the exact network argument principle on a 0.02 Hz grid; ambiguous phase steps
    are rejected.
- **Evoked responses.**
  - The ERP is the impulse response of each person's resting network to alpha-function inputs.
  - Spatial patterns are free over the 14 network × hemisphere groups.
  - Fitted on odd trials, scored on even trials.
  - Compared with the population network and other people's networks.

## 4. Results

### 4.1 Why the original fits failed (audit)

- In 318 of 327 subjects the original best-fitting state oscillated on its own when the noise was switched off. This
  limit cycle produced clock-like EEG.
- The feature objective had silently become alpha-only.
- The model class could not produce lagged connectivity.
- The bank was too coarse for 15 parameters.
- TVB's spherical lead field lacks a skull.

The audit's stable-regime model with a shared delayed drive (M5.3) was frozen and pre-registered. On 164 untouched
rTMS patients it beat the original as predicted (total 0.806 → 0.698, paired −0.069 [−0.083, −0.048]).

Scored directly on interpretable quantities, both earlier models predicted individual spectra and coherence *worse*
than the population average (coherence 4–12× worse). The earlier "good fit" was a property of the objective. This
motivated the likelihood model.

### 4.2 Fit and inference quality of M5.4

| Unseen half, error relative to the population average | M5.4 | Ceiling |
|---|---|---|
| Whole cross-spectrum (deviance) | **0.84** (99% beat the average) | 0.69 |
| Log spectrum | 0.77 | 0.20 |
| Alpha / beta topography | 0.98 / 0.84 | 0.16 / 0.15 |
| Zero-lag coherence θ/α/β | 1.19 / 1.12 / 1.04 | 0.39 / 0.21 / 0.29 |
| Alpha peak frequency | 0.77 | — |

- **Recovery:** 24/24 parameters, median r = 0.88; 95% intervals cover the truth in 90% of cases.
- **Reliability:** ICC 0.76–0.94.
- **Head model:** the BEM lead field beats TVB's formula for 82% of subjects.
- **Neural parameters matter:** freezing them at population values worsens the unseen fit for 99.6% of subjects.

### 4.3 Transfer to other EEG systems

- Unchanged except for each dataset's own population background, M5.4 beats the dataset's population average on
  unseen halves: MODMA 0.86 (100% of subjects), Mumtaz 0.86 (88%), ds003478 0.87 (95%).
- With 5-minute recordings the ceiling is much lower (0.29–0.49). The model then captures 18–26% of the achievable
  improvement, against 52% in TDBRAIN.

### 4.4 Structural limits of the model

- **Beta.**
  - About half of posterior beta is a harmonic of non-sinusoidal alpha: significant alpha–beta bicoherence in 54% of
    subjects, beta at 2× the alpha frequency in 48%.
  - Freeing the fast generator, globally or only in sensorimotor cortex, left the posterior equal to the prior.
  - A linear corticothalamic loop produced no beta peak in any of 1,008 settings.
- **Frontal-midline theta:** theta topography is worse than the population average in all datasets.
- **Peak sharpness:** sharp alpha peaks are under-predicted (median 74% of measured height).
- **Population template:** a large share of predicted power comes from the population template (median 62%).

### 4.5 Corticothalamic loops

**Private loops (one per region), development cohort:**

| | M5.4 | T1 | T2 |
|---|---|---|---|
| Median unseen deviance ratio | 0.840 | 0.827 | 0.829 |
| Subjects better than M5.4 | — | 75% | 61–62% |
| Beta shape / alpha frequency (ratio) | 0.900 / 0.771 | 0.878 / 0.725 | 0.870 / 0.625 |
| Zero-lag alpha coherence | — | worse | worse |

The population fit chose a loop delay of 87 ms (Robinson's range), and cortico-cortical coupling fell from 22 to 2.3.

**T2 on all seven cohorts:**
- The per-person delay correlates with individual alpha frequency in every dataset (ρ −0.19 to −0.56; 6 of 7
  significant).
- In TDBRAIN it rises with age (+0.16 SD per decade, p = 10⁻¹⁰), with no batch or depression effect.
- The fit gain holds within TDBRAIN's system but not on the other systems.

**Shared nuclei (S1/S2; development cohort, 252 subjects; `mddtvb-m54s-shared`).**

Setup:
- One core nucleus per network × hemisphere (14) plus one diffusely projecting matrix population.
- The population fit kept the loop near its starting values: gain 36, reticular fraction 0.50, delay 85 ms, matrix
  share 0.31. Cortico-cortical coupling was 3.3.
- Per subject, the free parameters are those of T1 (S1) or T2 (S2).

| Median, error relative to the population average | M5.4 | T1 | T2 | **S1** | S2 |
|---|---|---|---|---|---|
| Unseen deviance | 0.840 | 0.827 | 0.829 | **0.826** | 0.826 |
| Whole spectrum | 0.774 | 0.720 | 0.749 | **0.721** | 0.769 |
| Zero-lag alpha coherence | 1.116 | 1.133 | 1.164 | 1.086 | **1.052** |
| Alpha topography | 0.981 | 0.981 | 0.989 | 0.959 | **0.949** |
| Zero-lag beta coherence | 1.039 | 1.033 | 1.011 | 1.027 | **1.002** |
| Alpha peak frequency | 0.771 | 0.725 | **0.627** | 0.674 | 0.681 |
| Beta shape | 0.900 | 0.878 | **0.865** | 0.897 | 0.897 |

Paired per subject:
- **S1 vs M5.4:** unseen deviance better in 79% (p = 2×10⁻²⁴), spectrum in 78%, alpha frequency in 62%; coherence
  and topography no worse.
- **S1 vs T1:** same deviance (51%); zero-lag alpha coherence better in 64% (p = 2×10⁻¹⁰) and alpha topography in 64%
  (p = 6×10⁻⁹).
- **S2 vs T2:** coherence better in 72% (p = 3×10⁻¹⁴), but deviance and spectrum worse.
- **S2 recovery:** all 25 parameters recovered (median r = 0.89, delay r = 0.86, 95% coverage 0.91).
- **S2 delay:** still tracks alpha frequency (ρ = −0.29) and age (ρ = +0.29, both p < 10⁻⁵), though less tightly than
  with private loops.

**Reading.**
- Shared nuclei remove the coherence penalty of private loops, as the anatomy suggests: one nucleus drives many
  regions in phase (Saalmann 2012).
- S1 is the best resting-EEG model of the project.
- Beta is unchanged: shared or not, a linear loop does not make beta.

**Certification note.** Shared nuclei are certified by the exact network argument principle on a 0.02 Hz grid. A
cross-check against the exact per-region factorisation (private loops, 25 states) found the winding number correct
in every case. The additional rule that rejects any phase step above 1 rad was over-cautious: it rejected 44–72% of
genuinely stable states. The reported fits are therefore all stable and, if anything, slightly under-optimised.

### 4.6 Depression

**1. TDBRAIN's Healthy-vs-MDD contrast is an acquisition-batch artefact.**
- Healthy volunteers were recorded in an early era and patients later.
- The one robust parameter difference (left dorsal-attention gain, −0.51 SD) is equally present in late-era OCD,
  insomnia and ADHD patients, and absent in early-era memory-complaint patients. Batch explains it (+0.42 SD,
  p = 0.001); depression does not (−0.04, p = 0.63).
- Most data-level Healthy–MDD differences reproduce the batch pattern (r = 0.72–0.90).

**2. Across the batch-free datasets, no parameter is consistent.** The best candidate (inhibitory time scale) is
heterogeneous: +1.23, +0.48, −0.01 and +0.04 SD (I² = 0.82).

**3. Lagged alpha coherency** separates patients from healthy controls in two clinical datasets. It does not separate
depressed patients from other patients, and is absent in unmedicated high-BDI students.

**4. Within-person eyes-open change.**
- At the data level, the eyes-closed/eyes-open contrast halves the batch effect: alpha +0.34 → +0.11 SD; the
  correlation of depression and batch patterns falls from 0.91 to 0.55 (alpha) and from 0.89 to 0.0 (beta).
- In the joint fit, the best-predicting change (75% of subjects) was in visual-network drive:

| Contrast | Left | Right |
|---|---|---|
| TDBRAIN Healthy vs MDD | −0.29 SD (p = 0.025) | −0.41 SD (p = 0.001) |
| ds003478 high vs low BDI | −0.44 (p = 0.012) | −0.37 (p = 0.032) |
| Mumtaz MDD vs HC | −0.19 (n.s.) | −0.22 (n.s.) |
| **Batch-free pooled** | **−0.36 [−0.64, −0.08], p = 0.013** | **−0.31 [−0.58, −0.05], p = 0.021** |
| TDBRAIN depressed vs other late-era patients | +0.03 (p = 0.73) | −0.10 (p = 0.30) |

- The change weakens with age (p < 0.001), matching the data-level decline of alpha reactivity. Split-half ICC is
  0.5–0.7.
- It is a candidate for a pre-registered test, not an established marker.

### 4.7 rTMS response (pre-registered, 163 patients)

| Predictors | AUC |
|---|---|
| Clinical (age, sex, baseline BDI, protocol) | 0.67 (p = 0.002) |
| 8 standard EEG markers | 0.50 |
| Model parameters | 0.47 |
| Adding EEG / model to clinical | no improvement |

### 4.8 Positive controls

- **Age:** the M5.4 parameters predict age in the unseen rTMS cohort as well as plain spectral features do (r = 0.41
  vs 0.42; combined 0.48).
- **Thalamic delay:** it tracks alpha frequency in every dataset.
- **Eyes-open change:** it tracks age.

So the parameters carry biological information.

### 4.9 Evoked responses (ds003474)

- **Data:** the reward positivity is present (1.45 µV, t = 4.2), smaller with high BDI but not significantly
  (d = −0.21, p = 0.28).
- **Model, first version:** a brief input rings at alpha in the resting-fitted network (transfer 72 at 10 Hz vs 12 at
  5 and 20 Hz).
- **Model, revised (free spatial input patterns):**
  - own network R² 0.295 (stimulus) and 0.180 (feedback), vs 0.291 / 0.181 for other people's networks;
  - own better in 46–49% of subjects;
  - noise ceiling 0.79 / 0.69.
- **Conclusion:** the resting parameters carry no individual information about evoked responses, and cognitive ERPs do
  not constrain the network.

## 5. Discussion

### What worked
- **Regime and likelihood.** Moving to the stable regime and a proper likelihood made the fits honest: they predict
  unseen data, transfer across EEG systems, and yield identifiable, calibrated, reliable parameters.
- **Validation design.** Unseen halves, pre-registration, synthetic recovery and positive controls caught failures
  that feature-based fitting had hidden.

### What did not work, and why
- **No depression signal.**
  - In TDBRAIN the disease contrast is confounded by acquisition era.
  - In the batch-free datasets the effects are small (≈ 0.2–0.4 SD) at small sample sizes (≈ 110 depressed pooled).
  - What does replicate tends to separate patients from healthy controls rather than depression from other
    conditions.
- **Model structure:**
  - beta (a non-linear alpha harmonic plus an independent rolandic rhythm);
  - frontal-midline theta;
  - narrow spectral peaks.

  The model's most interpretable parameter, the thalamic delay, is biologically meaningful but not related to
  depression.

### Relation to the literature
- **Literature on fitting:** published EEG model fits mostly score selected features or a few sources, rarely use
  unseen data, and rarely check batch structure.
- **Literature on depression:** the depression literature's non-replication (Widge 2019) is mirrored here at the
  level of model parameters.
- **Batch-robust design:** within-person contrasts, used by Rowe 2004 and Hartoyo 2020 for alpha blocking, were the
  most batch-robust quantities we found.
- **Thalamus:** the corticothalamic delay result agrees with Robinson and van Albada's work. Shared thalamic nuclei
  (COALIA; Müller 2023) are motivated by the loss of coherence with private loops.

### Limitations
- **Recordings:** short TDBRAIN recordings (2 min).
- **Structure:** a template connectome and head model.
- **Linearity:** linear-regime dynamics cannot produce harmonics.
- **External data:** small samples.
- **Tasks:** cognitive task ERPs only, without simple sensory or TMS inputs.

### Recommendations
1. Test the eyes-open visual-drive change on an independent dataset with matched acquisition (e.g. EMBARC), with a
   pre-registered direction.
2. For the model:
   - adopt S1 (shared nuclei) as the resting model;
   - anatomical thalamocortical weights instead of network labels;
   - a non-linear (waveform) term for alpha harmonics;
   - region-specific resonators for rolandic beta and frontal theta.
3. For evoked constraints: steady-state (SSVEP/ASSR) or TMS-evoked data with known inputs.

## References (selection)

- Adams RA et al. (2022) Biol Psychiatry 91:202–215.
- Bastiaens SP et al. (2025) PLoS Comput Biol.
- Bensaid S et al. (2019) COALIA. Front Syst Neurosci.
- Cai H et al. (2022) MODMA. Sci Data.
- Cavanagh JF et al. (2019) Comput Psychiatry 3:1–17.
- Griffiths JD, McIntosh AR, Lefebvre J (2020) Front Comput Neurosci.
- Halgren M et al. (2019) PNAS 116:23772.
- Hartoyo A et al. (2020) PLoS Comput Biol 16:e1007662.
- Jafarian A et al. (2024) Hum Brain Mapp 45:e26782.
- Jansen BH, Rit VG (1995) Biol Cybern 73:357.
- Keren H et al. (2018) Am J Psychiatry 175:1111.
- Lopes da Silva FH et al. (1980) EEG Clin Neurophysiol 50:449.
- Müller EJ et al. (2023) Cell Rep 42:112844.
- Mumtaz W et al. (2016/2017) figshare 4244171.
- Nozari E et al. (2024) Nat Biomed Eng.
- Ranasinghe KG et al. (2022) eLife 11:e77850.
- Robinson PA et al. (2001) Phys Rev E 63:021903.
- Rowe DL et al. (2004) J Theor Biol 231:413.
- Saalmann YB et al. (2012) Science 337:753.
- Stroke corticothalamic MEG (2024) PNAS doi:10.1073/pnas.2409345121.
- van Albada SJ et al. (2010) Clin Neurophysiol 121:21.
- van der Vinne N et al. (2017) NeuroImage Clin 16:79.
- van Dijk H et al. (2022) TDBRAIN. Sci Data.
- Widge AS et al. (2019) Am J Psychiatry 176:44.

## Appendix A. Detailed reports

| Report | Content |
|---|---|
| `INDEPENDENT_AUDIT_2026-09-24.md` | why the original fits failed; analytic stable-regime model |
| `NEXT_STEPS_REPORT_2026-09-25.md` | external test, v2 cleaning, M5.4 likelihood, rTMS, batch confound |
| `BETA_AND_NEW_DATASETS_2026-09-27.md` | beta origin, three batch-free datasets |
| `LITERATURE_REST_AND_EVOKED_2026-09-27.md` | literature on model fits and evoked responses |
| `EOEC_THALAMUS_EVOKED_2026-09-29.md` | joint EO/EC, corticothalamic loops, evoked responses |
| `PROJECT_STATUS_2026-09-27.md`, `PRIMER_HOW_THE_PROJECT_WORKS.md` | status and plain-language method |

## Appendix B. Reproducibility

**Code:**
- `src/mdd_tvb/`: model, likelihood, cleaning, readers.
- `scripts/m54/`: fitting, population, synthetic recovery, group analyses, evaluation, joint and evoked models,
  Kaggle job generator.
- `scripts/datasets/`: downloads and cleaning.

**Tests:** `tests/` (all passing).

**GPU runs:** Kaggle kernels `mddtvb-*`, with private datasets `mdd-tvb-linear-bundle`, `-linear-code`, `-external`
and `-eo`.

**Pre-registrations:** `docs/M53_FROZEN_MODEL.md`, `docs/RTMS_PREDICTION_PLAN.md`.
