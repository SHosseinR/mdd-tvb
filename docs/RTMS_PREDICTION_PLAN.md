# rTMS response prediction: pre-specified analysis plan

**Written:** 2026-09-25, before any association between an EEG or model feature and a treatment outcome was
computed. Only the outcome counts had been tabulated (audit §F6).

## Question
Do baseline resting-EEG model parameters predict clinical response to rTMS, and do they add anything to
standard EEG markers and clinical variables?

## Data
- TDBRAIN `Dataset == MDD-rTMS`, session 1 (baseline) eyes-closed EEG; eyes-open for alpha reactivity only.
- Primary outcome: `Responder` (binary, as coded in TDBRAIN). Secondary outcomes: `Remitter`, and relative BDI
  change `(BDI_post − BDI_pre) / BDI_pre`.
- The protocol codes 1/2/3 are not defined in the dataset paper. The published cohort (Donse et al. 2018) used 10 Hz
  left DLPFC, 1 Hz right DLPFC, and a small bilateral group. The code-to-protocol mapping cannot be verified from
  the counts, so the code enters only as a categorical covariate and is never interpreted.

## Feature sets (all computed without outcome labels)
- **C (clinical):** age, sex, BDI_pre, protocol code (one-hot).
- **E (standard EEG markers, first-half spectra of the v2 cleaning):** frontal alpha asymmetry (ln α F4 − ln α F3),
  frontal theta (Fz, 4–7 Hz), posterior individual alpha frequency, posterior relative alpha power, frontal
  relative beta power (13–25 Hz), the theta/beta ratio at Fz, posterior alpha reactivity (EC/EO log ratio), and
  the aperiodic exponent (2–7 and 30–40 Hz fit, Cz).
- **M3 (frozen M5.3 model):** refined MAP parameters (TVB lead, v1 data). Only parameters recovered with r ≥ 0.5
  in the audit's synthetic test: global coupling, mean drive, a/b scale, the visual and dorsal-attention time
  contrasts, network × hemisphere gains except limbic/control, the shared-drive share, and the
  source-background and observation-noise fractions.
- **M4 (M5.4 model):** MAP parameters of the Whittle fit (TVB lead, v2 data), all free parameters except the tied
  limbic gain.

## Models and validation
- L2-regularised logistic regression (inverse regularisation chosen by an inner 5-fold CV from 10⁻³–10²),
  features standardised inside each training fold.
- Outer validation: stratified 5-fold CV repeated 50 times; report the mean AUC and balanced accuracy.
- Significance: label-permutation test (1,000 permutations of the full nested procedure, AUC statistic).
- Comparisons: C, E, M3, M4, C+E, C+E+M4. Incremental value is the difference in repeated-CV AUC, with a
  permutation null in which only the added block is permuted across subjects.
- Continuous outcome: ridge regression with the same nested scheme; report the out-of-fold R² and Spearman ρ.

## Pre-specified hypotheses
- **H1 (main, exploratory):** M4 predicts `Responder` with a cross-validated AUC above chance (permutation p < 0.05).
- **H2:** C+E+M4 improves on C+E (incremental permutation p < 0.05).
- **H3 (external group-difference replication):** the only Healthy–MDD difference that survived model changes in the
  audit (§5.6: lower left dorsal-attention input gain in MDD) is present in the rTMS MDD cohort vs the
  development Healthy group (M5.3 frozen parameters, age- and sex-adjusted, one-sided p < 0.05). The Healthy
  group is re-used, so this is a semi-independent replication (new MDD sample).

Expectation from the literature: previously reported EEG predictors of rTMS response did not replicate in a large
consortium re-analysis, and the best published result on this cohort (≈ 69 % balanced accuracy, 7,000+
features) is a training-set estimate. A null result is the most likely outcome and will be reported as such.
