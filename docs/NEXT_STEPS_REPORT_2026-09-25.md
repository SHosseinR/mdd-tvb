# Next steps carried out: external test, re-cleaned EEG, likelihood objective, rTMS outcomes

**Date:** 2026-09-25 · **Follows:** `docs/INDEPENDENT_AUDIT_2026-09-24.md` (the audit) · **Branch:**
`claude/audit-linear-regime`

This report covers the recommendations of the audit (§6) that were carried out in this session, in the order they
were done. Every number comes from files under `outputs/` produced by the scripts named in §9.

---

## 0. Bottom line

__BOTTOM_LINE__

---

## 1. What was done

| Audit recommendation | Status | Where |
|---|---|---|
| 0. Freeze the final model as the M5 baseline | done before any external data were touched: `docs/M53_FROZEN_MODEL.md`, `configs/m53_frozen/`; exact reproduction checked on the GPU | §2 |
| Untouched external test | done on 164 rTMS-treated MDD patients (never used before), with the pre-specified primary outcome | §2 |
| 6. Re-preprocess the EEG with EOG/EMG handling | done: re-implementation of the TDBRAIN authors' pipeline from the raw BDF files, eyes closed and eyes open | §3 |
| 3. Rebuild the objective (all bands, zero-lag structure, EMG) | done: complex-Wishart (Whittle) likelihood of the full cross-spectrum, muscle-prone channels marginalised above 20 Hz | §4, §5 |
| 4. Fix/tie non-identified parameters; posterior instead of MAP | done: speed, fast-generator ratio/fraction, noise τ fixed; limbic and control gains tied; Laplace posteriors from the Fisher information, calibrated on split halves | §5, §6 |
| 5. Both lead fields until zero-lag structure is scored | done: TVB and BEM fitted under the new objective | §5 |
| 7. Age/sex adjustment | done in every group comparison | §7 |
| 8. Gates against noise ceilings and replicability | done: same-subject ceilings and independent-subject replicability reported for every block | §3, §4 |
| 9. rTMS outcomes before any stimulation work | done: pre-specified prediction analysis (`docs/RTMS_PREDICTION_PLAN.md`) | §8 |
| 2. Competing lag mechanisms; 10. homotopic connections | not done (see §10) | — |

---

## 2. The frozen model on an untouched cohort

**Freeze.** The final model of audit §5.4 (M5.3) was frozen before any external recording was preprocessed:
population fits, bank recipes, fitting settings and the external-test protocol are in `docs/M53_FROZEN_MODEL.md`
and `configs/m53_frozen/` (commit `3ebda8a`). A reproduction run regenerated the banks on the GPU and refitted the
262 development subjects. It selected the same global state for 262/262 subjects, and the unseen-half scores were
identical (r = 1.000).

**Cohort.** TDBRAIN `MDD-rTMS`, session 1, eyes closed: 164 patients, disjoint from the development set and from the
consumed M5.1 holdout. They were preprocessed with the original EEGLAB script, changed only in its group
selection (`scripts/external/preprocess_tdbrain_rtms_restEC.m`), and all 164 passed its QC. Spectra came from the
unchanged M5 extraction. The transformer and pooled null were learned on all 262 development subjects.

**Result (pre-specified primary outcome: paired difference in unseen total, M5.3 TVB refined − M5.1):**

| Unseen / pooled null (median) | M5.1 bank | **M5.3 frozen, TVB** | paired change [95 % CI] | M5.3 frozen, BEM | paired change [95 % CI] |
|---|---|---|---|---|---|
| total | 0.806 | **0.698** | **−0.069 [−0.083, −0.048]** | 0.713 | −0.065 [−0.094, −0.037] |
| autospectrum | 0.498 | 0.438 | −0.046 [−0.058, −0.018] | 0.403 | −0.065 [−0.096, −0.043] |
| lagged coherency | 1.053 | 0.975 | −0.033 [−0.051, +0.008] | 0.976 | −0.045 [−0.077, −0.013] |
| alpha topography | 0.909 | 0.547 | −0.325 [−0.419, −0.256] | 0.801 | −0.066 [−0.170, +0.001] |
| subjects beating the null | 72 % | 80 % | | 79 % | |

The development result replicates on new patients with no loss of margin: the total improves by 0.069 in the
median (0.052 in development), and the alpha-topography gain is just as large. The prediction written into the freeze
document (M5.3 better, largest gain on topography) holds.

---

## 3. Re-cleaned EEG (v2)

**Pipeline** (`src/mdd_tvb/preprocess_v2.py`): a re-implementation of the TDBRAIN authors' automatic pipeline
(github.com/brainclinics/TDBRAIN), with their thresholds, working from the raw BDF files:
- bipolar EOG; 50 Hz notch; 0.5–100 Hz band-pass;
- EOG regression;
- detection of EMG bursts, jumps, kurtosis, extreme voltage swings and residual blinks;
- repair of bad or bridged channels from the authors' neighbour table.

Three deliberate deviations:
- the EOG regression uses one set of coefficients per recording, fitted and applied below 7 Hz, so frontal alpha that
  leaks into the EOG electrodes is not subtracted;
- channels dominated by tonic muscle activity are kept but flagged, instead of interpolated from neighbours that
  are usually also muscular; the likelihood drops their high frequencies (§4);
- spectra use only 4-s epochs that lie entirely in artefact-free time, and the two halves share no samples.

**Yield.** Eyes closed: 317/327 original recordings and 163/164 rTMS recordings usable. Of the 262 development
subjects, 252 remain. The 10 lost are 9 Healthy and 1 MDD: badly contaminated recordings, 8 of them from the earliest
Healthy batch (IDs 8795xxxx–8798xxxx), with up to 18 channels needing repair or kurtosis artefacts in 45 % of the
data. Median usable epochs per half: 24 (v1: 28). Eyes open: 307/327 and 154/164 usable.

**Data quality on the same 252 subjects** (`scripts/preprocess/compare_v1_v2.py`):

| | v1 (original) | v2 (re-cleaned) |
|---|---|---|
| split-half reliability, theta topography | 0.71 | **0.89** |
| split-half reliability, theta zero-lag coherence | 0.55 | **0.82** |
| split-half reliability, beta zero-lag coherence | 0.69 | **0.86** |
| split-half reliability, alpha / beta topography | 0.88 / 0.87 | 0.90 / 0.92 |
| split-half reliability, lagged alpha coherency / alpha frequency | 0.83 / 0.86 | 0.80 / 0.79 (fewer epochs) |
| frontal (Fp1/Fp2) vs central 2–4 Hz power | 4.9× | **2.3×** (ocular artefact halved) |
| 20–40 Hz slope, lateral vs central channels | −1.77 vs −3.60 | −1.86 vs −3.65 (tonic EMG remains; handled in the likelihood) |
| tonic-EMG channels flagged | — | 79 % of recordings have ≥ 1 (mostly Fp1/Fp2/F7/F8/T7/T8, as found in the audit) |
| bridged electrode pairs | — | 33 recordings |

**How replicable are the Healthy–MDD effects at all?** This is the ceiling for any group-effect gate. The median
correlation of age/sex-adjusted effects between two disjoint random halves of the subjects (≈ 126 per half):

| block | v1 | v2 |
|---|---|---|
| alpha topography | 0.66 | 0.67 |
| beta topography | 0.64 | 0.67 |
| beta zero-lag coherence | 0.64 | 0.60 |
| lagged alpha coherency | 0.56 | 0.54 |
| theta topography | 0.45 | 0.51 |
| full channel × frequency log power | **0.06** | **0.09** |
| lagged theta / beta coherency | 0.22 / −0.07 | 0.17 / −0.15 |

The channel × frequency power effect used by the M5.2 gate ("power group effect r ≥ 0.30") does not replicate
between independent halves of these subjects. A model cannot be required to reproduce it. Group-effect gates
should be restricted to alpha/beta topography, beta zero-lag coherence and lagged alpha coherency.

---

## 4. What the earlier objective did and did not fit

The M5.1 score is a weighted distance between reliability-selected, PCA-projected, standardised features. The
audit showed it had become alpha-only. To see what the models actually predict, `scripts/m54/m54_evaluate.py`
scores the unseen half directly on interpretable blocks. Each score is the error relative to the pooled population
null (< 1 beats the population average). The same-subject first half gives the noise ceiling.

| Unseen / null, median | noise ceiling (own first half) | M5.1 | M5.3 TVB | M5.3 BEM |
|---|---|---|---|---|
| log spectrum, all channels × 2–40 Hz | 0.21 | 1.64 | 1.74 | 1.57 |
| theta / alpha / beta topography | 0.25 / 0.15 / 0.16 | 2.17 / 1.27 / 1.50 | 2.01 / 1.35 / 1.42 | 2.73 / 1.61 / 1.55 |
| zero-lag coherence θ / α / β | 0.39 / 0.20 / 0.28 | 11.3 / 5.7 / 11.6 | 7.4 / 4.1 / 8.1 | 3.1 / 2.7 / 3.6 |
| lagged coherency θ / α / β | 1.31 / 0.41 / 0.90 | 1.15 / 1.29 / 1.46 | 1.23 / 1.02 / 1.45 | 1.36 / 1.12 / 1.65 |
| alpha peak frequency (abs. error; null 0.66 Hz) | 0.04 | 0.44 (0.50 Hz) | 0.35 (0.37 Hz) | 0.40 (0.44 Hz) |

(development subjects, v1 data; the external cohort gives the same picture, `outputs/m54_eval/ext_v1_all`.)

**Both the old and the audited model predict individual spectra, band topographies and zero-lag coherence worse than
the population average.** The only exception is the alpha peak frequency. Zero-lag coherence, the dominant feature
of scalp EEG, is 3–12 times worse than the null. The good scores of audit §5.4 were good scores on a feature
space that throws most of the spectrum away. This is the strongest argument for changing the objective. The BEM
reproduces zero-lag coherence 2–3× better than the TVB formula, as the audit's volume-conduction analysis predicted.

---

## 5. M5.4: likelihood-based fitting

__M54__

---

## 6. Identifiability, calibration and reliability of the parameters

__RECOVERY__

---

## 7. Healthy vs MDD-indication at the parameter level

__GROUP__

---

## 8. Does any of this predict rTMS response?

Pre-specified in `docs/RTMS_PREDICTION_PLAN.md` before any outcome association was computed. 163 patients with v2
spectra (96 responders). L2 logistic regression, inner CV for regularisation, stratified 5-fold CV repeated 50×,
label-permutation p values (500 permutations).

__RTMS__

**H3: the group difference, replicated in a new MDD sample.** The only parameter difference that survived the audit's
model change was a lower left dorsal-attention input gain in MDD (§5.6 of the audit). With the frozen M5.3
parameters, it is again lower in the 164 rTMS patients than in the development Healthy group: −0.43 SD
[−0.65, −0.21], one-sided p = 8 × 10⁻⁵, age/sex adjusted. The rTMS patients do not differ from the development MDD
group (−0.06 SD, p = 0.59). The parameter shows no recording-era trend within the Healthy group (r = −0.09 with
subject ID). Caveats: the Healthy group is re-used, so this is semi-independent. Healthy recordings are on average
older than MDD recordings, so an era effect cannot be fully excluded.

---

## 9. Reproducibility

__REPRO__

---

## 10. What is left and what I recommend next

__NEXT__
