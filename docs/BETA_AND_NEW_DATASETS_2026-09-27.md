# Beta and three batch-free depression datasets

**Date:** 2026-09-27 · **Follows:** `docs/NEXT_STEPS_REPORT_2026-09-25.md` · **Branch:** `claude/audit-linear-regime`

Two questions from the last round:
1. Why does the model miss individual beta, and can it be fixed?
2. Is there any depression signal once the acquisition-batch confound of TDBRAIN is removed?

---

## 0. Bottom line

1. **Beta has two sources, and the model can reach only one.** About half of posterior beta is a harmonic of
   non-sinusoidal alpha: significant alpha–beta phase coupling in 54 % of subjects, with the beta peak at twice the
   alpha frequency in 48 %. No linear model can produce harmonics. Central (rolandic) beta is an independent rhythm.
   Freeing the global beta generator changed nothing, and neither did a sensorimotor-specific one. In both cases the
   posterior equals the prior on real data. The Jansen–Rit generator makes only broad resonances that the background
   terms can imitate. Individual beta needs a generator with a free peak width (§1.3).
2. **The model generalises to other EEG systems.** Unchanged except for each dataset's own population background, it
   beats the dataset's population average on unseen halves:

   | dataset | unseen deviance ratio | subjects beating the average |
   |---|---|---|
   | MODMA (128-channel EGI) | 0.86 | 100 % |
   | Mumtaz (19-channel) | 0.86 | 88 % |
   | ds003478 (64-channel Neuroscan) | 0.87 | 95 % |

   With longer recordings, however, the ceiling is much lower (0.29–0.49 vs 0.68). The model's structural error
   then dominates: it captures 18–26 % of the achievable improvement (median over subjects), against 52 % in TDBRAIN.
3. **No model parameter separates depressed from control subjects consistently across datasets.**
   - The strongest candidate, the inhibitory time scale, is highly heterogeneous across datasets (I² = 0.82): +1.23 SD
     in Mumtaz, +0.48 in ds003478, −0.01 in MODMA, +0.04 in TDBRAIN batch-matched.
   - The left dorsal-attention gain, TDBRAIN's batch artefact, is −0.19 to −0.27 SD in all three new datasets
     (pooled −0.24, p = 0.08). That is too small to establish.
4. **At the data level, one Healthy–patient difference is real but not depression-specific.**
   - Lagged alpha coherency differs between patients and healthy controls in two independent clinical MDD datasets
     (pattern r ≈ 0.7 with TDBRAIN) and in TDBRAIN's same-batch memory-complaint patients.
   - Depressed patients do not differ from other patients.
   - It is absent in medication-free students with high BDI or current MDD.
   - The TDBRAIN topography and zero-lag differences do not replicate anywhere (batch, as concluded before).
5. **Consistent weak spots of the model:** theta topography and theta zero-lag coherence (worse than the population
   average in all four datasets), individual beta, and the height of sharp alpha peaks.

---

## 1. Beta

### 1.1 Freeing the global fast generator does nothing
The fast generator's time-scale ratio and power fraction were fitted per subject on the 252 TDBRAIN development
subjects, followed by synthetic recovery on 120 subjects (`mddtvb-m54c-fast`).

| | fixed (current) | free |
|---|---|---|
| beta shape 13–30 Hz, unseen / average (ceiling 0.31) | 0.900 | 0.901 |
| whole-spectrum deviance | 0.840 | 0.840 (paired −0.0001) |
| recovery r (ratio / fraction) | — | 0.47 / 0.39 |
| posterior SD / prior SD | — | 1.00 / 1.00 |

The data carry no information about these two parameters. The fast generator adds a broad hump that the aperiodic
background terms can imitate.

### 1.2 Where resting beta comes from (`scripts/preprocess/beta_origin_check.py`)
120 TDBRAIN development subjects, v2 cleaning, non-overlapping 4-s epochs. The bicoherence at (IAF, IAF) → 2·IAF
measures quadratic phase coupling between alpha and its harmonic. The surrogate pairs the harmonic from a different
epoch.

| | posterior (O1/Oz/O2/P3/Pz/P4) | central (C3/Cz/C4) |
|---|---|---|
| bicoherence median (surrogate) | 0.24 (0.11) | 0.14 (0.11) |
| subjects with bicoherence > 2 × surrogate | 54 % | 30 % |
| beta peak within ±1 Hz of 2 × IAF | 48 % | 42 % |
| correlation of bicoherence with beta-peak height | 0.32 | 0.04 |

Posterior beta is largely an alpha *harmonic*: the waveform of alpha is not sinusoidal. Central beta is mostly an
independent rhythm. A linear Gaussian model can never produce harmonic beta. Scoring posterior beta therefore
penalises every linear model equally; it is not a parameter problem.

### 1.3 A sensorimotor-specific beta generator
The model code now accepts per-region fast-generator parameters (`linear_jax`, backward compatible; all tests pass).
The somatomotor network gets its own fast-generator ratio and power fraction (2 parameters, prior centred on the
population value), fitted in the continuous refinement (`--somot-beta`; job `mddtvb-m54d-somot`).

| | current model | + somatomotor beta generator |
|---|---|---|
| beta shape 13–30 Hz, non-muscle channels (unseen / average) | 0.900 | 0.900 |
| beta shape at central channels (FC/C/CP rows) | 1.006 | 1.006 (paired 0.000) |
| whole-spectrum deviance | 0.840 | 0.840 |
| synthetic recovery r: ratio / fraction | — | 0.49 / 0.66 |
| posterior SD / prior SD on real data | — | 0.97–0.99 |

It changes nothing on real data. The parameters are partly recoverable when the data come from the model itself, but
real recordings do not inform them. Central beta is predicted exactly at the population-average level by both
versions. Both attempts point at the generator, not its parameters. A Jansen–Rit node at beta time scales produces a
broad resonance, and a broad hump is interchangeable with the aperiodic and population-background terms. Individual
beta needs a source whose peak *width* is free: a damped-oscillator source with its own quality factor (as in the
"sum of damped oscillators" accounts of alpha and 1/f), or the corticothalamic loop, whose delay sets a narrow
resonance. Posterior (harmonic) beta additionally needs a waveform term, which no linear model has.

---

## 2. Three batch-free datasets

| | MODMA | Mumtaz 2016 | OpenNeuro ds003478 |
|---|---|---|---|
| source | Lanzhou University (local copy) | figshare 4244171 | Cavanagh, OpenNeuro |
| participants | 24 MDD (MINI, PHQ-9), 29 controls | 34 MDD, 30 controls | 122 students selected by BDI; 11 with current MDD (SCID) |
| system / channels | EGI HydroCel 128, mapped to the 26 TDBRAIN positions | 19-channel 10–20 (7 of the 26 missing) | Neuroscan 64, all 26 TDBRAIN labels + HEOG/VEOG |
| eyes closed | 5 min | 5 min | 1-min blocks alternating with eyes open (~200 s) |
| usable after v2 cleaning | 52 (23 MDD / 29 HC) | 58 (30 / 28) | 117 (44 high BDI / 71 low BDI / 2 mid) |

**Processing.** The same artefact-aware cleaning was used as for TDBRAIN (`src/mdd_tvb/preprocess_v2.preprocess_array`),
generalised to any channel subset, any mains frequency and eyes-closed masks. The TDBRAIN output is reproduced to
10⁻⁹. Dataset-specific details:
- **MODMA:** the unstored Cz reference was restored and the data average-referenced over all 129 electrodes before
  picking the 26 positions. 20 positions come from EGI's published 10–20 equivalents; FC3/FCz/FC4/CP3/CPz/CP4
  (E29/E6/E111/E42/E55/E93) by nearest electrode after aligning on those anchors.
- **Mumtaz:** the 7 missing channels are marginalised exactly in the likelihood (test in `tests/test_whittle.py`).
- **ds003478:** 33 files code events only numerically; odd codes are eyes closed.
- **Downloads:** three downloader problems were found and fixed. One Mumtaz control file (`H S12 EC.edf`) has been
  withdrawn from figshare.

### 2.1 Data level: does TDBRAIN's Healthy–MDD difference appear? (`scripts/datasets/cross_dataset_effects.py`)
Each entry is the correlation (95 % bootstrap CI) between the new dataset's case − control pattern (age/sex adjusted
where available) and TDBRAIN's MDD − Healthy pattern, on the same channels:

| feature | MODMA | Mumtaz | ds003478 (high vs low BDI) |
|---|---|---|---|
| alpha topography | +0.10 [−0.34, 0.40] | +0.32 [−0.14, 0.65] | +0.28 [−0.47, 0.70] |
| beta topography | +0.25 [−0.26, 0.50] | −0.30 [−0.63, 0.27] | −0.24 [−0.55, 0.31] |
| theta topography | +0.42 [−0.17, 0.59] | +0.21 [−0.05, 0.41] | −0.05 |
| alpha zero-lag coherence | +0.28 [−0.42, 0.56] | +0.07 [−0.19, 0.26] | +0.16 |
| **alpha lagged coherency** | **+0.72** [−0.32, 0.82] | **+0.70 [0.07, 0.81]** | −0.23 [−0.65, 0.53] |

In TDBRAIN, memory-complaint patients recorded in the *same early batch* as the Healthy volunteers reproduce the
lagged-alpha pattern (r = +0.64, 85 % of the size). They do not reproduce the topography or zero-lag differences
(r = 0.11–0.20).
- **Topography and zero-lag differences** are TDBRAIN acquisition batch. They do not replicate anywhere.
- **The lagged-alpha coherency difference is real:** it appears in two independent clinical MDD datasets and in
  same-batch TDBRAIN patients. It is not depression-specific:
  - TDBRAIN MDD vs other late-batch patients gives r = −0.36;
  - it is absent in medication-free students with high BDI (r = −0.23) or with interview-confirmed current MDD
    (n = 11, r = −0.34).

  A clinical-sample factor such as medication or chronicity is a plausible explanation. None of these datasets can
  test it.

### 2.2 Model fits (`scripts/datasets/new_dataset_model_results.py`)
Model: M5.4 final (BEM, 10 spatial modes, population background), fitted within each dataset: 5 stratified folds,
with the population background taken from that dataset's own training subjects. A second fit per subject used the
other half, for calibration.

| unseen / population average (median) | TDBRAIN dev | MODMA | Mumtaz | ds003478 |
|---|---|---|---|---|
| Wishart deviance (ceiling) | 0.84 (0.69) | 0.86 (0.33) | 0.86 (0.29) | 0.87 (0.49) |
| log spectrum | 0.77 | 0.80 | 0.95 | 0.85 |
| beta shape 13–30 Hz | 0.90 | 0.89 | 0.93 | 0.98 |
| α / β topography | 0.98 / 0.84 | 1.01 / 0.90 | 1.07 / 0.89 | 1.14 / 1.03 |
| **θ topography** | 1.16 | 1.13 | 1.27 | 1.17 |
| θ zero-lag coherence | 1.19 | 1.08 | 1.76 | 1.24 |
| alpha peak frequency error (null) | 0.51 (0.66) Hz | 0.51 (0.69) | 0.61 (0.66) | 0.47 (0.57) |

### 2.3 Parameter group differences
Random-effects group comparison with split-half-calibrated posteriors (`m54_group.py`). Effects in SD units
(± 95 %):

| parameter | MODMA | Mumtaz | ds003478 | TDBRAIN, MDD vs late-batch patients | heterogeneity (3 new) |
|---|---|---|---|---|---|
| inhibitory time scale (b) | −0.01 ± 0.55 | **+1.23 ± 0.51** | +0.48 ± 0.37 | +0.04 ± 0.19 | I² = 0.82, p = 0.004 |
| excitatory time scale (a) | −0.00 ± 0.35 | −0.59 ± 0.51 | −0.49 ± 0.37 | −0.08 ± 0.19 | I² = 0.60 |
| source-background fraction | −0.12 ± 0.31 | −0.89 ± 0.47 | −0.30 ± 0.31 | +0.03 ± 0.18 | I² = 0.73 |
| mean drive | +0.04 ± 0.55 | −0.82 ± 0.51 | −0.12 ± 0.37 | −0.02 ± 0.18 | I² = 0.68 |
| left dorsal-attention gain | −0.22 ± 0.56 | −0.19 ± 0.51 | −0.27 ± 0.37 | −0.02 ± 0.17 | I² = 0 (pooled −0.24, p = 0.08) |

- **MODMA:** no parameter differs (Holm p = 1 for all).
- **Mumtaz:** large differences in several parameters. Its MDD recordings also have about half the total power of its
  controls, with no alpha-frequency difference. That suggests the two groups may have been recorded differently.
  This cannot be verified from the published files, and Mumtaz should not carry the pooled conclusion alone.
- **Pooled:** no parameter shows a consistent depression effect.

---

## 3. What this means and what next

- **The disease question.** Four independent sources (TDBRAIN batch-matched, MODMA, Mumtaz, ds003478) give no
  consistent depression signature, neither in resting-EEG features nor in model parameters. The only robust group
  difference, lagged alpha coherency, separates patients from healthy people but not depressed patients from other
  patients. At these sample sizes (50–120 per dataset), resting EEG plus this model class is not a depression
  biomarker. This matches the literature's replication record for resting-EEG depression markers.
- **The model.** It transfers across systems, but three structural gaps are now clear:
  - no harmonic (non-sinusoidal) alpha;
  - no narrow independent rhythm besides alpha. Beta cannot be fitted by adding Jansen–Rit parameters (§1.3);
  - no frontal-midline theta source: theta topography is worse than the population average in all four datasets.

  A generator with a free resonance width is needed. That could be damped-oscillator sources for beta and
  frontal-midline theta, or the corticothalamic loop, together with a waveform (harmonic) term for posterior beta.
  That is a new model component, not a parameter change.
- **For TMS.** Resting EEG does not constrain individual differences that matter clinically here. The better use of
  the model is to fit TMS-evoked responses (PyTepFit data), where stimulation constrains the dynamics directly.

Reproduce: `scripts/datasets/download_new_datasets.py`, `scripts/datasets/preprocess_new_datasets.py`,
`scripts/datasets/cross_dataset_effects.py`, `scripts/preprocess/beta_origin_check.py`, and the Kaggle kernels
`mddtvb-m54-newds`, `mddtvb-m54-ds003478`, `mddtvb-m54c-fast`, `mddtvb-m54d-somot`
(`scripts/m54/kaggle/make_kernels.py`), then `scripts/datasets/new_dataset_model_results.py`.
