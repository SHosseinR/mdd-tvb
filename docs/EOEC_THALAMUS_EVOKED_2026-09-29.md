# Eyes open + eyes closed, a thalamic loop, and evoked responses (2026-09-29)

This report covers the three steps chosen after the literature review
(`docs/LITERATURE_REST_AND_EVOKED_2026-09-27.md`):

1. fit eyes-closed and eyes-open recordings together, and check the parameters
   against age;
2. add a thalamus (corticothalamic loop) to the model;
3. model evoked responses (ds003474) from each person's resting-state fit.

## Summary

| Step | Question | Answer |
|---|---|---|
| 1a | Does the eyes-closed/eyes-open **contrast** escape TDBRAIN's recording-era batch effect? | **Mostly** at the data level. The batch effect on alpha falls from +0.34 SD (p = 0.006) to +0.11 SD (n.s.). How closely the depression pattern resembles the batch pattern falls from r = 0.91 to 0.55 (alpha) and from 0.89 to 0.0 (beta). |
| 1b | Do the model parameters track **age** (positive control)? | **Yes, within TDBRAIN.** 9 parameters trend with age (Holm). They predict age in unseen people as well as plain spectral features do (r = 0.41 vs 0.42). They don't replicate in MODMA (n = 52, narrow ages). |
| 1c | Which parameter explains eye opening? | **The drive into the visual network.** It improves 75% of subjects over "no change" (100-subject comparison of 9 options); freeing everything is no better. |
| 1d | Does that within-person change relate to depression? | **Same direction in all three datasets.** Batch-free pooled: −0.36 SD (p = 0.013) and −0.31 SD (p = 0.021). **But** in TDBRAIN it is not depression-specific (MDD vs other late-era patients: +0.03 and −0.10, n.s.). It declines with age as expected. Split-half ICC 0.5–0.7. |
| 2 | Does a thalamic loop fix beta? | **No real beta peak** in any of 1,008 settings. Within TDBRAIN the fit still improves: 61–75% of subjects better, whole spectrum, beta shape a little (0.90 → 0.87–0.88), alpha frequency. It does not improve the other EEG systems, and it costs spatial alpha coherence. The per-person **loop delay** tracks alpha frequency **in all 7 cohorts** (ρ −0.19 to −0.56) and age (+0.16 SD per decade), with no batch effect and no consistent depression effect. |
| 3 | Does a person's resting network predict their **evoked** responses? | **No.** Own network ≈ other people's ≈ population (R² 0.30 vs 0.29; 0.18 vs 0.18; p ≥ 0.36). The reward positivity is clear in the data (1.45 µV, t = 4.2), slightly smaller with high BDI (d = −0.21, n.s.). The model's frontal input does not capture the reward difference. |

One-line reading:
- The within-person contrast was the right idea for removing batch effects.
- The thalamus makes a better and more interpretable resting model, but not a beta generator.
- The resting parameters carry no individual information about evoked responses.
- The depression signal is the familiar one: present in single-system data, not specific among patients.

---

## 1. Eyes closed + eyes open

### 1a. Premise check (data level; `scripts/m54/eoec_batch_check.py`)

Eyes-open data were cleaned for every cohort. Usable eyes-open recordings: dev 217,
rTMS 147, late-era controls 182, SMC 84, ds003478 111, Mumtaz 58. MODMA has eyes
closed only.

Setup: 648 TDBRAIN subjects with both recordings. Same decomposition as the batch
analysis: early era + clinical + depression + age + sex.

| Measure (z units) | Early-era effect | Depression effect | Age (per year) | Depression pattern vs batch pattern (r) |
|---|---|---|---|---|
| Absolute alpha power (EC) | **+0.34, p = 0.006** | +0.05 | −0.013*** | 0.91 |
| Alpha reactivity log(EC/EO) | +0.11, p = 0.39 | −0.10, p = 0.28 | −0.008** | 0.55 |
| Absolute beta power | **+0.29, p = 0.02** | +0.16 | n.s. | 0.89 |
| Beta reactivity | +0.18, p = 0.16 | −0.10, p = 0.27 | −0.017*** | −0.005 |
| Whole spectrum (pattern) | — | — | — | 0.74 → 0.53 (reactivity) |

The contrast removes about half or more of the batch effect. It carries a small
depression effect, and a clear age effect (reactivity declines with age).

### 1b. Age as a positive control (`scripts/m54/age_control.py`; existing eyes-closed fits)

- **TDBRAIN adults (n = 684):** 9 of 25 parameters have Holm-significant age slopes.
  For example, visual-network drive falls by 0.16–0.20 SD per decade, and visual
  time constants rise by 0.18 SD per decade.
- **Predicting age in the unseen rTMS cohort (ridge regression):**
  - parameters: r = 0.41 (R² 0.15);
  - plain spectral features: r = 0.42 (R² 0.12);
  - both: r = 0.48 (R² 0.18).
- **MODMA (n = 52, ages 16–56):** none of the 9 replicate. Signs agree only at chance
  (4/9), and the slope correlation is 0.28. That test has little power.

So the parameters carry real biological information (age), about as much as the
spectrum itself.

### 1c. Joint fitting (`scripts/m54/m54_joint.py`)

The joint fit works as follows:
- One parameter set explains both recordings of a person.
- Only the parameters named in the **change set** may differ with eyes open.
- Recording nuisances (noise levels, background share) are separate per condition.
- **One overall scale** is shared, so the eyes-open power drop must come from the
  model.

Comparison of change sets (100 TDBRAIN subjects; lower deviance is better):

| Change set | Total unseen deviance vs "none" | Subjects better | Mean change (t) |
|---|---|---|---|
| free eyes-open level | −0.38% | 62% (p = 4×10⁻⁴) | −0.18 (−7.4) |
| **visual-network drive L/R** | **−0.35%** | **75%** (p < 10⁻⁴) | −0.02 / −0.06 (n.s.; individual) |
| all 19 neural parameters | −0.39% | — | — |
| cortico-cortical coupling | −0.29% | 56% (n.s.) | +0.17 (3.1) |
| shared alpha drive | +0.07% | 65% | −0.21 (−4.7) |
| mean drive μ | +0.12% | 66% | +0.10 (1.4), not determined |
| cortical time constants a / b | +0.6% / +1.9% | 41% / 46% | determined, but hurt prediction |

- The **visual-network drive** was taken forward: 2 parameters, as good as freeing 19.
- The model under-predicts the eyes-open power drop (log ratio −0.25 vs −0.55 in the
  data).
- Sharing parameters costs the eyes-closed fit a little (median ratio 0.841 vs 0.833
  when eyes closed is fitted alone).

### 1d. Production fits and depression (`scripts/m54/eoec_group.py`)

Fit quality (median unseen deviance ratio, eyes closed / eyes open):

| Cohort | n | EC | EO |
|---|---|---|---|
| TDBRAIN dev | 213 | 0.844 | 0.880 |
| rTMS | 145 | 0.853 | 0.905 |
| late-era controls | 177 | 0.868 | 0.868 |
| SMC | 84 | 0.868 | 0.841 |
| ds003478 | 108 | 0.884 | 0.905 |
| Mumtaz | 53 | 0.871 | 0.966 |

Eyes-open change in visual drive, depression contrast:

| Contrast | Left | Right | ICC (split halves) |
|---|---|---|---|
| TDBRAIN Healthy vs MDD | −0.29 SD, p = 0.025 | **−0.41 SD, p = 0.001** (Holm 0.003) | 0.67 / 0.71 |
| ds003478 high vs low BDI | **−0.44, p = 0.012** (Holm 0.025) | **−0.37, p = 0.032** | 0.70 / 0.47 |
| Mumtaz MDD vs HC | −0.19, p = 0.45 | −0.22, p = 0.32 | 0.49 / 0.48 |
| **Batch-free pooled** (ds003478 + Mumtaz) | **−0.36 [−0.64, −0.08], p = 0.013** | **−0.31 [−0.58, −0.05], p = 0.021** | |
| TDBRAIN, depressed vs other late-era patients (n = 421) | +0.03, p = 0.73 | −0.10, p = 0.30 | |

- **Age:** the change becomes less negative with age (+0.12 SD per decade,
  p < 0.001), matching the data-level decline of reactivity. This is a positive
  control.
- **TDBRAIN by group and era:**
  - early-era Healthy +0.06 / +0.05 and SMC +0.18 / +0.12;
  - late-era groups all lower: MDD −0.08 / −0.16, OCD −0.18 / −0.21,
    insomnia −0.20 / −0.27, late Healthy (n = 12) −0.26 / −0.11.
- The early-era term in the full decomposition is +0.18 / +0.08 (n.s.). So the
  batch share is smaller than for the old parameters, but TDBRAIN still cannot
  isolate a depression effect.

**Reading:**
- People with depressive symptoms reduce visual-network drive more when they open
  their eyes. The direction is consistent across three datasets, and the effect is
  significant in the single-system data.
- Among patients the effect is not specific to depression.
- It is a candidate marker for a pre-registered test on an independent dataset
  (e.g. EMBARC), not a finding yet.

## 2. Thalamic loop (`src/mdd_tvb/linear_jax.py`: `thalamic_loop`, `thalamic_certificate`)

Every region gets its own linearised relay + reticular loop. This adds one delayed
self-connection
`θ(ω) = g e^(−iωt0) L(1 − γL)/(1 + κL²)`
to the network kernel. The mean feedback is absorbed into the mean drive.

Tests confirm three things (`tests/test_thalamus.py`):
- the formula equals an explicit relay/reticular solve;
- a vanishing loop reproduces the old model exactly;
- an unstable loop is flagged.

Population fit (dev, 252 subjects):
- loop gain 49, reticular fraction 0.18, **delay 87 ms**, intrathalamic 0.14;
- cortico-cortical coupling falls from 22 to 2.3, because the loop takes over;
- mean drive 0.315, which is **off its upper bound** (the old M5.4 problem 5);
- the population likelihood itself is about unchanged (0.1673 vs 0.1665).

Subject fits:

| | T1: loop gains free | T2: gains + delay free, cortical time constants fixed | M5.4 |
|---|---|---|---|
| median unseen deviance ratio | **0.827** | 0.829 | 0.840 |
| subjects better than M5.4 | **75%** (p = 4×10⁻²¹) | 61% (p = 1×10⁻⁵) | — |
| total deviance vs M5.4 | −2.4% | −1.5% | — |
| whole spectrum, subjects better | 72% | 59% | — |
| beta shape (median ratio) | 0.878 (60% better) | 0.870 (60% better) | 0.900 |
| alpha peak frequency | 0.725 | **0.625** | 0.771 |
| zero-lag alpha coherence | worse (35% better) | worse (29%) | 1.116 |
| synthetic recovery | all 26 parameters r ≥ 0.5 (loop gain 0.80, reticular 0.63) | all 25 (delay 0.82, gain 0.90) | — |

What the loop can and cannot do:
- **No beta peak.** In a scan of 1,008 loop settings on the population state, the
  largest beta excess was 1.03. A purely excitatory loop mainly amplifies slow
  activity and quickly breaks stability.
- **It weakens alpha coherence.** Separate loops per region replace much of the
  cortico-cortical coupling. A shared thalamic nucleus would be the natural next
  step for that.
- **Delay positive control (T2):**
  - delay vs individual alpha frequency: ρ = −0.47 (p = 3×10⁻¹⁵);
  - delay vs age in adults: ρ = +0.32 (p = 2×10⁻⁷), +0.18 SD per decade after group
    and batch;
  - MDD vs Healthy: +0.33 SD, n.s.
- **Certification caveat.** The stability bound had been checked only at 2–40 Hz.
  Over 0–150 Hz it certifies:
  - M5.4: 90%;
  - T1: 85%;
  - T2: 86%.

  On subjects certified under both models the conclusions hold. The wide-band check
  is now the default whenever the loop is on.

### 2b. T2 on every cohort (`scripts/m54/thal_replication.py`; wide-band certification, 100% certified)

| Cohort | n | Median ratio T2 / M5.4 | Better than M5.4 | Delay (ms) | Delay vs IAF (ρ) | Delay vs age (ρ) |
|---|---|---|---|---|---|---|
| TDBRAIN dev | 252 | 0.829 / 0.840 | 62% (p = 2×10⁻⁶) | 88 | **−0.45** | **+0.31** |
| rTMS | 162 | 0.826 / 0.834 | 61% (p = 6×10⁻⁶) | 89 | **−0.48** | **+0.30** |
| late controls | 187 | 0.846 / 0.857 | 62% (p = 9×10⁻⁵) | 87 | **−0.31** | **+0.21** |
| SMC | 85 | 0.838 / 0.853 | 62% (p = 0.06) | 91 | −0.19 (p = 0.09) | +0.20 (p = 0.06) |
| MODMA | 52 | 0.857 / 0.860 | 46% (n.s.) | 81 | **−0.56** | −0.02 (ages 16–56) |
| Mumtaz | 58 | 0.879 / 0.858 | 45% (n.s.) | 86 | **−0.37** | — |
| ds003478 | 117 | 0.870 / 0.869 | 53% (n.s.) | 84 | **−0.49** | — (students) |

- **The delay–alpha frequency relation replicates in all 7 cohorts** (6 significant).
  So the delay is a physiologically meaningful, portable parameter.
- **The fit gain does not transfer.** It holds within TDBRAIN's recording system but
  not on the other systems, where T2 (cortical time constants fixed at TDBRAIN's
  population values) is no better than M5.4.
- **TDBRAIN decomposition of the delay:**
  - batch −0.05, clinical +0.03, depression +0.04 (all n.s.);
  - age +0.16 SD per decade (p = 10⁻¹⁰).
- **Loop gain** carries batch (+0.29, p = 0.02) and clinical status (+0.36,
  p = 0.007), but no depression effect.
- **Depression contrasts on the delay:** MODMA MDD vs HC +0.60 SD (p = 0.03),
  Mumtaz +0.13, ds003478 −0.14. They are inconsistent, so there is no depression
  signal.

## 3. Evoked responses (ds003474; `scripts/datasets/ds003474_erps.py`, `scripts/m54/m54_evoked.py`)

**Data:**
- ds003474: probabilistic selection task, same 122 people as ds003478.
- Cleaning: same pipeline as rest.
- Epochs from −0.2 to 0.8 s, amplitude rejection (100 µV / 150 µV peak-to-peak).
- 106 subjects have all conditions. Median trials: 298 stimulus, 72 correct,
  52 incorrect feedback.
- **Reward positivity** (correct − incorrect, FCz/Cz, 250–350 ms):
  - overall 1.45 µV, t = 4.2;
  - high BDI 1.01 µV vs low BDI 1.78 µV, d = −0.21, p = 0.28.

  This is Cavanagh's direction, but not significant here.

**Model.** The evoked potential is the impulse response of the person's own linear
network, i.e. the same transfer function that shapes their resting spectrum.

- **First finding:** with inputs into fixed network sets, the model gives almost
  nothing (R² ≈ 0.01–0.04). The resting-fitted network strongly amplifies 10 Hz
  (transfer 72 vs 12 at 5 and 20 Hz), so a brief input rings at alpha. Real ERPs are
  slower broadband deflections.
- **Revised model:**
  - each input's spatial pattern is free over the 14 network × hemisphere groups;
  - timing comes from the network;
  - latencies and widths are grid-searched;
  - fitted on odd trials, scored on even trials.

Results (74 subjects with a resting fit and ≥ 16 trials per feedback half):

| R² on even trials | Own network | Other people's | Population | Odd average as predictor |
|---|---|---|---|---|
| stimulus ERP | 0.295 | 0.291 | 0.288 | 0.79 |
| feedback ERP | 0.180 | 0.181 | 0.182 | 0.69 |
| own better than others / population | 46% / 49% (p ≥ 0.36) | | | |

- The resting parameters carry **no individual information** about evoked responses
  in this model.
- The medial-frontal input is used (amplitude ≈ 0.32, latency ≈ 175 ms), but it is
  the same for correct and incorrect feedback. So the model's "reward input" is
  ≈ 0 (t = −0.15) and cannot be used for the depression question.

Why evoked responses did not help here:
- The resting fit identifies a strongly alpha-resonant transfer function whose
  individual differences do not show in ERPs.
- ERP shape is set mostly by the inputs, which are free per person.
- For an evoked paradigm to constrain the network, the input must be known and
  simple: steady-state (SSVEP / ASSR) or TMS pulses, not cognitive feedback
  responses.

## Files

**Code:**
- `src/mdd_tvb/linear_jax.py`: loop, certificate.
- `scripts/m54/m54_core.py`: T1/T2, wide-band certification.
- `scripts/m54/m54_joint.py`, `joint_compare.py`, `eoec_group.py`, `eoec_batch_check.py`,
  `age_control.py`, `thal_scan.py`, `thal_certify.py`, `m54_evoked.py`.
- `scripts/datasets/ds003474_erps.py`.
- Readers for eyes-open data and ds003474 in `src/mdd_tvb/new_datasets.py`.

**Results:**
- `outputs/m54_eoec/`: batch check, age control, joint comparison, group analyses.
- `outputs/m54e_joint*/`: Kaggle joint fits.
- `outputs/m54t_thal*/`: thalamus fits, scan, recovery, evaluation.
- `outputs/erp/ds003474/`, `outputs/m54_evoked/`.

**Kaggle:** kernels `mddtvb-m54t-thal`, `-thal2`, `mddtvb-m54e-joint1/2/3a/3b`; dataset
`shahmadi/mdd-tvb-eo`.

## What this changes, and next steps

1. **Keep the within-person contrast.**
   - It is the most batch-robust quantity we have.
   - Its depression effect is consistent in direction across three datasets, and
     significant in the batch-free pooled data.
   - Next: a pre-registered test on independent single-system data. EMBARC has eyes
     open and eyes closed for MDD and controls. The pre-registration should state
     the parameter (eyes-open change in visual-network drive), the direction
     (more negative with depression) and the analysis.
2. **Keep the thalamic delay (T2)** as an interpretable parameter.
   - It replicates everywhere and tracks alpha frequency and age.
   - To improve the model further:
     - let regions share thalamic nuclei (to restore alpha coherence);
     - free the cortical time constants per dataset (for transfer to other systems).
   - Beta needs a non-linear (harmonic) description; a linear loop cannot make it.
3. **Drop cognitive ERPs as a way to constrain the network.**
   - Individual resting dynamics do not show in them.
   - Evoked data would only help with a known, simple input (steady-state
     stimulation or TMS).
