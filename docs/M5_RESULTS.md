# M5 result audit

This file records the original 128-member run and the revised 64-member M5
pilot. They are quality-control results, not diagnostic claims.

## Current status

The revised pilot improves unseen subject-level fit and dynamic-parameter
recoverability, but it **fails** the mechanistic group-effect gate. It must not
yet supply parameters for stimulation targeting.

| Metric | Original M5 | Revised pilot |
| --- | ---: | ---: |
| Synthetic median normalized parameter RMSE | 0.661 | 0.360 |
| Healthy held-out median unseen improvement | 15.8% | 17.9% |
| MDD-indication held-out median unseen improvement | 15.7% | 16.5% |
| Healthy held-out median unseen PSD correlation | 0.901 | 0.923 |
| MDD-indication held-out median unseen PSD correlation | 0.914 | 0.911 |
| Healthy held-out median unseen coherence correlation | 0.461 | 0.527 |
| MDD-indication held-out median unseen coherence correlation | 0.423 | 0.454 |

For the revised pilot, empirical-unseen versus fitted-simulation group-effect
vector correlations are 0.50 (PSD), 0.89 (mechanistic spectrum), 0.02 (alpha
topography), 0.00 (alpha coherence), and -0.06 (beta coherence). The dedicated
plot is `outputs/m5_v2/feature_audit/m5_fitted_group_effects.png`. This is the
decisive reason the fit is not yet accepted, despite better individual metrics.

The revised objective uses general and mechanistic spectral features, alpha
topography, and 128 split-half-reliable theta/alpha/beta coherence coordinates.
Diagnosis labels are not used to select those coordinates or fit subjects.

All 28 within/between-Yeo structural block modes were tested at ±10%. Their EEG
sensitivity and alignment were too weak for individual estimation, so the
revised pilot fixes the structural matrix. A second audit found much stronger
sensitivity to mean-preserving network-local time scales. The Dorsal Attention
time-scale mode had a 0.61 cosine with the empirical alpha-topography contrast,
but alpha/beta coherence alignment remained weak. This supports testing a small
regularized spatial-physiology parameterization next; it does not yet establish
an MDD mechanism.

## Original M5 data and coverage

- 327 recordings: 176 Healthy and 151 MDD; zero extraction failures.
- Group split: 262 training subjects and 65 held-out subjects.
- Each recording was split temporally into fitting and validation halves.
- Simulation-bank IAF: 8–11 Hz; empirical fit-half p5/median/p95: 8/9.5/11.5 Hz.
- Simulation-bank spectral entropy: 0.676–0.861; empirical p5/median/p95:
  0.691/0.846/0.972.
- Simulation-bank alpha fraction: 0.143–0.745; empirical p5/median/p95:
  0.109/0.368/0.637.

The bank covers the center of the empirical distribution but not the highest
IAF and high-entropy tails.

## Original M5 group fit

Both diagnosis groups selected candidate 99. Its parameters are:

| Parameter | Value |
| --- | ---: |
| Global coupling | 7.307866 |
| Mean drive `mu` | 0.252617 |
| `a` scale | 0.922287 |
| `b` scale | 1.086319 |
| Neural-noise `nsig` | 0.013549 |
| Regional time log-SD | 0.069480 |
| Simulated alpha peak | 9.0 Hz |

Endpoint gains are Control 1.0159, Default 0.9032, Dorsal Attention 1.0170,
Limbic 1.0098, Salience/Ventral Attention 1.0318, Somatomotor 1.0338, and
Visual 0.9885. The identical selection means this bank does **not** resolve a
diagnosis-specific group parameter difference. It would be invalid to force one.

Group feature distance improves from 1.121 to 0.810 for Healthy training and
from 1.138 to 0.824 for MDD training. On held-out subjects it improves from
1.190 to 0.894 and from 1.104 to 0.872, respectively.

The old “improvement fraction” is only `1 - fitted_cost/reference_cost`. It is a
relative comparison to candidate 0, not an absolute goodness-of-fit statistic.
The revised CSV and QC therefore lead with `fitted_cost/reference_cost` on both
the fitting and unseen halves and also report PSD, alpha-topography, and
coherence correlations. Ratios below one beat the reference; the distance from
zero is not interpretable as percent EEG variance explained.
The tables also include weighted standardized feature RMSE: zero would be exact
feature agreement, while one is approximately one training-standard-deviation
of mismatch after the declared block weighting.

## Original M5 individual fit and validation

With subject shrinkage 0.01:

| Group | Distinct candidates | Group-candidate fraction | Median unseen-half improvement | Positive unseen-half fraction |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 28 | 43.2% | 17.1% | 96.6% |
| MDD | 24 | 38.4% | 15.9% | 93.4% |

Restricting evaluation to the completely held-out subjects, the median
unseen-half improvement is 15.8% for Healthy and 15.7% for MDD; improvement is
positive in 91.4% and 90.0%. Fitted IAF correlates with fitting-half IAF at
`r=0.70` and with unseen-half IAF at `r=0.49`.

## Reading the revised figures

In `m5_fit_summary.png`:

- the top row compares fitting-half and unseen-half empirical PSD averages with
  the average of each group's independently selected subject simulations;
- the bottom row makes the same comparison for alpha topography;
- Healthy and MDD simulated averages are drawn separately. They are close but
  not identical because the individual candidate distributions differ.

`m5_individual_validation.png` contains the per-subject fitting-versus-unseen
cost-ratio scatter, selected-model versus unseen IAF, and median unseen-half
PSD/topography/coherence correlations.

`m5_subject_parameter_distributions.png` shows the distributions of independently
selected dynamics and endpoint gains. Exact numerical group summaries are in
`group_summary_of_subject_parameters.csv`. These descriptive differences are
not corrected inferential group effects.

`m5_eeg_examples.png` shows one representative held-out subject per group. Each
column contains fitting-half empirical EEG, unseen-half empirical EEG, and a
fresh rerun of that subject's selected TVB candidate. Fz, Cz, Pz, and Oz are
shown after a 1 Hz display high-pass and per-channel z-scoring; phase matching is
neither expected nor optimized for stochastic resting EEG.

The procedure is finite-bank MAP selection, not continuous gradient fitting.
It is objective-based parameter selection, but its resolution is limited to 128
simulated candidates. A later claim of precise parameter estimation requires
iterative local refinement or simulation-based posterior inference.

## What remains uncertain

The simulated high-frequency PSD tail remains too steep and alpha topography is
not fully reproduced. Leave-one-design-point-out synthetic recovery has median
normalized parameter RMSE 0.661; individual network gains are especially weakly
recoverable. Consequently, fitted connectivity gains are disabled in the
revised pilot and must not be carried into stimulation experiments. Any future
weight parameter must first pass synthetic recovery, EEG sensitivity,
unseen-half improvement, and held-out group-effect preservation.
