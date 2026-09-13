# M5 subject and group modeling

## Scope

M5 estimates resting-state Jansen–Rit/TVB parameter sets from the preprocessed
TDBRAIN eyes-closed EEG. It does not yet stimulate the model, model plasticity,
or optimize a TMS protocol. The same Schaefer-200 count and distance matrices,
TDBRAIN Table 3 electrode coordinates, analytic single-sphere lead field,
average reference, and optional spherical-spline CSD operator are used for
every empirical and simulated record.

## Empirical design

- Groups: Healthy and MDD.
- Channels: the exact 26-channel TDBRAIN order.
- Input preprocessing: the existing 1–60 Hz, 49–51 Hz notch, artifact-rejected,
  average-referenced `.set` files.
- Observation matching: average reference and the same CSD operator are applied
  idempotently to both sides.
- Individual validation: first temporal half for fitting, second half for an
  unseen-recording check.
- Group validation: a seeded, diagnosis-stratified 80/20 subject split.

The loss never uses DC level or absolute EEG amplitude. A per-subject log
amplitude offset is recorded only as a nuisance observation-gain estimate.

## Features and loss

The feature vector contains:

1. mean log-PSD shape from 2–45 Hz, centered to remove amplitude;
2. alpha peak, alpha-power fraction, spectral entropy, aperiodic exponent,
   theta/beta fractions, alpha prominence, and alpha width;
3. centered log alpha-power topography;
4. magnitude-squared coherence in theta, alpha, and beta bands. The revised
   fit keeps at most 128 channel-pair/band coordinates whose split-half
   reliability exceeds 0.40 in training subjects. Selection never uses the
   diagnosis label.

Feature coordinates are standardized from the training subjects. Block weights
are declared in `configs/m5_fit.toml`; dimensional normalization prevents a
large block from winning merely because it has more entries.

## Parameterization

Every bank member is a complete delayed stochastic TVB simulation. The fitted
dynamic parameters are global coupling, mean external drive, independent
scales for Jansen–Rit's `a` and `b`, neural-noise magnitude, and regional
time-scale log dispersion.

The implementation can represent structural variation in a deliberately
low-dimensional form. Seven gains correspond
to Schaefer's Control, Default, Dorsal Attention, Limbic, Salience/Ventral
Attention, Somatomotor, and Visual networks. Gains have mean one and stay in
`[0.9, 1.1]`. For an edge between regions `i` and `j`, its original weight is
multiplied by the geometric mean of the two endpoint gains. Symmetry, zero-edge
support, tract lengths, and delays are preserved. These gains are disabled in
the revised primary pilot because neither synthetic recovery nor the complete
28-mode ±10% network-pair sensitivity audit supported subject-level estimation.

Parameters are sampled with a scrambled Sobol design. Neural-noise magnitude is
sampled logarithmically. All candidates share stochastic seeds (common random
numbers), reducing Monte Carlo noise in comparisons.

## Hierarchy and outputs

For each diagnosis, a descriptive group MAP candidate minimizes feature
distance to the training-subject group mean plus a weak reference prior. These
group fits do not select individual parameters. A separate pooled candidate is
learned from all training subjects without diagnosis labels, and each subject
selects a MAP candidate from their fitting half with weak shrinkage toward that
pooled candidate. Both fit-half and unseen-half costs are saved. The legacy
diagnosis-conditioned prior remains an explicit configuration option but is not
the primary analysis.

The original audit is below `outputs/m5`. The revised 64-design-point pilot is
below `outputs/m5_v2` and is configured by `configs/m5_fit_v2.toml`. Its
empirical arrays are an exact copy of the first extraction because preprocessing
and raw feature extraction are unchanged; only the training-only feature
transform and simulation bank differ.

Outputs include:

- `empirical/empirical_features.npz` and
  `empirical_validation_features.npz`;
- `empirical/subjects.csv` and a complete failure audit;
- `bank/simulation_bank.npz`, candidate parameters, and simulation failures;
- `fit/group_fits.csv` and `fit/subject_fits.csv`;
- fitted/reference cost ratios, weighted standardized RMSE, generalization gaps,
  and PSD/topography/coherence agreement for both temporal halves;
- `fit/m5_eeg_examples.png` and its representative-subject audit CSV;
- group averages of independently fitted simulations, individual validation,
  subject-parameter distributions, and a numerical group summary of subject
  parameters;
- one fitted group connectome per diagnosis;
- the training-only feature transformer, data split, JSON provenance/limitations,
  synthetic-recovery diagnostics, and QC figure.

## Current audit result

The revised pilot is a useful improvement but is not a mechanistically accepted
M5 model. Relative to the original bank, median synthetic normalized parameter
RMSE improved from 0.661 to 0.360. On the held-out subjects, the median unseen
cost improvement was 17.9% for Healthy and 16.5% for the MDD-indication group;
91.4% and 93.3% respectively improved over the declared reference. Median
unseen PSD correlation was 0.923/0.911 and full-coherence correlation was
0.527/0.454.

Those subject-level results do not imply that the fitted simulations preserve
the empirical between-group mechanism. Correlations between empirical unseen-
half and fitted-simulation Healthy–MDD effect vectors were 0.50 for PSD, 0.89
for the compact mechanistic spectrum, 0.02 for alpha topography, 0.00 for alpha
coherence, and -0.06 for beta coherence. The exploratory quality gate therefore
fails. `feature_audit/m5_fitted_group_effects.png` is the primary group-effect
plot; `feature_audit/feature_audit_summary.json` records the machine-readable
gate and thresholds.

A separate mean-preserving ±10% regional-physiology audit found that network-
local Jansen–Rit time-scale modes were much more observable than structural
weight blocks or local drive/noise. The Dorsal Attention time-scale mode had a
0.61 cosine with the empirical alpha-topography contrast. Its alpha/beta
coherence alignment remained weak. It is therefore a candidate for a future
training-split-derived, held-out-tested spatial parameterization, not an
accepted disease parameter.

## Interpretation limits

This is parameter estimation against a shared template connectome, not recovery
of a person's anatomical tractography. The bounded connectivity gains are an
effective structural hypothesis and can trade off against dynamics. Discrete
bank resolution and stochastic duration limit precision. Group differences
must not be declared when both groups select the same regime, and individual
parameters should be trusted only when fit improvement survives the held-out
temporal half and synthetic recovery is adequate.

The dataset label used here is an MDD *indication* cohort, not a uniformly
confirmed diagnostic cohort. In the available participant table, formal
diagnostic status is missing for nearly all indication rows. Results must use
the term `MDD-indication` unless a separately verified diagnosis table is
introduced.

No stimulation-target or plasticity claim may use the current M5 estimates
while the group-effect gate is failed. The next model-development step is to
test a small number of sensitivity-qualified network-local time-scale modes and,
if connectivity still fails, a multi-frequency/two-subpopulation JR extension.
Only after synthetic recovery, unseen-half improvement, and held-out group-
effect preservation all pass should the bank be expanded or used for TMS.

The subject shrinkage strength is intentionally weak. It was selected from the
training-subject validation halves; the separate subject holdout remains the
unbiased check reported in `fit_summary.json`.
