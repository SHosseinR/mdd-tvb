# M5 resting-state spectral fitting

## Scope and scientific status

M5 fits one whole-brain TVB model to each subject's resting-state EEG. It does
not train a diagnosis classifier and it does not fit one Healthy model and one
MDD model. Group summaries are computed only after diagnosis-blind individual
fits, as a second-level descriptive analysis.

The original eight-candidate pilot was genuine underfitting: its prediction of
the unseen half of each subject was worse than a pooled empirical null. The
completed 2,048-candidate M5.1 production run beats that null for the total
objective, autospectra, and lagged coherency. It still narrowly misses the
alpha-topography fit gate and fails to preserve the held-out lagged-coherency
Healthy--MDD effect. Therefore it is not yet an accepted mechanistic MDD model
and must not be used for TMS target or protocol claims.

## Data split and leakage controls

The 327 TDBRAIN recordings comprise 176 Healthy and 151 MDD recordings. Every
recording is divided chronologically into a fitting half and an unseen half.
The fitting half is divided again into two quarters for temporal-reliability
selection.

A fixed, stratified 20% subject holdout contains 65 subjects. The remaining 262
training subjects are used to learn the label-blind sensor basis, choose
reliable objective coordinates, audit objective resolution, derive two
label-blind spatial modes, and select one global posterior temperature. The
second halves of the 65 subject holdouts are untouched until final evaluation.
Their first halves are still used to fit those subjects individually; this is
the intended prospective-within-subject validation design.

Diagnosis labels never enter feature reduction, candidate selection, posterior
weights, or subject-parameter estimation. Labels are used afterward only to
ask whether independently fitted models preserve empirical Healthy--MDD
effects.

## Neural and observation model

Each Schaefer-200 parcel contains two locally coupled Jansen--Rit generators:
an alpha generator and a faster generator. Both share the same delayed TVB
structural network. Separate excitatory and inhibitory inverse-time-constant
scales are retained because tying them shifted the reference regime into theta.

The regional pyramidal PSP is projected to the published 26-channel TDBRAIN
montage with the documented analytic spherical lead field. This is a common
template forward model, not a personalized BEM/FEM solution. Per-channel DC is
excluded and absolute sensor gain is treated as a nuisance; the empirical EEG
does not need its removed DC offset restored.

The subject objective contains three blocks:

1. reliable coordinates from channel-resolved 2--40 Hz periodic residuals and
   aperiodic exponents;
2. reliable lagged-coherency coordinates; and
3. a direct centered 8--13 Hz scalp-topography block.

The declared production weights are 0.45, 0.45, and 0.10. The topography weight
was selected as a training-only Pareto point: compared with zero topography
weight it improves internal unseen alpha-topography error while retaining a
total internal unseen cost below the pooled empirical null. The subject-holdout
set was not used for this choice.

Diagonal colored observation noise is marginalized over a fixed fraction and
exponent grid. It is an observation/unmodelled-background nuisance, never a
disease parameter. A boundary optimum is treated as model-mismatch evidence.

## Parameterization

The bank stores 17 parameter columns:

- nine global dynamics parameters: coupling, conduction speed, mean drive,
  excitatory and inhibitory time scales, fast-generator ratio and fraction,
  neural-noise magnitude, and neural-noise correlation time;
- six structured spatial-physiology parameters: dorsal-attention and visual
  time-scale contrasts, default and visual noise contrasts, and two fixed
  seven-network noise modes derived by label-blind PCA from training-subject
  alpha-topography variation; and
- two symmetric, graph-support-preserving network-pair connectome contrasts,
  bounded to +/-10% and constrained to preserve total weight during
  calibration, but fixed at zero in production.

Production fits the first 15 columns. Its broad-plus-adaptive design contains
2,048 unique candidates, three stochastic seeds, and 60 analysed seconds after
a two-second transient.

The connectome modes were investigated because disorder-related coupling may
be important. Two diagnosis-blind calibration banks failed their declared
synthetic-recovery gate, even after broader mode design. Production therefore
fixes both at zero: resting EEG alone did not support estimating them. This
does not imply that structural connectivity is irrelevant to MDD; it means
these particular subject-level structural degrees of freedom were not
identifiable with this observation model and data.

## Subject inference and regularization

For each subject, every neural candidate is combined with the declared
observation-nuisance grid. The squared standardized feature error is augmented
with quadratic Gaussian penalties on range-normalized parameters. Spatial
physiology receives additional shrinkage. During structural calibration the
two connectome modes received the strongest shrinkage; in production they are
fixed rather than regularized. The prior penalty is added to the data objective
before temperature scaling:

`weight(state) proportional to exp[-(data_cost + prior_cost)/(2 temperature)]`.

This is the requested Gaussian regularization idea: it discourages implausible
or boundary solutions and reduces overfitting, but it cannot make an
insensitive parameter identifiable. Different biological parameter vectors can
produce nearly identical EEG; in that case EEG constrains only combinations of
parameters, not every parameter independently.

One global temperature is selected using only the unseen halves of training
subjects. Posterior means, 5--95% intervals, MAP states, candidate effective
sample size, and posterior-predictive CSDs are saved for every subject. The
posterior is a finite-bank kernel approximation, not exact Bayesian inference
or amortized simulation-based inference.

## Validation and acceptance

The explicit null is the pooled fitting-half empirical feature vector from
training subjects. A ratio below one means the subject-specific TVB prediction
is closer to unseen EEG than that pooled empirical null. The empirical
first-half-to-second-half prediction is also reported as a measurement-noise
floor, not as a model competitor.

`fit/ACCEPTANCE.md` and `fit/acceptance_report.json` require all of the
following:

- held-out unseen total, autospectral, selected-connectivity, and alpha-topography
  costs below the pooled null;
- a majority of held-out subjects beating the null;
- preservation of held-out channel-by-frequency power, alpha-topography, and
  selected-connectivity Healthy--MDD effect directions;
- an interior population observation-nuisance optimum;
- at least three recoverable parameters in stochastic synthetic recovery; and
- structural modes either recoverable in synthetic data or explicitly fixed
  from diagnosis-blind calibration.

Failure of any required gate keeps stimulation optimization blocked. More
simulation time can reduce Monte Carlo error; it cannot by itself repair a
forward-model or parameter-sensitivity failure.

## Accelerated simulation

TVB is the scientific reference. The JAX backend mirrors its delayed
dual-generator equations, stochastic Heun integration, colored neural noise,
temporal monitor, candidate-specific edge weights, and analytic EEG projection.
Deterministic TVB/JAX channel correlation was 0.9972 and stochastic normalized
log-PSD correlation was 0.9962 on the real connectome. See `JAX_BACKEND.md` for
the full validation and timing record.

The private Kaggle notebook is `shahmadi/tvbgpu`. It generates only the
simulation bank; all empirical fitting and held-out evaluation run locally.
VBI is not a runtime dependency: its abstractions did not replace the custom
dual-Jansen--Rit simulator, observation model, or subject objective, so adding
it would have increased integration surface without accelerating most work.

## Production outcome (2026-09-18)

Kaggle notebook version 21 ran repository commit `a302fdf`. The final bank
contains 2,048 candidates, three stochastic replicates, 39 frequency bins, and
26 x 26 complex cross-spectra. The GPU simulation took 11,291.2 seconds (about
3 h 8 min). All outputs were finite, there were zero failed simulations, both
structural columns were exactly zero, and the downloaded bank was validated
against its declared 1,471,969,366-byte size and schema before fitting.

On the 65 held-out subjects:

- the median unseen total-cost ratio to the pooled empirical null was 0.759;
- 72.3% of subjects beat that null;
- autospectral, lagged-coherency, and alpha-topography ratios were 0.438,
  0.997, and 1.046 respectively; and
- held-out Healthy--MDD effect correlations were 0.447 for channel-by-frequency
  power, 0.332 for alpha topography, and 0.048 for lagged coherency.

All 15 active parameters passed the declared synthetic-recovery correlation
threshold (range 0.638--0.955). The structural columns were inactive by the
calibration decision. The result is nevertheless `not_accepted`: nine of
eleven gates pass. The remaining failures are scientific validation failures,
not simulation crashes or non-finite results, so the fit is not sufficient
evidence for individualized TMS targeting.

## Commands and outputs

```powershell
& .\.conda\python.exe scripts\run_m5_spectral.py `
  --config configs\m5_spectral.toml --stages extract

& .\.conda\python.exe scripts\run_m5_spectral.py `
  --config configs\m5_spectral.toml --stages fit
```

The bank can be generated with TVB or with `--backend jax --batch-size 64`.
Stages are independently resumable.

Training-only design audits are reproducible with
`scripts/audit_spectral_objectives.py`,
`scripts/audit_spectral_topography.py`, and
`scripts/audit_network_topography.py`. They must not be rerun repeatedly to
select whatever setting performs best on the subject holdout.

Primary outputs are:

- `fit/subject_posteriors.csv` for per-subject fit, unseen, null, posterior,
  and parameter summaries;
- `fit/m5_spectral_validation.png` for subject-holdout fit/unseen validation;
- `fit/m5_spectral_group_effects.png` for held-out Healthy--MDD effect
  preservation;
- `fit/m5_heldout_parameter_effects.png` and its CSV for descriptive
  posterior-parameter effects, visibly guarded by synthetic recoverability;
- `fit/m5_eeg_examples.png` for representative empirical and selected-model
  stationary traces and PSDs;
- `fit/synthetic_recovery.csv` for parameter identifiability;
- `fit/training_only_temperature_selection.csv` for the regularization-kernel
  choice; and
- `fit/fit_summary.json` and `fit/ACCEPTANCE.md` for the authoritative result.

The EEG examples add a deterministic realization of the fitted posterior-mean
colored sensor-noise nuisance to the MAP neural simulation. They are stationary
examples, not time-aligned predictions of empirical samples. A good-looking
trace is useful QC but is not evidence of subject-level fit.
