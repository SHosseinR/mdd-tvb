# M5 spectral redesign

## Outcome

The new pipeline is implemented and runs end to end, but the current eight-
candidate pilot still fails its fit gate. It is a diagnostic pilot, not an
accepted Healthy or MDD-indication model and not an input to stimulation
optimization.

The decisive comparison is against a pooled empirical training-subject null,
not against an arbitrary TVB candidate. On the 65 subjects excluded from all
feature-reduction fitting, the posterior predictive median unseen cost ratio is
about 1.40; lower than one is required. Even an invalid oracle that chooses a
candidate using the unseen data remains above one. The empirical first half,
by contrast, predicts the same subject's unseen half at about 0.40 times the
pooled-null cost. Thus stable individual EEG information exists, but this pilot
model does not capture it. This is underfitting/model mismatch, not overfitting.

## What changed from M5 v2

1. Every Schaefer parcel contains two locally coupled Jansen--Rit generators:
   an alpha generator and a faster generator. Their weighted pyramidal PSPs
   drive and are projected from the same delayed TVB structural network.
2. Excitatory and inhibitory inverse time constants have separate parameters.
   Tying them together was tested and rejected because it shifted the reference
   alpha regime to approximately 6 Hz.
3. The empirical objective is the 2--40 Hz complex sensor cross-spectral matrix,
   including autospectra and real/imaginary coherency. It no longer collapses
   the PSD across channels.
4. A group-blind eigenspace reduces the average-referenced 26-channel data to
   ten sensor modes. A diagonal shrinkage of 0.05 stabilizes each spectral
   matrix.
5. Periodic residual spectra are separated from a log-linear aperiodic
   background fitted outside 8--29 Hz. Per-mode intercepts are observation-gain
   nuisances; the exponents remain in the objective.
6. Two non-overlapping quarters inside the fitting half select temporally
   reliable coordinates. Real and imaginary coherency receive equal quotas, so
   reliable zero-lag volume-conduction structure cannot exclude phase.
7. The second half of every recording is never used for feature reduction or
   fitting. A stratified 20% subject holdout also remains outside the
   group-blind feature transformer.
8. Each subject gets an approximate finite-bank posterior, not only a winning
   candidate. White/pink diagonal observation-noise amount and exponent are
   marginalized as nuisance variables. Neural posterior means, 5--95% ranges,
   MAP states, and effective sample sizes are saved.
9. The production configuration uses three stochastic seeds and 60 analysed
   seconds per seed. The pilot uses two seeds and ten analysed seconds solely
   to verify the machinery.

Structural weights remain fixed. The earlier complete +/-10% network-pair scan
showed inadequate EEG sensitivity and recovery. Conduction speed and a single
mean-preserving Dorsal Attention time-scale contrast are retained. Structural
weight modes may be reconsidered only after this observation/model mismatch is
fixed and a new sensitivity/recovery audit passes under the complex-spectral
objective.

## Commands

Pilot (already completed):

```powershell
& .\.conda\python.exe scripts\run_m5_spectral.py --config configs\m5_spectral_pilot.toml
```

Production settings (implemented but intentionally not launched while the
pilot gate fails):

```powershell
& .\.conda\python.exe scripts\run_m5_spectral.py --config configs\m5_spectral.toml
```

Stages can be resumed independently with `--stages extract`, `bank`, or `fit`.
The production bank is 128 candidates x 3 random seeds x 62 seconds and is
expected to require roughly an overnight run on the present CPU.

## Current pilot diagnosis

- The corrected candidate bank peaks mainly from 8--11 Hz; the accidental
  theta-dominant reference problem is fixed.
- The posterior remains broad across almost all eight neural candidates. This
  is expected with such a small bank and is explicitly reported by posterior
  effective sample size.
- Autospectral and complex-coherency posterior predictions both remain worse
  than the pooled empirical null on held-out unseen EEG.
- The pooled-target calibration selects the maximum allowed observation-noise
  fraction (0.75). That boundary saturation is evidence of unresolved neural
  or forward-model mismatch, not permission to explain EEG as sensor noise.
- Empirical zero-lag coherency was highly reliable, but it is not a clean neural
  connectivity target with a shared approximate spherical lead field.
- Balanced phase-sensitive selection is therefore mandatory even though it
  makes the current failure more visible.
- Subject-holdout group-effect correlations must be interpreted together with
  effect-norm retention. A moderate correlation with a near-zero retained norm
  is not successful preservation.

## Required next gate

Before the production subject bank, calibrate the group-blind model to the
pooled empirical cross spectrum and determine why the shared lead field and
neural covariance miss reliable coherency. Candidate remedies must be tested,
not assumed: stronger network coupling, a small recoverable set of regional
excitability/noise contrasts, and a source-space or surface-Laplacian validation
view. Do not add thousands of edges or arbitrary correlated observation noise.

Only proceed to subject inference when all of the following hold:

- the population model beats the pooled spectral-shape baseline on declared
  model discrepancy metrics;
- subject-holdout unseen posterior prediction is competitive with the pooled
  empirical null;
- posterior candidate ESS and synthetic recovery demonstrate information about
  at least a few neural parameter combinations;
- Healthy--MDD differences retain both direction and non-trivial magnitude in
  autospectral/topographic and phase-sensitive connectivity views.

## Main outputs

- `outputs/m5_spectral_pilot/empirical/cross_spectra_*.npz`
- `outputs/m5_spectral_pilot/bank/spectral_simulation_bank.npz`
- `outputs/m5_spectral_pilot/fit/subject_posteriors.csv`
- `outputs/m5_spectral_pilot/fit/fit_summary.json`
- `outputs/m5_spectral_pilot/fit/m5_spectral_validation.png`
- `outputs/m5_spectral_pilot/fit/m5_spectral_group_effects.png`
