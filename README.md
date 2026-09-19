# MDD–TVB baseline

This repository implements a whole-brain TMS-modeling core and its first
hierarchical EEG-fitting stage:

1. audit and scale the existing Schaefer-200 structural connectome;
2. run a delayed, stochastic, spatially heterogeneous TVB Jansen–Rit network;
3. monitor the pyramidal PSP at all 200 parcels;
4. project it to the exact 26-channel TDBRAIN montage with TVB's analytic EEG
   monitor;
5. optionally apply the same spherical-spline surface-Laplacian transform used
   by the earlier EEG project;
6. save numerical data, provenance, spectra, and QC plots.

M5 adds diagnosis-level and individual EEG fitting. Plasticity, FEM fields,
stimulation, and protocol optimization remain later milestones.

**Current M5.2 status (2026-09-19):** A correctly registered, distributed
template EEG BEM was evaluated against the analytic gain on 262 development
subjects in five outer folds. The BEM did not pass the fixed unseen EEG gates:
alpha-topography fit was worse, and its small lagged-connectivity point
improvement was uncertain. The analytic gain remains selected for the next
prespecified correlated-background test; no TMS optimization is justified yet.
See `docs/M52_DEVELOPMENT_LOG.md` and `docs/PROJECT_TECHNICAL_REPORT.md` for
the full numerical record. The earlier misregistered BEM run is preserved but
explicitly invalid for that comparison.

## Scientific definition

The supplied streamline-count matrix is transformed with `log1p`, then globally
scaled so the largest regional input sum is one. TVB's global coupling parameter
therefore remains separate from connectivity pattern. Tract lengths are treated
as millimetres; conduction speed is in mm/ms (numerically equivalent to m/s).

The simulated regional observable is `y1 - y2`, the excitatory-minus-inhibitory
postsynaptic potential of the Jansen–Rit pyramidal population. TVB's EEG monitor
projects `y1` and `y2` separately through one gain matrix and the code subtracts
them afterward.

The present gain matrix is an explicitly documented template approximation:

- source locations: Schaefer-200 MNI centroids;
- source orientations: radial vectors derived from those centroids;
- sensors: the exact 26 TDBRAIN labels at the published TDBRAIN Table 3 XYZ
  positions;
- volume conductor: TVB analytic single sphere;
- near field: a declared 20 mm source–sensor distance floor prevents individual
  coarse parcel centroids from creating an inverse-square singularity;
- reference: average reference.

It is suitable for simulator development and later feature-fitting experiments.
It is not a subject-specific BEM/FEM forward solution and amplitudes are not
calibrated to microvolts.

The TDBRAIN channel *labels* and order are taken from the actual EEGLAB files.
Those files contain NaN channel coordinates, so the lead field uses the
dataset-publication coordinates rather than digitized subject positions. The
published directions are close to, but not identical to, MNE's
`colin27_1005`/`standard_1005` template. The full construction is documented in
`docs/EEG_FORWARD_MODEL.md`.

## Setup on this machine

A dedicated Python 3.11 environment has been created at `.conda` so the older
`eeg-graph` environment is untouched.

```powershell
& .\.conda\python.exe -m pip install -e ".[dev]"
```

For a clean recreation elsewhere:

```powershell
conda env create -f environment.yml
conda activate mdd-tvb
```

## Validate inputs

```powershell
& .\.conda\python.exe -m mdd_tvb validate --config configs\baseline.toml
```

This checks matrix shapes, finiteness, symmetry, diagonal, matched count/length
support, graph connectedness, atlas ordering, delays, and sensor availability.

## Run the baseline

```powershell
& .\.conda\python.exe -m mdd_tvb run --config configs\baseline.toml
```

The default run simulates 30 seconds, discards the first 2 seconds, and samples
the regional and EEG monitors at 500 Hz. Results are written to `outputs/baseline`:

- `baseline_timeseries.npz`: time, 200 parcel PSPs, 26 sensor EEG, gain matrix,
  and optional surface-Laplacian EEG;
- `scaled_connectome.npy` and `tract_lengths_mm.npy`;
- `regions.csv` and `sensors.csv` defining exact ordering;
- `regional_parameters.csv` with every seeded parameter/noise multiplier;
- `eeg_psd.csv`;
- `run_metadata.json` with parameters, audits, and limitations;
- `baseline_summary.png` for visual QC.

The NPZ stores the raw average-referenced model observation. Only the stacked
EEG trace in the QC figure is zero-phase high-pass filtered at 1 Hz, because the
Jansen–Rit equilibrium produces large channel-specific DC offsets. Multitaper
spectra are computed after per-channel demeaning; neither operation alters the
stored monitor signal.

The empirical `.set` files used by the preceding project were already filtered
at 1–60 Hz, notch filtered at 49–51 Hz, high-amplitude segments removed, and
average referenced. Future fitting must apply a matched observation-
preprocessing operator to simulated EEG; it should not fit the arbitrary model
DC equilibrium or request biased empirical data.

## M5: group and subject fitting

M5 fits one model per subject; it does not fit one Healthy model and one MDD
model. Group summaries are diagnosis-label-blind individual fits aggregated
afterward. The current method uses two locally coupled Jansen--Rit generators
per parcel, delayed whole-brain coupling, complex 2--40 Hz cross spectra, a
direct alpha-topography block, explicit periodic/aperiodic separation, and a
finite-bank posterior with Gaussian shrinkage.

The production bank fits nine global-dynamics and six structured
regional-physiology parameters. Two additional columns represent symmetric
network-pair connectome modes bounded to +/-10%, but production fixes both at
zero. Two diagnosis-blind calibration banks showed that resting scalp EEG did
not recover these structural perturbations reliably; allowing them to move
would therefore create prior-driven subject differences rather than measured
tract changes. The common Schaefer connectome remains the validated reference.

```powershell
& .\.conda\python.exe scripts\run_m5_spectral.py --config configs\m5_spectral.toml
& .\.conda\python.exe scripts\plot_m5_eeg_examples.py --config configs\m5_spectral.toml
```

Every recording is split temporally. Its first half fits that subject and its
second half evaluates the same prediction. A stratified 20% subject holdout is
excluded from sensor-basis, feature, objective, nuisance, and temperature
selection; its second halves are the final test. Diagnosis labels enter only
the second-level group-effect audit.

For batched simulation-bank generation on an NVIDIA GPU, install the optional
`accelerated` dependencies and select `--backend jax`. TVB remains the default
and the scientific reference; the accelerated path mirrors its delayed
dual-generator equations, coloured neural noise, temporal averaging, and EEG
observation model. See `docs/JAX_BACKEND.md`.

The original eight-candidate pilot underfit the unseen data. The revised M5.1
workflow therefore adds training-only objective/temperature calibration,
quadratic Gaussian shrinkage, and a broad-plus-adaptive 2,048-candidate design.
The completed production bank uses three random seeds and 62 seconds per
simulation (two seconds discarded, 60 analysed). On the 65-subject holdout,
its median unseen cost is 0.759 times the pooled-null cost and 72.3% of subjects
beat that null. It passes 9 of 11 acceptance gates. Alpha-topography prediction
is narrowly worse than the null (1.046), and the held-out lagged-coherency group
effect is not preserved (r=0.048). `fit/ACCEPTANCE.md` is authoritative, and
stimulation work remains blocked until the required gates pass. See
`docs/M5_SPECTRAL_REDESIGN.md` for the estimand, leakage controls,
metrics, outputs, and limitations. The validation and group-effect figures use
subject holdouts only; the held-out parameter-effect figure fades parameters
that fail synthetic recovery. The earlier `scripts/run_m5.py` workflow is
retained only as a historical feature-based prototype.

## Fitting cautions

The baseline parameters are a transparent reference, not a calibrated healthy
or MDD model. M5 therefore uses a small structured parameterization rather than
13,861 independently adjustable edges, reports synthetic-recovery diagnostics,
and evaluates held-out data. Results remain mechanistic candidates rather than
identified biological ground truth.

Neural noise is applied only to Jansen–Rit's `y4` derivative state, corresponding
to the excitatory-input pathway, and has a configurable temporal correlation.
Small, seeded network- and parcel-level variations in mean drive, common E/I
time scale, and drive variance prevent an unrealistically homogeneous periodic
orbit. The visual network receives a small, explicit increase in drive and drive
variance as an eyes-closed baseline prior. Every realized value is saved.
Observation noise is disabled in the baseline run. The revised M5 pilot adds a
fixed, group-blind colored background before the same CSD transform. It is a
pragmatic observation/unmodelled-neural component, not a fitted disease
parameter.

The 30-second duration follows a conservative convergence-oriented baseline for
network/spectral summaries. The model remains a hypothesis generator: these
heterogeneity priors must later be checked by synthetic recovery and fitted or
rejected using held-out empirical EEG.

The rationale and the non-fitting regime-selection record are documented in
`docs/BASELINE_SELECTION.md`. The current inverse synaptic time constants
(`a=0.13 ms^-1`, `b=0.065 ms^-1`) are within TVB's declared JR ranges but differ
from the canonical 0.10/0.05 values; this is an explicit reference-regime choice,
not a fitted physiological conclusion.

The count and distance matrices contain no embedded region labels. This project
uses the official Schaefer centroid ordering assumed by PyTepFit, but the
headerless matrix files cannot independently prove that ordering. This must be
confirmed from the connectome-generation provenance before publication.

A small non-fitting regime scan is included for baseline quality control:

```powershell
& .\.conda\python.exe scripts\scan_reference_regimes.py
```

It can vary mean drive, global coupling, time scale, noise, and the visual-noise
prior, then reports spectral, spatial, and low-rank diagnostics. This is a
reference-regime diagnostic, not subject fitting or protocol optimization.

If only the observation model changes, the saved linear regional monitor can be
reprojected without rerunning the neural dynamics:

```powershell
& .\.conda\python.exe scripts\reproject_saved_run.py
```

## FEM status

SimNIBS and its template mesh are not installed locally. This does not block the
resting-state model. The required later parcel-kernel contract is specified in
`docs/FEM_INTERFACE.md`; no FEM output is fabricated by this repository.
