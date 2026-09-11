# MDD–TVB baseline

This repository implements the pre-inference core of a planned whole-brain TMS
model:

1. audit and scale the existing Schaefer-200 structural connectome;
2. run a delayed, stochastic, spatially heterogeneous TVB Jansen–Rit network;
3. monitor the pyramidal PSP at all 200 parcels;
4. project it to the exact 26-channel TDBRAIN montage with TVB's analytic EEG
   monitor;
5. optionally apply the same spherical-spline surface-Laplacian transform used
   by the earlier EEG project;
6. save numerical data, provenance, spectra, and QC plots.

There is intentionally no subject fitting, effective-connectivity fitting,
plasticity, FEM field, or stimulation/protocol optimization in this version.

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
- sensors: the exact 26 TDBRAIN labels in MNE `colin27_1005` positions (the
  current name for the positions previously exposed as `standard_1005`);
- volume conductor: TVB analytic single sphere;
- near field: a declared 20 mm source–sensor distance floor prevents individual
  coarse parcel centroids from creating an inverse-square singularity;
- reference: average reference.

It is suitable for simulator development and later feature-fitting experiments.
It is not a subject-specific BEM/FEM forward solution and amplitudes are not
calibrated to microvolts.

The TDBRAIN channel *labels* and order are taken from the actual EEGLAB files.
Those files contain NaN channel coordinates, so the lead field necessarily uses
MNE's `colin27_1005` template locations for the matching names; they are not
digitized subject electrode positions. The full construction is documented in
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

## What must happen before fitting

The current parameters are a transparent baseline, not a calibrated healthy or
MDD model. Before fitting real EEG, the next stage must perform parameter sweeps,
stability mapping, integration-step sensitivity, simulated-data recovery, and
held-out feature definitions. A fit should estimate only a small hierarchical
parameter set and structured effective-coupling deviations—not 13,861 free
edges.

Neural noise is applied only to Jansen–Rit's `y4` derivative state, corresponding
to the excitatory-input pathway, and has a configurable temporal correlation.
Small, seeded network- and parcel-level variations in mean drive, common E/I
time scale, and drive variance prevent an unrealistically homogeneous periodic
orbit. The visual network receives a small, explicit increase in drive and drive
variance as an eyes-closed baseline prior. Every realized value is saved.
Observation noise is disabled; downstream work can add a separately calibrated
sensor-noise model.

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
