# MDD–TVB baseline

This repository implements the pre-inference core of a planned whole-brain TMS
model:

1. audit and scale the existing Schaefer-200 structural connectome;
2. run a delayed, stochastic TVB Jansen–Rit network;
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
- reference: average reference.

It is suitable for simulator development and later feature-fitting experiments.
It is not a subject-specific BEM/FEM forward solution and amplitudes are not
calibrated to microvolts.

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

The default run simulates 5 seconds, discards the first second, and samples the
regional and EEG monitors at 500 Hz. Results are written to `outputs/baseline`:

- `baseline_timeseries.npz`: time, 200 parcel PSPs, 26 sensor EEG, gain matrix,
  and optional surface-Laplacian EEG;
- `scaled_connectome.npy` and `tract_lengths_mm.npy`;
- `regions.csv` and `sensors.csv` defining exact ordering;
- `eeg_psd.csv`;
- `run_metadata.json` with parameters, audits, and limitations;
- `baseline_summary.png` for visual QC.

## What must happen before fitting

The current parameters are a transparent baseline, not a calibrated healthy or
MDD model. Before fitting real EEG, the next stage must perform parameter sweeps,
stability mapping, integration-step sensitivity, simulated-data recovery, and
held-out feature definitions. A fit should estimate only a small hierarchical
parameter set and structured effective-coupling deviations—not 13,861 free
edges.

Neural noise is applied only to Jansen–Rit's `y4` derivative state, corresponding
to the excitatory-input pathway. Observation noise is disabled; downstream work
can add a separately calibrated sensor-noise model.

The count and distance matrices contain no embedded region labels. This project
uses the official Schaefer centroid ordering assumed by PyTepFit, but the
headerless matrix files cannot independently prove that ordering. This must be
confirmed from the connectome-generation provenance before publication.

A small non-fitting regime scan is included for baseline quality control:

```powershell
& .\.conda\python.exe scripts\scan_reference_regimes.py
```

It varies only the mean drive and global coupling, then reports regional and EEG
spectral peaks and alpha-power fractions. This is a reference-regime diagnostic,
not subject fitting or protocol optimization.

## FEM status

SimNIBS and its template mesh are not installed locally. This does not block the
resting-state model. The required later parcel-kernel contract is specified in
`docs/FEM_INTERFACE.md`; no FEM output is fabricated by this repository.
