# Accelerated dual-Jansen--Rit backend

TVB remains the scientific reference simulator. The optional JAX backend is an
equation-matched implementation used to evaluate many M5 candidates and random
replicates together on an accelerator.

It includes:

- the same two Jansen--Rit generators per Schaefer parcel;
- the mixed firing-rate long-range coupling;
- edge-specific delays obtained with TVB's nearest-integer rule;
- stochastic Heun integration;
- TVB's exponentially correlated neural-noise recurrence;
- temporal-average monitoring;
- the regularized analytic EEG gain and EEG reference; and
- batched candidate-specific regional parameters, speeds, noise, and optional
  edge weights.

The ordinary installation does not import JAX. A CPU-capable development
installation is:

```powershell
pip install -e ".[dev,analysis,accelerated]"
```

For NVIDIA acceleration, install the JAX CUDA wheel appropriate for the Linux
or WSL environment. Native Windows JAX is CPU-only. The validated Kaggle
notebook `shahmadi/tvbgpu` provides the project GPU environment.

Build a bank with:

```powershell
python scripts/run_m5_spectral.py --config configs/m5_spectral.toml `
  --stages bank --backend jax --batch-size 64
```

The generated NPZ, candidate table, and replicate table have the same schema
as the TVB path. `backend_metadata.json` records the device, precision, graph
size, delay-buffer size, and timing.

## Validation rule

The accelerated backend should not be accepted from speed alone. Validate a
new model change in this order:

1. deterministic waveform agreement with TVB;
2. multi-seed PSD and cross-spectrum agreement;
3. parameter-perturbation direction agreement;
4. finite-output and reproducibility tests; and
5. candidate-bank timing and memory measurement.

The random-number generators differ between NumPy/TVB and JAX, so stochastic
trajectories are not expected to match sample by sample. Their distributions
and spectral summaries must agree. The surface-Laplacian transform, if enabled,
is applied afterward by the shared CPU/MNE implementation.

## Kaggle validation (2026-09-17)

The private `shahmadi/tvbgpu` notebook was run on a Tesla T4 with JAX 0.7.2.
The model used the real 200-region connectome with 27,722 directed non-zero
weights and a 125-step delay buffer.

- Zero-noise TVB/JAX waveform comparison: median channel correlation 0.9972,
  zero sample shift, normalized RMSE 0.0735 using production float32.
- Stochastic comparison: mean normalized log-PSD correlation 0.9962. TVB alpha
  peaks across two seeds were 7.75 and 9.25 Hz; JAX peaks across eight seeds
  ranged from 8.75 to 10.75 Hz.
- Eight candidates by two seeds, 12 seconds each: 8.8 seconds end to end,
  including CSD creation and bank serialization.
- Production-duration stress test: 64 simulations by 62 seconds completed in
  114.8 seconds on one T4 and produced finite `32 x 2 x 39 x 26 x 26` CSDs.
- A 256-candidate by two-seed, 12-second calibration bank completed in 213.0
  seconds and produced a finite `256 x 2 x 39 x 26 x 26` bank.
- A 512-candidate factorized-spatial by two-seed, 12-second bank completed in
  425.8 seconds and produced a finite `512 x 2 x 39 x 26 x 26` bank.
- The final 512-candidate by three-seed, 62-second production run completed in
  2876.7 seconds (47.9 minutes). It produced a finite
  `512 x 3 x 39 x 26 x 26` bank of 367,969,178 bytes, which was downloaded and
  schema-validated locally before inference.

The measured production timing agrees with the 45--55 minute stress-test
estimate. Acceleration does not make an underfitting model scientifically
valid; the generated `fit/ACCEPTANCE.md` remains authoritative.
