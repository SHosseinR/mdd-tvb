# Baseline regime selection

The initial homogeneous baseline was numerically correct but scientifically too
regular. It produced a narrow 9.5 Hz limit cycle with conspicuous harmonics near
19, 28.5, and 38 Hz. Its normalized 1–45 Hz spectral entropy was approximately
0.30, and the first five EEG principal components explained about 91% of sensor
variance.

The revised baseline changes the generator, not merely the plot:

- fixed, seeded network- and parcel-level heterogeneity in mean drive and common
  excitatory/inhibitory time scale;
- heterogeneous temporally coloured drive entering only the `y4` excitatory
  input derivative;
- a small explicit visual-network drive/noise prior for eyes-closed EEG;
- global coupling reduced from 15 to 7;
- a 30-second simulation with a 2-second transient;
- multitaper spectra rather than a short Welch estimate.

The first revised scalp projection remained front-heavy. Audit of the analytic
gain identified two mirrored prefrontal parcel-centroid/sensor pairs at 13.1 and
15.3 mm; the inverse-square point-dipole approximation let these two pairs
dominate the map. A 20 mm minimum distance regularizes only those 2 of 5,200
pairs. This is preferable to forcing a large visual-network amplitude merely to
hide a forward-model artifact, but remains a template approximation that must be
replaced or validated with a surface BEM/FEM lead field later.

A small non-fitting scan indicated that `a=0.13 ms^-1`, `b=0.065 ms^-1`,
`mu=0.22`, global coupling 7, drive dispersion `1e-4`, and a 5 ms noise
correlation time retained an approximately 11 Hz alpha peak while increasing
spectral entropy and reducing low-rank sensor synchronization. These are
reference-simulation choices, not estimated healthy-population parameters.

The rationale is consistent with work treating resting EEG as a mixture of
stochastically driven/damped alpha processes and with whole-brain TVB examples
that use longer simulations and global coupling near 7:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC9045666/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12047647/

Before subject fitting, these settings require comparison with empirical
healthy EEG distributions and integration-step/seed sensitivity. They must not
be described as a fitted normative brain.
