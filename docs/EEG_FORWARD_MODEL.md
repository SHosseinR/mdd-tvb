# EEG forward model and TDBRAIN coordinates

## What comes from TDBRAIN

The exact 26 channel labels and their order come from the local TDBRAIN EEGLAB
files:

`Fp1, Fp2, F7, F3, Fz, F4, F8, FC3, FCz, FC4, T7, C3, Cz, C4, T8, CP3, CPz,
CP4, P7, P3, Pz, P4, P8, O1, Oz, O2`.

An inspected source file has a 500 Hz sampling rate and exposes all 26 labels,
but every XYZ electrode coordinate is NaN. The preceding EEG project documents
the same dataset-wide limitation. Therefore, this project does **not** claim to
use measured subject-level TDBRAIN electrode positions.

Instead, the coordinates come directly from Table 3 of the TDBRAIN data
descriptor: [van Dijk et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9198070/).
The versioned values are stored in
`data/sensors/TDBRAIN_Table3_electrode_coordinates.csv`. They use millimetres
with left/right X, anterior/posterior Y, and inferior/superior Z. Their unit
directions differ from MNE's `colin27_1005` montage by a median 1.38 degrees and
a maximum 3.59 degrees. These remain publication-level template coordinates,
not individual digitization.

## Lead-field construction

The current observation model is TVB's analytic single-sphere EEG monitor:

1. sources are the 200 Schaefer MNI parcel centroids;
2. each source orientation is the radial unit vector from the mean centroid;
3. published TDBRAIN coordinates are reduced to unit vectors and placed on
   TVB's fitted sphere around the source cloud;
4. each sensor/source gain is calculated as a radial point-dipole potential,
   proportional to
   `q · (sensor - source) / distance³ / (4πσ)`;
5. a 20 mm minimum effective distance regularizes the inverse-square near field;
6. the projected potentials are average referenced.

The distance floor changes only two of 5,200 sensor/source pairs in this
template. Without it, mirrored prefrontal centroids at 13.1 and 15.3 mm from Fp1
and Fp2 dominated the scalp map. The exact audit is saved in
`outputs/baseline/run_metadata.json`.

## What it is not

This is not a three-shell BEM, finite-element head model, calibrated dipole
moment, or subject-specific lead field. It ignores individual skull geometry,
conductivity, electrode digitization, and distributed source extent. A later
stimulation/FEM stage should use SimNIBS with an explicit template or subject
head mesh and aggregate the resulting cortical field to Schaefer parcels. If
individual TDBRAIN digitized electrode coordinates become available, they
should replace the publication coordinates independently of that FEM step.

The optional spherical-spline surface Laplacian uses the same Table 3 XYZ
positions. Its sphere is fixed at the published coordinate origin with radius
equal to the median electrode radius (93.60 mm); this avoids an underconstrained
automatic sphere fit from only 26 electrodes.
