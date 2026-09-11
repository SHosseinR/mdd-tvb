# Template-FEM boundary

FEM is deliberately outside the current baseline simulation. SimNIBS and a
template head mesh are not installed in this workspace, and this stage does not
claim to have produced an electric field.

The later stimulation module should consume a precomputed table with one row per
candidate coil pose and 200 columns in the exact Schaefer atlas order used by
`regions.csv`. Each row represents a parcel-reduced cortical electric-field
kernel. Metadata must record at least:

- SimNIBS version and template head mesh;
- coil file/model;
- coil centre, orientation, scalp distance, and dI/dt;
- whether field magnitude or signed normal field was reduced;
- parcel reduction statistic and normalization;
- SHA-256 hashes of the mesh, coil, atlas, and kernel files.

The optimizer must select or scale a precomputed kernel. It must not alter the
subject's fitted baseline connectivity or rerun FEM inside every TVB evaluation.

