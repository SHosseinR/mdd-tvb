# Schaefer-200 fsaverage annotations

These annotation files are the official Schaefer 2018 200-parcel, seven-network
parcellation in FreeSurfer `fsaverage` space. They are used only to aggregate a
distributed template-BEM lead field to the same 200-region order as the
connectome.

Source:

`https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label`

Files and SHA-256 hashes:

- `lh.Schaefer2018_200Parcels_7Networks_order.annot`:
  `c225937c31584c74d3e7c8fba7c17916ce4e730faa542ee61659041db136ac87`
- `rh.Schaefer2018_200Parcels_7Networks_order.annot`:
  `5439cd1fb913cbb76366f9590b5733efcb7adf9cde25413f6e0b681ff0abdcf8`

The medial-wall labels are excluded. The remaining label names are checked
exactly against `Schaefer2018_200Parcels_7Networks_centroids.csv` before a gain
matrix is saved.
