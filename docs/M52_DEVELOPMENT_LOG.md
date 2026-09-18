# M5.2 development log

**Opened:** 2026-09-18
**Scope:** leakage-safe improvement of the resting-state EEG observation and
connectivity model after the frozen M5.1 result

This log records results in the order they were obtained. The acceptance rules
are fixed in `M52_VALIDATION_PROTOCOL.md`. The 65 previously evaluated M5.1
holdout subjects are not used for M5.2 architecture or hyperparameter choices.

## 1. Reproducibility freeze

`scripts/freeze_m51_release.py` validates and hashes the final M5.1 release.
The resulting `M51_RELEASE_MANIFEST.json` covers 29 files (1,887,414,297 bytes)
and verifies:

- repository commit used for the GPU bank: `a302fdf`;
- 2,048 candidates and 6,144 candidate-replicate simulations;
- three stochastic replicates per candidate;
- finite saved arrays and zero simulation failures;
- 60 analysed seconds after the transient; and
- both unsupported structural-connectome columns fixed exactly at zero.

The deterministic M5.2 split contains the original 262 development subjects:
141 Healthy and 121 MDD-indication. Five outer validation folds contain
54/52/52/52/52 subjects. Four inner-fold labels are defined within every outer
training partition. All 65 consumed M5.1 holdout subjects are excluded.

## 2. Frozen M5.1 model under M5.2 nested evaluation

The unchanged analytic-sphere M5.1 bank was evaluated on out-of-fold
predictions for all 262 development subjects. Every outer fold relearned the
spectral transformer, population nuisance state, posterior temperature, and
subject posteriors from its training partition.

### Aggregate out-of-fold result

| Metric | Result | Gate |
| --- | ---: | ---: |
| Median total unseen cost / pooled null | 0.745 | pass |
| Subjects beating pooled null | 77.1% | pass |
| Median autospectrum cost / null | 0.471 | pass |
| Median alpha-topography cost / null | 0.912 | pass |
| Median selected-connectivity cost / null | 1.032 | **fail** |
| Power Healthy–MDD effect correlation | 0.598 | pass |
| Alpha-topography effect correlation | 0.693 | pass |
| Lagged-connectivity effect correlation | 0.150 | pass |
| Lagged-connectivity effect-norm retention | 0.099 | **fail** |
| Minimum active-parameter recovery correlation | 0.584 | pass |
| Outer folds passing all primary individual gates | 1/5 | **fail** |

The aggregate status is `not_eligible`. Connectivity is above the pooled null
in four folds and its group-effect magnitude is attenuated to about 10% of the
empirical magnitude. Power and alpha topography generalize substantially
better. This reproduces the main M5.1 limitation without reading its holdout,
so it is not an accidental result of one subject split.

Local evidence hashes:

- `nested_summary.json`:
  `e49e12a7685331b8ef01c375a5d2b67e8578a75d9a63d6ff1cc1f62ce1a20777`
- `out_of_fold_subject_posteriors.csv`:
  `f5d989517b9a607e4042f568374ee14820c3fc3105e9bc260a6ce48366c3d352`
- `nested_group_effects.png`:
  `13492abe8223e0a28c1a9491ea6956b68635cbb44244bbc62f0fcb67bd615b4f`

## 3. Template-BEM forward model

A distributed EEG forward model has been implemented with:

- the published TDBRAIN Table 3 electrode coordinates;
- MNE `fsaverage` three-layer BEM;
- the `fsaverage-ico-5` cortical source space;
- fixed cortical-normal source orientation;
- the official Schaefer-2018 200-parcel/7-network annotations; and
- area-weighted aggregation of all active cortical vertices in each parcel.

The final gain is average referenced and whole-matrix RMS normalized because
absolute subject/electrode gain remains a nuisance. It is a template forward
model, not personalized anatomy. It is more realistic than the analytic
centroid/sphere model but does not claim individualized skull geometry.

Geometry and integrity audit:

- regional gain shape: 26 sensors × 200 parcels;
- vertex gain shape: 26 × 20,484 sources;
- post-reference rank: 25;
- source vertices per parcel: minimum 23, median 88, maximum 219;
- median TDBRAIN-to-colin27 electrode angular difference: 1.38°;
- maximum electrode angular difference: 3.59°;
- analytic-versus-BEM flattened gain correlation: 0.211;
- median per-parcel topography cosine: 0.336;
- sensor spatial-Gram correlation: 0.777; and
- gain SHA-256:
  `299efec33c8a38879ef480e5aa4757ff9b62e48636067d23dd61f6ba4c9ebf8b`.

The moderate-to-low parcel-level agreement confirms that this is a material
observation-model test rather than a cosmetic rewrite. Empirical superiority
is not inferred from geometry; it must be established by the nested folds.

## 4. Controlled forward-model comparison

The full BEM bank is specified in `configs/m52_template_bem.toml`. Its only
scientific change from `configs/m5_spectral.toml` is the observation gain and
output directory. The exact 2,048 M5.1 candidate rows are copied to
`configs/m52_m51_candidate_parameters.csv`; the source and copy have the same
SHA-256:

`1f2b5a79abb98ddf7c496aea9ee735fb354ed9d0ae2d47cf13faceea0b40b4da`.

A local JAX smoke test completed 8 candidates × 2 replicates with finite
cross-spectra and zero failures. The full controlled GPU run is the next active
step. After it completes, the BEM bank will be passed through the same five
outer development folds and compared with the table in Section 2. No BEM
choice will be made using the consumed M5.1 holdout.

## 5. Current scientific decision

The evidence currently supports continuing M5.2, not beginning TMS target
optimization. Specifically:

1. the baseline is clearly under-modeling lagged connectivity magnitude;
2. this deficit persists across nested development folds;
3. a materially different, anatomically distributed forward model is ready for
   a controlled test; and
4. structural-connectome fitting remains disabled because scalp resting EEG did
   not recover the two bounded structural modes in prior calibration.

If BEM improves neither alpha-topography nor connectivity, the protocol's next
candidate is a constrained, diagnosis-blind correlated-background model. If it
improves total fit only by further suppressing connectivity effects, it must be
rejected. Neural-dynamics expansion comes only after these observation-model
tests, with each new parameter required to pass sensitivity and synthetic
recovery.
