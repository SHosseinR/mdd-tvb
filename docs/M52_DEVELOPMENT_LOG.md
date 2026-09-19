# M5.2 development log

**Opened:** 2026-09-18
**Scope:** leakage-safe improvement of the resting-state EEG observation and
connectivity model after the frozen M5.1 result

This log records results in the order they were obtained. The acceptance rules
are fixed in `M52_VALIDATION_PROTOCOL.md`. The 65 previously evaluated M5.1
holdout subjects are not used for M5.2 architecture or hyperparameter choices.

> **Correction, 2026-09-19:** The version-24 template-BEM forward model below
> contained a coordinate-frame implementation error: TDBRAIN/Colin27 MRI
> electrode coordinates were mislabeled as head coordinates. The BEM fit and
> stopping decision in Sections 5–6 are **invalid as tests of the intended
> forward model** and are retained only as a failure record. The fixed gates
> have not changed. See Section 7 for the geometry audit and corrected rerun.

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
cross-spectra and zero failures. The full controlled GPU bank completed on
Kaggle notebook `shahmadi/tvbgpu`, version 24, from repository commit
`9a1b6ad`. Two CUDA devices ran 2,048 candidates × three replicates over
62 simulated seconds (two transient, 60 analysed). The run took 11,365 seconds
for simulation. Its downloaded 1,474,607,398-byte bank passed every check in
`scripts/validate_m52_bem_bank.py`: finite 2,048 × 3 × 39 × 26 × 26 CSD,
complete candidate and replicate tables, zero simulation failures, GPU
backend, both structural columns exactly zero, and the pinned candidate and
gain hashes. The full seven-artifact checksum inventory is in
`M52_BEM_BANK_MANIFEST.json`. The 65 consumed M5.1 holdout subjects were not
used.

## 5. Five-fold out-of-fold BEM result — invalid geometry

The BEM bank was passed through the same five development-only outer folds and
the same chronological first-half fitting / second-half unseen evaluation as
M5.1. Every subject appears once in the out-of-fold assessment. A paired
262-subject bootstrap (1,000 resamples) compares identical subjects, features,
and candidate parameter values. Lower cost ratios are better.

| Unseen metric | Analytic M5.1 | Template BEM | Paired BEM − M5.1 median [95% CI] |
| --- | ---: | ---: | ---: |
| Total cost / null | 0.745 | 1.076 | +0.228 [+0.188, +0.278] |
| Autospectrum cost / null | 0.471 | 0.588 | +0.041 [+0.020, +0.066] |
| Lagged-connectivity cost / null | 1.032 | 1.046 | +0.012 [−0.0004, +0.0235] |
| Alpha-topography cost / null | 0.912 | 3.538 | +2.476 [+2.182, +2.830] |

Only 6.5% of subjects had lower total cost under BEM; 3.1% had lower
alpha-topography cost. BEM's total, connectivity, and topography ratios failed
the absolute below-null gates. Its total ratio exceeded one in four of five
folds and its alpha-topography ratio ranged from 2.93 to 4.74 across all five.
No outer fold passed every primary individual gate. Only 45.8% of subjects
beat the pooled null, versus 77.1% under M5.1. The minimum active-parameter
synthetic-recovery correlation was 0.456, below the fixed 0.50 gate.

Group-effect preservation was mixed but does not rescue individual prediction.
Power-effect correlation rose from 0.598 to 0.659, with a paired-bootstrap
difference CI of [−0.180, +0.300]. Alpha-topography effect correlation fell
from 0.693 to 0.490. Lagged-connectivity effect correlation fell from 0.150
to 0.073 and its effect-norm retention remained only 0.112 (M5.1: 0.099),
far below the 0.25 eligibility gate. The BEM model therefore has status
`not_eligible`.

Evidence artifacts, all from the 262 development subjects:

- `outputs/m52_nested_template_bem/nested_summary.json` — SHA-256
  `aa4694229c4160dc540ec655947e63da8be95f02166b37612809e6b89e5f62e9`;
- `outputs/m52_nested_template_bem/out_of_fold_subject_posteriors.csv` —
  `1b9ef75502ce1387a36357ee546383cb4c212b00fa23115b4a360db3e2c4eb77`;
- `outputs/m52_nested_template_bem/nested_group_effects.png` —
  `c085e973a066996037b27f08859b4ce970e534f2bf3ac9e506d767eb8b10d4c1`;
- `outputs/m52_compare_template_bem/model_comparison.json` —
  `8e7f74b2a3a798a305ae61eb1b11adfa29518457488009ad52529de894b155af`;
- `outputs/m52_compare_template_bem/model_comparison.png` —
  `a3430ca7708b3df82c1a901fc01428b9a18980ae692b83465dc64ce15bf9827c`.

## 6. Historical stopping decision — withdrawn

Before the registration audit, the poor spatial endpoints appeared to trigger
the predeclared stopping rule. That inference was withdrawn: although the
simulation bank was finite, correctly ordered, and reproducible, the EEG
electrode locations were in the wrong frame for the template BEM. This is an
implementation failure, not evidence that a correctly registered BEM or its
parcel source representation fails. The 65-subject M5.1 holdout remains
off-limits for repair decisions. No TMS target/protocol optimization is
justified while the corrected forward model is unevaluated.

## 7. Coordinate-frame defect and corrected forward-model audit

The published TDBRAIN electrode coordinates are close to MNE's `colin27_1005`
sensor locations in the *MRI* coordinate frame (for example, Fz differs by
about 1.6 mm). MNE reports that standard montage's coordinate frame as `mri`.
The original `tdbrain_montage()` labeled the same coordinates as `head`, so MNE
did not apply the MRI-to-head fiducial transform before the fsaverage forward
solve. The audit found a median 15.8 mm and maximum 56.2 mm electrode-to-scalp
distance. After labeling the montage as MRI coordinates, those distances were
3.13 mm median and 6.70 mm maximum. The builder now rejects any gain whose
electrode-to-scalp maximum exceeds 10 mm; a regression test checks the frame.

The corrected gain is saved separately in `data/forward/template_bem_corrected`
so the invalid version-24 bank remains reproducible. Its parcel-to-parcel
sensor-topography cosine with the analytic gain rose from 0.336 to 0.446, and
its sensor-Gram correlation rose from 0.777 to 0.854. Signed cortical-field
retention after within-parcel aggregation has a median of 0.631 and minimum
0.287, so cancellation remains a model limitation even after registration.

A deliberately cheap exploratory test asked whether seven nonnegative network
power profiles could explain each development subject's first-half alpha
topography and predict the second half. It did not fit neural dynamics or read
any of the 65 consumed M5.1 holdout subjects:

| Forward gain | Median unseen alpha-profile RMSE / pooled null | Subjects below null |
| --- | ---: | ---: |
| Analytic | 0.894 | 63.7% |
| Old misregistered BEM | 1.600 | 23.3% |
| Corrected BEM | 0.988 | 51.1% |

The corrected gain greatly improves over the defective gain but does not beat
the analytic baseline on this descriptive power-profile oracle. Its inherited
spatial-mode maps also align more closely with the analytic maps (cosines 0.707
and 0.825, versus 0.249 and 0.145 for the defective gain). This evidence
justifies one controlled corrected full-bank test to establish the actual
spectral/connectivity result; it is **not** evidence that the corrected BEM
passes M5.2. The same candidate design, frozen scientific gates, five-fold
development subjects, and exclusion of the old holdout remain in force. The
corrected run is a defect repair, not a post-hoc expansion of parameter space.
The exploratory audit outputs are
`outputs/m52_forward_mapping_audit/forward_mapping_audit.json` (SHA-256
`8f8782cf71affe30bcebd2fb24d6d37c45945c504ed2ea6f71a17370e6ee3972`)
and `forward_mapping_audit.png` (SHA-256
`a61b720aa253539415599846068a31603e43feb7e6b035fc56d80bc0fae80170`).
MNE documents that Colin27 montage locations are already in fsaverage MRI
coordinates and recommends checking electrode-to-scalp distances for
coregistration: [standard montage](https://mne.tools/stable/generated/mne.channels.make_standard_montage.html),
[distance check](https://mne.tools/stable/generated/mne.dig_mri_distances.html).
