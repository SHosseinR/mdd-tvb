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

## 5. Five-fold out-of-fold BEM result

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

## 6. Scientific interpretation and stopping decision

The stronger anatomical forward solve did **not** improve the prespecified
spatial endpoints in this controlled test. The paired connectivity CI includes
zero and its point estimate is slightly worse; alpha topography is much worse
with a CI entirely above zero. The predeclared Section 13 stopping rule in
`M52_VALIDATION_PROTOCOL.md` therefore applies: stop this model-expansion
sequence and report, rather than fitting a correlated background or adding
neural-dynamics degrees of freedom to rescue this failed forward-model branch.
Do not proceed to TMS target/protocol optimization or use the already-consumed
65-subject M5.1 holdout to revise this decision.

This result rejects the **specific** template-BEM regional projection used
here, not BEM physics in general. Its parcel operator assumes one synchronous,
uniform cortical-normal source amplitude per Schaefer parcel and averages
signed vertex lead fields; cortical folding can produce substantial
cancellation. The two spatial noise modes were also originally derived in the
analytic-gain basis and deliberately kept fixed to isolate the gain change.
Those are plausible representational limitations, not demonstrated bugs. The
bank hashes, channel/parcel ordering, average reference, finite arrays, GPU
provenance, and zero-failure checks passed. A future BEM study would need a
prospectively specified parcel source distribution or subparcel state model,
with new development data for selection, rather than post-hoc tuning on these
five folds.

The next scientific decision is whether to acquire independent spatial data
(individual electrode digitization/MRI, source-space constraints, or a new EEG
cohort) or to formulate a narrowly prespecified dynamics hypothesis that can
be tested in a new locked design. Additional optimization of this frozen
candidate bank is not a credible path to TMS target selection.
