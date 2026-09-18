# M5.2 preregistered development and validation protocol

**Protocol date:** 2026-09-18  
**Predecessor:** frozen M5.1 fixed-connectome production run  
**Purpose:** improve spatial/topographic and connectivity fidelity without
reusing the consumed M5.1 final holdout for model selection

## 1. Fixed scientific question

M5.2 asks whether a prespecified improvement to the EEG observation model,
background covariance, or low-dimensional neural dynamics improves unseen
resting-state EEG prediction while retaining Healthy–MDD-indication effects.

It is not a diagnosis classifier. Every subject continues to be fitted without
using diagnosis labels. Labels are used only to balance folds and to evaluate
second-level group-effect preservation after fitting.

## 2. Data that may and may not be used

### Development data

M5.2 model development uses only the 262 subjects assigned to `train` in the
frozen M5.1 split. The deterministic assignments are stored in:

- `configs/m52_nested_splits.csv`; and
- `configs/m52_nested_splits.json`.

There are five outer folds and four inner folds within each outer-training
partition. Splits are stratified only to balance the Healthy and MDD-indication
counts.

### Excluded data

The 65 M5.1 holdout subjects have already been evaluated and must not be used to
choose M5.2:

- forward model;
- nuisance/background model;
- neural-model family;
- parameter ranges;
- feature coordinates or weights;
- regularization strength;
- posterior temperature; or
- acceptance thresholds.

They remain a frozen record of M5.1 only. A genuinely independent final M5.2
claim requires an external replication cohort. A newly locked internal subset
can provide a prospective comparison, but it is not fully independent because
the current project has already used all 262 development subjects.

## 3. Temporal separation within a subject

The existing chronological split is retained:

1. first half: subject fitting;
2. first-half quarters: reliability estimation where applicable; and
3. second half: unseen within-subject prediction.

No model is optimized to match the stochastic waveform phase. Outcomes are
stationary spectral and cross-spectral statistics.

## 4. Nested procedure

For each outer fold:

1. exclude the outer-validation subjects completely from transformer fitting,
   hyperparameter selection, nuisance selection, and posterior-temperature
   selection;
2. use the four inner folds to compare only the declared candidate models;
3. select one model by the rule in Section 8;
4. refit its label-blind transformer and global inference settings on all
   outer-training subjects; and
5. generate posterior predictions for the outer-validation subjects using only
   their fitting halves.

After all five outer folds, concatenate the out-of-fold predictions. Every
reported M5.2 development subject must therefore have a prediction from a model
that did not use that subject to select shared components or hyperparameters.

## 5. Prespecified model families

Model additions are evaluated sequentially so their effect can be attributed.

| ID | Forward model | Background model | Neural model | Role |
| --- | --- | --- | --- | --- |
| M51 | Analytic sphere | Diagonal colored sensor noise | Current dual JR | Frozen comparator |
| FWD | Distributed template BEM | Diagonal colored sensor noise | Current dual JR | Tests observation geometry |
| BKG | Selected forward model | Constrained label-blind correlated background | Current dual JR | Tests residual covariance |
| DYN | Selected forward/background | Same as BKG | One prespecified dynamics extension | Tests missing mechanism |

`DYN` must not become an unrestricted search. Before it is simulated, its exact
equations, parameter ranges, and biological interpretation must be recorded in
an amendment to this protocol.

## 6. Fixed primary outcomes

All cost ratios use the pooled empirical development null. Lower is better;
below one beats the null.

Primary individual outcomes:

1. median unseen total-cost ratio;
2. median unseen alpha-topography cost ratio;
3. median unseen selected-connectivity cost ratio; and
4. fraction of subjects with unseen total-cost ratio below one.

Primary second-level outcomes from out-of-fold predictions:

1. channel × frequency power-effect correlation;
2. alpha-topography effect correlation;
3. lagged-connectivity effect correlation; and
4. predicted/empirical lagged-connectivity effect-norm ratio.

Secondary outcomes:

- autospectrum cost ratio;
- fitting-to-unseen generalization gap;
- posterior effective sample size;
- temporal-persistence reference;
- parameter boundary frequency;
- nuisance boundary frequency;
- synthetic-recovery correlation and normalized RMSE; and
- stability across folds and simulation seeds.

## 7. Minimum eligibility gates

A candidate M5.2 model is not eligible for the TMS stage unless its concatenated
out-of-fold predictions satisfy all of the following:

| Gate | Threshold |
| --- | ---: |
| Median unseen total cost / null | < 1.00 |
| Subjects beating null unseen | > 50% |
| Median autospectrum cost / null | < 1.00 |
| Median alpha-topography cost / null | < 1.00 |
| Median selected-connectivity cost / null | < 1.00 |
| Power group-effect correlation | >= 0.30 |
| Alpha-topography group-effect correlation | >= 0.30 |
| Lagged-connectivity group-effect correlation | >= 0.10 |
| Lagged-connectivity effect-norm retention | >= 0.25 |
| Active-parameter synthetic-recovery correlation | >= 0.50 each |
| Observation nuisance at tested upper boundary | No |

The effect-norm gate is new and is declared before M5.2 evaluation because M5.1
showed that a correlation alone can obscure near-complete attenuation.

The model should also pass each primary individual gate in at least four of five
outer folds. Fold-level group-effect correlations are reported but not required
to pass separately because the per-fold group sample is smaller; their direction
and heterogeneity must be shown.

## 8. Inner-fold selection rule

The inner loop must not select a model by one favorable metric. Candidate models
are ordered lexicographically:

1. discard any model with non-finite simulations, failed recovery, or a
   nuisance optimum fixed at an unexplored boundary;
2. discard any model whose median total unseen ratio is >= 1.00;
3. among the remainder, prefer models that bring both alpha-topography and
   selected-connectivity ratios below one;
4. if more than one remains, minimize the larger of those two spatial ratios;
5. use median total unseen ratio as the next tie-breaker; and
6. use the simpler model as the final tie-breaker.

Group-effect metrics are eligibility audits, not continuously optimized inner-
loop targets. This limits diagnosis-label feedback into architecture selection.

## 9. Model-comparison statistics

For each candidate versus its immediate predecessor:

- report paired per-subject out-of-fold cost differences;
- report medians and bootstrap 95% confidence intervals;
- report every outer fold separately;
- bootstrap subjects, not feature coordinates;
- report group-effect correlations with bootstrap intervals; and
- show predicted versus empirical effect norms.

Statistical intervals support interpretation but do not replace the declared
eligibility gates.

## 10. Parameter and connectome policy

- The common Schaefer-200 connectome remains fixed in M5.2.
- No per-edge or network structural weight is fitted from EEG.
- A new parameter must pass perturbation sensitivity and synthetic recovery
  before empirical interpretation.
- Parameters at a search boundary trigger a range/sensitivity audit; ranges are
  not widened automatically.
- Effective-coupling parameters, if later introduced, must not be described as
  measured tract changes.

## 11. Forward-model policy

The FWD candidate should use a standard template anatomy because personal MRI
is unavailable. It must:

- use the versioned TDBRAIN Table 3 coordinates;
- document the electrode-to-head coordinate transform;
- use distributed cortical vertices rather than one point centroid per parcel;
- aggregate the vertex lead field to the exact Schaefer-200 order;
- record BEM conductivities and geometry hashes;
- preserve average reference; and
- provide a direct audit against the analytic-sphere gain.

An EEG BEM and a TMS FEM solve different problems. MNE-compatible BEM is the
first observation-model improvement. SimNIBS FEM remains the later TMS
electric-field tool.

## 12. Correlated-background policy

The BKG candidate may add a low-rank positive-semidefinite covariance, but:

- its basis is learned only from outer-training residuals;
- its rank and strength are chosen only by inner folds;
- it is shared across diagnosis labels;
- it may not use outer-validation residuals;
- its spatial pattern and explained covariance are saved; and
- improvements must persist in neural-plus-observation posterior predictions,
  not only in an unconstrained empirical reconstruction.

## 13. Stopping rules

Stop model expansion and report if:

- the template forward model does not improve either spatial endpoint;
- correlated background improves total cost but further suppresses the
  connectivity group effect;
- a new dynamics parameter fails recovery;
- results reverse across folds or seeds;
- the only apparent improvement requires reading the consumed M5.1 holdout; or
- computational cost grows without a prespecified scientific gain.

Proceed to FEM/TMS integration only after the M5.2 eligibility gates pass in a
new locked or external evaluation.

## 14. Reproducibility

Fixed values:

- M5.1 subject split seed: `20260913`;
- M5.2 nested split seed: `20260918`;
- outer folds: 5;
- inner folds: 4;
- frequency range: 2–40 Hz unless a protocol amendment states otherwise;
- empirical temporal halves: unchanged; and
- diagnosis labels: excluded from individual fitting.

Any protocol change must be committed before its affected result is evaluated.
