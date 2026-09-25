# M5.3 frozen model and external-test plan

**Frozen:** 2026-09-25, before any external recording was preprocessed or looked at.
**Code:** branch `claude/audit-linear-regime`, commit `80c2d53` (the external-mode switch added afterwards
in `scripts/linear/plans.py` does not change nested results).

## Model (from `docs/INDEPENDENT_AUDIT_2026-09-24.md` §5.4)

Stable-regime analytic cross-spectrum of the delayed dual Jansen–Rit network, plus:
- a shared delayed alpha drive: origin (0, −15, 60) mm, 3 mm/ms, share fitted per subject;
- 14 network × hemisphere input-noise log-gains;
- a lead-field-projected aperiodic source background (fraction and exponent);
- diagonal observation noise (fraction and exponent).

| Item | Frozen value |
|---|---|
| population fits | `configs/m53_frozen/population_fit_{tvb,bem}.json` (the Kaggle final run) |
| Sobol bank | `build_analytic_bank.py --samples 1600 --half-width 1.2` (seed 7) |
| adaptive bank | `--samples 1200 --half-width 0.4 --seed 11 --adaptive-from configs/m53_frozen/stage1_subject_fits_<lead>.csv` |
| certification | node abscissa < −1 s⁻¹, residual < 1e-4 (float32), small gain < 1, fold margin > 0.3 |
| subject fit | `fit_subjects_nested.py --spatial network_hemisphere --common-drive --source-background` (top 24, 150 Adam steps, lr 0.08, prior 0.02) |
| refinement | `refine_subjects.py --common-drive --source-background --steps 40` (lr 0.04, prior 0.02, global prior 0.01) |
| primary lead field | TVB analytic; the BEM is reported as secondary |
| precision | float32 on a Kaggle T4 (`MDD_TVB_JAX_X64=0`) |

No setting may change after this point for the external test.

## External test (pre-specified)

**Cohort.** TDBRAIN `Dataset == MDD-rTMS`, session 1, eyes-closed rest (164 patients). No subject overlaps with
the 262 development subjects or the consumed M5.1 holdout. They are preprocessed with the original EEGLAB
script, unchanged except for the group selection (`scripts/external/preprocess_tdbrain_rtms_restEC.m`).
Spectra come from the unchanged M5 extraction (`scripts/external/extract_external_spectra.py`).

**Scoring.** One transformer and pooled null learned on all 262 development subjects' first halves. Each
external subject's first half is fitted and its second half scored with the M5.1 metric (total and the three
blocks, cost / pooled null). Treatment-outcome labels are not read by any fitting step.

**Comparator.** The M5.1 production bank posterior, with its temperature calibrated on the same 262 development
subjects (`scripts/external/run_m51_external.py`).

**Primary outcome.** Paired per-subject difference in the unseen total ratio (M5.3 TVB refined − M5.1), median with
a 95 % bootstrap CI. **Prediction:** negative (M5.3 better), with the largest gain on alpha topography.
**Secondary outcomes.** The three block ratios; the fraction of subjects beating the null; the BEM variant; the
fitted diagonal observation-noise fraction.
