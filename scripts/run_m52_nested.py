"""Run fixed-model M5.2 outer-fold evaluation on development subjects only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import (
    CrossSpectralCollection,
    load_cross_spectral_collection,
    subset_cross_spectral_collection,
)
from mdd_tvb.spectral_fit import _plot_group_effects, fit_spectral_subjects
from mdd_tvb.spectral_parameterization import PARAMETER_NAMES


def _load_collections(empirical_dir: Path) -> tuple[CrossSpectralCollection, ...]:
    return tuple(
        load_cross_spectral_collection(empirical_dir / name)
        for name in (
            "cross_spectra_fit.npz",
            "cross_spectra_validation.npz",
            "cross_spectra_reliability_a.npz",
            "cross_spectra_reliability_b.npz",
        )
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--empirical-dir",
        type=Path,
        default=Path("outputs/m5_spectral_m51_production/empirical"),
    )
    parser.add_argument(
        "--split-file", type=Path, default=Path("configs/m52_nested_splits.csv")
    )
    parser.add_argument("--bank-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    config = load_spectral_m5_config(resolved(args.config))
    empirical_dir = resolved(args.empirical_dir)
    split_table = pd.read_csv(resolved(args.split_file))
    output_dir = (
        resolved(args.output_dir)
        if args.output_dir is not None
        else config.paths.output_dir / "nested"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_file = (
        resolved(args.bank_file)
        if args.bank_file is not None
        else config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    bank = load_spectral_simulation_bank(bank_file)
    collections = _load_collections(empirical_dir)
    fitting_full = collections[0]

    development_ids = set(split_table.subject_id.astype(str))
    selected = np.asarray(
        [
            index
            for index, subject_id in enumerate(fitting_full.subject_ids.astype(str))
            if subject_id in development_ids
        ],
        dtype=int,
    )
    if len(selected) != 262:
        raise ValueError(f"Expected 262 M5.2 development subjects, found {len(selected)}")
    selected_collections = tuple(
        subset_cross_spectral_collection(collection, selected)
        for collection in collections
    )
    fitting, validation, reliability_first, reliability_second = selected_collections
    subject_to_index = {
        subject_id: index
        for index, subject_id in enumerate(fitting.subject_ids.astype(str))
    }

    folds = sorted(split_table.outer_fold.unique().astype(int).tolist())
    if args.outer_folds:
        folds = [fold for fold in folds if fold in set(args.outer_folds)]
    fold_summaries: list[dict[str, Any]] = []
    out_of_fold_tables: list[pd.DataFrame] = []
    prediction_by_subject: dict[str, np.ndarray] = {}
    for fold in folds:
        fold_rows = split_table[split_table.outer_fold == fold]
        training_ids = fold_rows.loc[
            fold_rows.outer_role == "training", "subject_id"
        ].astype(str)
        validation_ids = fold_rows.loc[
            fold_rows.outer_role == "validation", "subject_id"
        ].astype(str)
        train_indices = np.asarray(
            [subject_to_index[value] for value in training_ids], dtype=int
        )
        holdout_indices = np.asarray(
            [subject_to_index[value] for value in validation_ids], dtype=int
        )
        fold_dir = output_dir / f"outer_{fold}"
        table = fit_spectral_subjects(
            config,
            fitting,
            validation,
            bank,
            reliability_first,
            reliability_second,
            split_indices=(train_indices, holdout_indices),
            fit_directory=fold_dir,
        )
        holdout = table[table.subject_split == "holdout"].copy()
        holdout.insert(0, "outer_fold", fold)
        out_of_fold_tables.append(holdout)
        with np.load(fold_dir / "posterior_predictive_csd.npz") as payload:
            prediction_ids = payload["subject_ids"].astype(str)
            predictions = payload["csd"]
        for subject_id in validation_ids:
            index = int(np.flatnonzero(prediction_ids == subject_id)[0])
            prediction_by_subject[subject_id] = predictions[index]
        summary = json.loads(
            (fold_dir / "fit_summary.json").read_text(encoding="utf-8")
        )
        fold_summaries.append(
            {
                "outer_fold": fold,
                "validation": summary["validation"]["holdout"],
                "group_effects": summary[
                    "subject_holdout_unseen_group_effect_preservation"
                ],
                "population_calibration": summary["population_calibration"],
                "synthetic_recovery": summary["synthetic_recovery"],
            }
        )
        print(f"Completed M5.2 outer fold {fold}", flush=True)

    out_of_fold = pd.concat(out_of_fold_tables, ignore_index=True)
    out_of_fold.to_csv(output_dir / "out_of_fold_subject_posteriors.csv", index=False)
    if len(prediction_by_subject) != len(out_of_fold):
        raise RuntimeError("Out-of-fold predictions do not cover every evaluated subject")
    ordered_ids = fitting.subject_ids.astype(str)
    evaluated = np.asarray(
        [index for index, value in enumerate(ordered_ids) if value in prediction_by_subject]
    )
    prediction_array = np.stack([prediction_by_subject[ordered_ids[index]] for index in evaluated])
    np.savez_compressed(
        output_dir / "out_of_fold_posterior_predictive_csd.npz",
        subject_ids=ordered_ids[evaluated],
        groups=fitting.groups[evaluated],
        frequency_hz=fitting.frequency_hz,
        channel_names=fitting.channel_names,
        csd=prediction_array,
    )
    group_effects = _plot_group_effects(
        output_dir / "nested_group_effects.png",
        subset_cross_spectral_collection(validation, evaluated),
        prediction_array,
        cross_metric=config.spectral.cross_metric,
    )

    active_names = PARAMETER_NAMES if config.design.fit_structural_modes else PARAMETER_NAMES[:-2]
    minimum_recovery = min(
        float(fold["synthetic_recovery"]["parameter_recovery_correlations"][name])
        for fold in fold_summaries
        for name in active_names
    )
    total = float(out_of_fold.validation_cost_ratio_to_pooled_null.median())
    fraction = float(np.mean(out_of_fold.validation_cost_ratio_to_pooled_null < 1.0))
    auto = float(
        out_of_fold.validation_auto_spectrum_cost_ratio_to_pooled_null.median()
    )
    connectivity = float(
        out_of_fold.validation_complex_coherency_cost_ratio_to_pooled_null.median()
    )
    topography = float(
        out_of_fold.validation_alpha_topography_cost_ratio_to_pooled_null.median()
    )
    nuisance_interior = all(
        not fold["population_calibration"][
            "best_observation_noise_fraction_at_grid_maximum"
        ]
        and not fold["population_calibration"][
            "best_source_background_fraction_at_grid_maximum"
        ]
        for fold in fold_summaries
    )
    primary_fold_passes = sum(
        fold["validation"]["median_validation_cost_ratio_to_pooled_null"] < 1.0
        and fold["validation"][
            "median_auto_spectrum_validation_cost_ratio_to_pooled_null"
        ]
        < 1.0
        and fold["validation"][
            "median_selected_connectivity_validation_cost_ratio_to_pooled_null"
        ]
        < 1.0
        and fold["validation"][
            "median_alpha_topography_validation_cost_ratio_to_pooled_null"
        ]
        < 1.0
        for fold in fold_summaries
    )
    gates = {
        "unseen_total_below_null": total < 1.0,
        "majority_of_subjects_beat_null": fraction > 0.5,
        "autospectrum_below_null": auto < 1.0,
        "alpha_topography_below_null": topography < 1.0,
        "selected_connectivity_below_null": connectivity < 1.0,
        "power_group_effect_preserved": group_effects.get(
            "channel_frequency_log_power_effect_correlation", 0.0
        )
        >= 0.30,
        "alpha_topography_group_effect_preserved": group_effects.get(
            "alpha_topography_effect_correlation", 0.0
        )
        >= 0.30,
        "lagged_connectivity_group_effect_preserved": group_effects.get(
            "fitted_connectivity_effect_correlation", 0.0
        )
        >= 0.10,
        "lagged_connectivity_effect_norm_retained": group_effects.get(
            "fitted_connectivity_effect_norm_retained", 0.0
        )
        >= 0.25,
        "all_active_parameters_recoverable": minimum_recovery >= 0.50,
        "population_nuisance_interior_in_all_folds": nuisance_interior,
        "primary_individual_gates_pass_in_four_of_five_folds": primary_fold_passes
        >= 4,
    }
    summary = {
        "status": "eligible_for_new_locked_test" if all(gates.values()) else "not_eligible",
        "development_only": True,
        "consumed_m51_holdout_read": False,
        "folds_completed": folds,
        "subjects": int(len(out_of_fold)),
        "aggregate": {
            "median_unseen_total_cost_ratio": total,
            "fraction_beating_null": fraction,
            "median_auto_cost_ratio": auto,
            "median_selected_connectivity_cost_ratio": connectivity,
            "median_alpha_topography_cost_ratio": topography,
            "minimum_active_parameter_recovery_correlation_across_folds": minimum_recovery,
            "primary_fold_passes": primary_fold_passes,
            "group_effects": group_effects,
        },
        "gates": gates,
        "folds": fold_summaries,
    }
    (output_dir / "nested_summary.json").write_text(
        json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_json_value(summary["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
