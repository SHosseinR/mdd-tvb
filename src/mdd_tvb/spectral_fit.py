"""Label-blind approximate posterior fitting of complex EEG cross spectra."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spectral_bank import SpectralSimulationBank, load_spectral_simulation_bank
from .spectral_config import SpectralM5Config
from .spectral_features import (
    CrossSpectralCollection,
    SpectralFeatureTransformer,
    add_diagonal_observation_noise,
    fit_spectral_transformer,
    load_cross_spectral_collection,
    save_spectral_transformer,
)
from .spectral_parameterization import PARAMETER_NAMES, normalized_spectral_parameters


def _stratified_split(
    groups: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    holdout: list[int] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        holdout.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(sorted(train)), np.asarray(sorted(holdout))


def _softmax(log_weight: np.ndarray) -> np.ndarray:
    shifted = log_weight - np.max(log_weight)
    weight = np.exp(shifted)
    return weight / weight.sum()


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    selected_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(probability, cumulative, selected_values))


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).ravel()
    y = np.asarray(second, dtype=float).ravel()
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _channel_log_power(csd: np.ndarray) -> np.ndarray:
    matrices = np.asarray(csd)
    diagonal = np.maximum(
        np.real(np.diagonal(matrices, axis1=-2, axis2=-1)),
        np.finfo(float).tiny,
    )
    values = np.log(diagonal)
    if values.ndim == 2:
        values -= values.mean()
    else:
        values -= values.mean(axis=(-2, -1), keepdims=True)
    return values


def _complex_coherency(csd: np.ndarray) -> np.ndarray:
    matrices = np.asarray(csd)
    diagonal = np.maximum(
        np.real(np.diagonal(matrices, axis1=-2, axis2=-1)),
        np.finfo(float).tiny,
    )
    denominator = np.sqrt(
        diagonal[..., :, np.newaxis] * diagonal[..., np.newaxis, :]
    )
    coherency = matrices / denominator
    upper = np.triu_indices(matrices.shape[-1], k=1)
    selected = coherency[..., upper[0], upper[1]]
    return np.concatenate((np.real(selected), np.imag(selected)), axis=-1)


def _expand_bank(
    config: SpectralM5Config,
    transformer: SpectralFeatureTransformer,
    bank: SpectralSimulationBank,
) -> dict[str, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    csd_rows: list[np.ndarray] = []
    candidate_indices: list[int] = []
    fractions: list[float] = []
    exponents: list[float] = []
    for candidate_index in range(len(bank.parameters)):
        for fraction in config.posterior.observation_noise_fractions:
            for exponent in config.posterior.observation_noise_exponents:
                adjusted = add_diagonal_observation_noise(
                    bank.csd_replicates[candidate_index],
                    bank.frequency_hz,
                    fraction,
                    exponent,
                )
                feature_rows.append(transformer.transform(adjusted))
                csd_rows.append(adjusted.mean(axis=0))
                candidate_indices.append(candidate_index)
                fractions.append(fraction)
                exponents.append(exponent)
    return {
        "features": np.stack(feature_rows),
        "csd": np.stack(csd_rows),
        "candidate_index": np.asarray(candidate_indices, dtype=int),
        "observation_noise_fraction": np.asarray(fractions),
        "observation_noise_exponent": np.asarray(exponents),
    }


def _group_difference(
    values: np.ndarray, groups: np.ndarray, first: str, second: str
) -> np.ndarray:
    return values[groups == second].mean(axis=0) - values[groups == first].mean(axis=0)


def _plot_validation(
    path: Path,
    fitting: CrossSpectralCollection,
    validation: CrossSpectralCollection,
    predictions: np.ndarray,
    table: pd.DataFrame,
) -> None:
    groups = list(dict.fromkeys(fitting.groups.astype(str).tolist()))
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    frequency = fitting.frequency_hz
    fit_power = _channel_log_power(fitting.csd).mean(axis=2)
    validation_power = _channel_log_power(validation.csd).mean(axis=2)
    prediction_power = _channel_log_power(predictions).mean(axis=2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for group in groups:
        mask = fitting.groups == group
        axes[0, 0].plot(frequency, fit_power[mask].mean(axis=0), color=colors[group], label=f"{group} empirical fit")
        axes[0, 0].plot(frequency, prediction_power[mask].mean(axis=0), color=colors[group], linestyle="--", label=f"{group} posterior prediction")
        axes[0, 1].plot(frequency, validation_power[mask].mean(axis=0), color=colors[group], label=f"{group} empirical unseen")
        axes[0, 1].plot(frequency, prediction_power[mask].mean(axis=0), color=colors[group], linestyle="--", label=f"{group} same prediction")
        selected = table[table["group"] == group]
        axes[1, 0].scatter(
            selected["fit_cost_ratio_to_pooled_null"],
            selected["validation_cost_ratio_to_pooled_null"],
            s=18,
            alpha=0.65,
            color=colors[group],
            label=group,
        )
    axes[0, 0].set_title("First half: channel-resolved spectral fit")
    axes[0, 1].set_title("Second half: genuinely unseen within-subject EEG")
    for axis in axes[0]:
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Centered mean log power")
        axis.legend(fontsize=8)
    axes[1, 0].axhline(1.0, color="black", linestyle=":")
    axes[1, 0].axvline(1.0, color="black", linestyle=":")
    axes[1, 0].set_xlabel("Fit cost / pooled empirical null")
    axes[1, 0].set_ylabel("Unseen cost / pooled empirical null")
    axes[1, 0].set_title("Each point is one independently fitted subject")
    axes[1, 0].legend()

    if len(groups) == 2:
        actual = _group_difference(validation_power, validation.groups, groups[0], groups[1])
        predicted = _group_difference(prediction_power, fitting.groups, groups[0], groups[1])
        axes[1, 1].scatter(actual.ravel(), predicted.ravel(), s=10, alpha=0.5)
        correlation = _safe_correlation(actual, predicted)
        axes[1, 1].set_title(f"Unseen group spectral effect preservation: r={correlation:.3f}")
        axes[1, 1].set_xlabel(f"Empirical {groups[1]} - {groups[0]}")
        axes[1, 1].set_ylabel("Posterior-predicted difference")
    else:
        axes[1, 1].axis("off")
    fig.suptitle("M5 spectral posterior validation (group labels used only after fitting)", fontsize=14)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=170)
    plt.close(fig)
    temporary.replace(path)


def _plot_group_effects(
    path: Path,
    validation: CrossSpectralCollection,
    predictions: np.ndarray,
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    selected_indices = (
        np.arange(len(validation.subject_ids)) if indices is None else np.asarray(indices)
    )
    selected_groups = validation.groups[selected_indices]
    selected_predictions = predictions[selected_indices]
    selected_empirical = validation.csd[selected_indices]
    groups = list(dict.fromkeys(selected_groups.astype(str).tolist()))
    if len(groups) != 2:
        return {}
    first, second = groups
    empirical_power = _channel_log_power(selected_empirical)
    predicted_power = _channel_log_power(selected_predictions)
    empirical_effect = _group_difference(empirical_power, selected_groups, first, second)
    predicted_effect = _group_difference(predicted_power, selected_groups, first, second)
    empirical_coherency = _complex_coherency(selected_empirical)
    predicted_coherency = _complex_coherency(selected_predictions)
    empirical_cross_effect = _group_difference(empirical_coherency, selected_groups, first, second)
    predicted_cross_effect = _group_difference(predicted_coherency, selected_groups, first, second)
    alpha = (validation.frequency_hz >= 8.0) & (validation.frequency_hz < 13.0)
    empirical_alpha_subject = empirical_power[:, alpha].mean(axis=1)
    predicted_alpha_subject = predicted_power[:, alpha].mean(axis=1)
    empirical_alpha_subject -= empirical_alpha_subject.mean(axis=1, keepdims=True)
    predicted_alpha_subject -= predicted_alpha_subject.mean(axis=1, keepdims=True)
    empirical_alpha = _group_difference(
        empirical_alpha_subject, selected_groups, first, second
    )
    predicted_alpha = _group_difference(
        predicted_alpha_subject, selected_groups, first, second
    )
    metrics = {
        "channel_frequency_log_power_effect_correlation": _safe_correlation(empirical_effect, predicted_effect),
        "alpha_topography_effect_correlation": _safe_correlation(empirical_alpha, predicted_alpha),
        "complex_coherency_effect_correlation": _safe_correlation(empirical_cross_effect, predicted_cross_effect),
        "channel_frequency_log_power_effect_norm_retained": float(np.linalg.norm(predicted_effect) / max(np.linalg.norm(empirical_effect), np.finfo(float).tiny)),
        "complex_coherency_effect_norm_retained": float(np.linalg.norm(predicted_cross_effect) / max(np.linalg.norm(empirical_cross_effect), np.finfo(float).tiny)),
        "alpha_topography_effect_norm_retained": float(np.linalg.norm(predicted_alpha) / max(np.linalg.norm(empirical_alpha), np.finfo(float).tiny)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].scatter(empirical_effect.ravel(), predicted_effect.ravel(), s=9, alpha=0.45)
    axes[0].set_title(f"Channel x frequency power effect\nr={metrics['channel_frequency_log_power_effect_correlation']:.3f}")
    axes[0].set_xlabel(f"Empirical {second} - {first}")
    axes[0].set_ylabel("Posterior prediction")
    channels = np.arange(validation.channel_names.size)
    axes[1].plot(channels, empirical_alpha, marker="o", label="empirical unseen")
    axes[1].plot(channels, predicted_alpha, marker="o", label="posterior prediction")
    axes[1].set_xticks(channels)
    axes[1].set_xticklabels(validation.channel_names, rotation=90, fontsize=7)
    axes[1].set_title(f"Alpha topography effect\nr={metrics['alpha_topography_effect_correlation']:.3f}")
    axes[1].legend(fontsize=8)
    axes[2].scatter(empirical_cross_effect.ravel(), predicted_cross_effect.ravel(), s=7, alpha=0.35)
    axes[2].set_title(f"Complex coherency effect\nr={metrics['complex_coherency_effect_correlation']:.3f}")
    axes[2].set_xlabel("Empirical unseen effect")
    axes[2].set_ylabel("Posterior prediction")
    fig.suptitle("Subject-holdout group-effect preservation—not diagnosis prediction", fontsize=14)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=170)
    plt.close(fig)
    temporary.replace(path)
    return metrics


def fit_spectral_subjects(
    config: SpectralM5Config,
    fitting: CrossSpectralCollection | None = None,
    validation: CrossSpectralCollection | None = None,
    bank: SpectralSimulationBank | None = None,
) -> pd.DataFrame:
    empirical_dir = config.paths.output_dir / "empirical"
    bank_dir = config.paths.output_dir / "bank"
    fitting = fitting or load_cross_spectral_collection(empirical_dir / "cross_spectra_fit.npz")
    validation = validation or load_cross_spectral_collection(empirical_dir / "cross_spectra_validation.npz")
    bank = bank or load_spectral_simulation_bank(bank_dir / "spectral_simulation_bank.npz")
    reliability_first = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_a.npz"
    )
    reliability_second = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_b.npz"
    )
    if not np.array_equal(fitting.subject_ids, validation.subject_ids):
        raise ValueError("Fitting and validation subject order differs")
    if not np.array_equal(fitting.frequency_hz, bank.frequency_hz):
        raise ValueError("Empirical and simulated frequency grids differ")
    if not np.array_equal(fitting.channel_names, bank.channel_names):
        raise ValueError("Empirical and simulated channel orders differ")
    train_indices, holdout_indices = _stratified_split(
        fitting.groups, config.spectral.holdout_fraction, config.spectral.split_seed
    )
    transformer = fit_spectral_transformer(
        fitting,
        train_indices,
        config.spectral,
        reliability_first,
        reliability_second,
    )
    fit_dir = config.paths.output_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    save_spectral_transformer(fit_dir / "spectral_transformer.npz", transformer)
    fit_features = transformer.transform(fitting.csd)
    validation_features = transformer.transform(validation.csd)
    expanded = _expand_bank(config, transformer, bank)
    state_features = expanded["features"]
    state_mean = state_features.mean(axis=1)
    empirical_repeat_cost = np.sum(
        (fit_features[train_indices] - validation_features[train_indices]) ** 2,
        axis=1,
    )
    temperature = max(
        config.posterior.minimum_temperature,
        float(np.median(empirical_repeat_cost)),
    )
    pooled_feature = fit_features[train_indices].mean(axis=0)
    normalized = normalized_spectral_parameters(bank.parameters, config.design)
    neural_state_parameters = normalized[expanded["candidate_index"]]
    prior_cost = config.posterior.prior_strength * np.mean(neural_state_parameters**2, axis=1)

    population_cost = np.sum((state_mean - pooled_feature) ** 2, axis=1)
    population_order = np.argsort(population_cost)
    population_rows: list[dict[str, Any]] = []
    for rank, state in enumerate(population_order, start=1):
        candidate_index = int(expanded["candidate_index"][state])
        row: dict[str, Any] = {
            "rank": rank,
            "candidate_index": candidate_index,
            "pooled_training_feature_cost": float(population_cost[state]),
            "observation_noise_fraction": float(expanded["observation_noise_fraction"][state]),
            "observation_noise_exponent": float(expanded["observation_noise_exponent"][state]),
        }
        row.update(
            {
                name: float(bank.parameters[candidate_index, column])
                for column, name in enumerate(PARAMETER_NAMES)
            }
        )
        population_rows.append(row)

    predictions: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    top_weight_rows: list[dict[str, Any]] = []
    holdout_set = set(holdout_indices.tolist())
    for subject_index, subject_id in enumerate(fitting.subject_ids.astype(str)):
        residual = state_features - fit_features[subject_index][np.newaxis, np.newaxis, :]
        cost = np.mean(np.sum(residual**2, axis=2), axis=1)
        weight = _softmax(-0.5 * (cost / temperature + prior_cost))
        map_state = int(np.argmax(weight))
        map_candidate = int(expanded["candidate_index"][map_state])
        prediction_feature = weight @ state_mean
        prediction_csd = np.einsum("s,sfcd->fcd", weight, expanded["csd"], optimize=True)
        predictions.append(prediction_csd)
        fit_cost = float(np.sum((prediction_feature - fit_features[subject_index]) ** 2))
        validation_cost = float(np.sum((prediction_feature - validation_features[subject_index]) ** 2))
        map_fit_cost = float(
            np.sum((state_mean[map_state] - fit_features[subject_index]) ** 2)
        )
        map_validation_cost = float(
            np.sum((state_mean[map_state] - validation_features[subject_index]) ** 2)
        )
        oracle_validation_cost = float(
            np.min(np.sum((state_mean - validation_features[subject_index]) ** 2, axis=1))
        )
        fit_null = float(np.sum((pooled_feature - fit_features[subject_index]) ** 2))
        validation_null = float(np.sum((pooled_feature - validation_features[subject_index]) ** 2))
        temporal_persistence_cost = float(
            np.sum((fit_features[subject_index] - validation_features[subject_index]) ** 2)
        )
        candidate_weight = np.bincount(
            expanded["candidate_index"], weights=weight, minlength=len(bank.parameters)
        )
        record: dict[str, Any] = {
            "subject_id": subject_id,
            "group": str(fitting.groups[subject_index]),
            "subject_split": "holdout" if subject_index in holdout_set else "train",
            "map_candidate_index": map_candidate,
            "posterior_effective_sample_size_states": float(1.0 / np.sum(weight**2)),
            "posterior_effective_sample_size_candidates": float(1.0 / np.sum(candidate_weight**2)),
            "posterior_map_probability": float(weight[map_state]),
            "fit_posterior_predictive_cost": fit_cost,
            "validation_posterior_predictive_cost": validation_cost,
            "map_fit_cost": map_fit_cost,
            "map_validation_cost": map_validation_cost,
            "oracle_validation_cost_not_a_fit": oracle_validation_cost,
            "fit_pooled_null_cost": fit_null,
            "validation_pooled_null_cost": validation_null,
            "temporal_persistence_validation_cost": temporal_persistence_cost,
            "fit_cost_ratio_to_pooled_null": fit_cost / fit_null if fit_null > 0 else np.nan,
            "validation_cost_ratio_to_pooled_null": validation_cost / validation_null if validation_null > 0 else np.nan,
            "map_fit_cost_ratio_to_pooled_null": map_fit_cost / fit_null if fit_null > 0 else np.nan,
            "map_validation_cost_ratio_to_pooled_null": map_validation_cost / validation_null if validation_null > 0 else np.nan,
            "oracle_validation_cost_ratio_to_pooled_null_not_a_fit": oracle_validation_cost / validation_null if validation_null > 0 else np.nan,
            "temporal_persistence_cost_ratio_to_pooled_null": temporal_persistence_cost / validation_null if validation_null > 0 else np.nan,
            "observation_noise_fraction_posterior_mean": float(weight @ expanded["observation_noise_fraction"]),
            "observation_noise_exponent_posterior_mean": float(weight @ expanded["observation_noise_exponent"]),
        }
        state_parameter_values = bank.parameters[expanded["candidate_index"]]
        for column, name in enumerate(PARAMETER_NAMES):
            values = state_parameter_values[:, column]
            record[f"{name}_posterior_mean"] = float(weight @ values)
            record[f"{name}_posterior_q05"] = _weighted_quantile(values, weight, 0.05)
            record[f"{name}_posterior_q95"] = _weighted_quantile(values, weight, 0.95)
            record[f"{name}_map"] = float(bank.parameters[map_candidate, column])
        records.append(record)
        for state in np.argsort(weight)[-10:][::-1]:
            top_weight_rows.append(
                {
                    "subject_id": subject_id,
                    "state_rank": len([row for row in top_weight_rows[-10:] if row["subject_id"] == subject_id]) + 1,
                    "candidate_index": int(expanded["candidate_index"][state]),
                    "observation_noise_fraction": float(expanded["observation_noise_fraction"][state]),
                    "observation_noise_exponent": float(expanded["observation_noise_exponent"][state]),
                    "posterior_weight": float(weight[state]),
                }
            )
    prediction_array = np.stack(predictions)
    table = pd.DataFrame(records)
    fit_auto, fit_cross = transformer.transform_blocks(fitting.csd)
    validation_auto, validation_cross = transformer.transform_blocks(validation.csd)
    prediction_auto, prediction_cross = transformer.transform_blocks(prediction_array)
    for name, fit_block, validation_block, prediction_block in (
        ("auto_spectrum", fit_auto, validation_auto, prediction_auto),
        ("complex_coherency", fit_cross, validation_cross, prediction_cross),
    ):
        pooled_block = fit_block[train_indices].mean(axis=0)
        numerator = np.mean((prediction_block - validation_block) ** 2, axis=1)
        denominator = np.mean((pooled_block - validation_block) ** 2, axis=1)
        table[f"validation_{name}_cost_ratio_to_pooled_null"] = np.divide(
            numerator,
            denominator,
            out=np.full(len(table), np.nan),
            where=denominator > 0,
        )
    table.to_csv(fit_dir / "subject_posteriors.csv", index=False)
    pd.DataFrame(population_rows).to_csv(
        fit_dir / "population_calibration_ranking.csv", index=False
    )
    pd.DataFrame(top_weight_rows).to_csv(fit_dir / "subject_top_posterior_states.csv", index=False)
    np.savez_compressed(
        fit_dir / "posterior_predictive_csd.npz",
        subject_ids=fitting.subject_ids,
        frequency_hz=fitting.frequency_hz,
        channel_names=fitting.channel_names,
        csd=prediction_array,
    )
    pd.DataFrame(
        {
            "subject_id": fitting.subject_ids,
            "group": fitting.groups,
            "subject_split": [
                "holdout" if index in holdout_set else "train"
                for index in range(len(fitting.subject_ids))
            ],
        }
    ).to_csv(fit_dir / "data_split.csv", index=False)

    group_rows: list[dict[str, Any]] = []
    for group in config.empirical.groups:
        selected = table[table["group"] == group]
        for name in PARAMETER_NAMES:
            values = selected[f"{name}_posterior_mean"].to_numpy()
            group_rows.append(
                {
                    "group": group,
                    "parameter": name,
                    "mean_of_subject_posterior_means": float(values.mean()),
                    "standard_deviation_across_subjects": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                }
            )
    pd.DataFrame(group_rows).to_csv(fit_dir / "descriptive_group_parameter_summary.csv", index=False)
    _plot_validation(
        fit_dir / "m5_spectral_validation.png",
        fitting,
        validation,
        prediction_array,
        table,
    )
    group_effect_metrics = _plot_group_effects(
        fit_dir / "m5_spectral_group_effects.png",
        validation,
        prediction_array,
        holdout_indices,
    )
    split_summary: dict[str, Any] = {}
    for split in ("train", "holdout"):
        selected = table[table["subject_split"] == split]
        split_summary[split] = {
            "n": int(len(selected)),
            "median_fit_cost_ratio_to_pooled_null": float(selected["fit_cost_ratio_to_pooled_null"].median()),
            "median_validation_cost_ratio_to_pooled_null": float(selected["validation_cost_ratio_to_pooled_null"].median()),
            "fraction_beating_pooled_null_on_validation": float(np.mean(selected["validation_cost_ratio_to_pooled_null"] < 1.0)),
            "median_map_validation_cost_ratio_to_pooled_null": float(selected["map_validation_cost_ratio_to_pooled_null"].median()),
            "median_oracle_validation_cost_ratio_to_pooled_null_not_a_fit": float(selected["oracle_validation_cost_ratio_to_pooled_null_not_a_fit"].median()),
            "median_temporal_persistence_cost_ratio_to_pooled_null": float(selected["temporal_persistence_cost_ratio_to_pooled_null"].median()),
            "median_candidate_posterior_ess": float(selected["posterior_effective_sample_size_candidates"].median()),
            "median_auto_spectrum_validation_cost_ratio_to_pooled_null": float(selected["validation_auto_spectrum_cost_ratio_to_pooled_null"].median()),
            "median_complex_coherency_validation_cost_ratio_to_pooled_null": float(selected["validation_complex_coherency_cost_ratio_to_pooled_null"].median()),
        }

    def json_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    summary = {
        "status": "pilot" if len(bank.parameters) < 64 or config.design.replicates < 3 else "production-scale bank completed",
        "method": "label-blind discrete simulation-bank approximate posterior with observation-noise marginalization",
        "neural_model": "two locally coupled Jansen-Rit generators per Schaefer parcel; shared delayed TVB structural network",
        "objective": {
            "data": "channel-resolved 2-40 Hz complex cross-spectral density",
            "sensor_reduction": f"{config.spectral.sensor_modes} label-blind spatial modes learned from training subjects",
            "summary_reduction": {
                "periodic_residual_plus_aperiodic_exponent_coordinates": int(
                    transformer.auto_components.shape[0]
                    if transformer.auto_components.shape[0]
                    else transformer.auto_mean.size
                ),
                "complex_coherency_coordinates": int(
                    transformer.cross_components.shape[0]
                    if transformer.cross_components.shape[0]
                    else transformer.cross_mean.size
                ),
                "reliability_calibration": (
                    "two non-overlapping quarters inside the fitting half; "
                    "the final half remains unseen"
                ),
            },
            "periodic_aperiodic": "per-sensor-mode log-linear background from 2-7 and 30-40 Hz; fit retains residual spectra and explicit exponents",
            "absolute_amplitude": "per-sensor-mode intercepts removed as observation-gain nuisances",
            "dc": "excluded by demeaning and the 2-Hz lower fit bound",
            "structural_weights": "fixed after the v2 identifiability failure",
        },
        "posterior": {
            "kind": "finite-bank kernel posterior, not an exact posterior and not yet amortized SBI",
            "temperature_from_median_train_subject_fit_vs_unseen_discrepancy": temperature,
            "observation_noise_grid_size": int(
                len(config.posterior.observation_noise_fractions)
                * len(config.posterior.observation_noise_exponents)
            ),
        },
        "population_calibration": {
            "target": "label-blind pooled fitting-half spectrum from training subjects",
            "best_state": population_rows[0],
            "best_observation_noise_fraction_at_grid_maximum": bool(
                population_rows[0]["observation_noise_fraction"]
                == max(config.posterior.observation_noise_fractions)
            ),
            "interpretation": (
                "A boundary-saturating observation nuisance indicates unresolved "
                "neural/forward-model mismatch; do not expand the subject bank yet."
            ),
        },
        "simulation": {
            "candidates": int(len(bank.parameters)),
            "replicates": int(bank.csd_replicates.shape[1]),
            "analyzed_seconds_per_replicate": float((config.design.duration_ms - config.design.transient_ms) / 1000.0),
            "common_random_numbers_across_candidates": True,
        },
        "validation": split_summary,
        "subject_holdout_unseen_group_effect_preservation": group_effect_metrics,
        "guardrails": [
            "Diagnosis labels never enter subject fitting or feature reduction.",
            "Group summaries are descriptive second-level outputs, not group-level fits.",
            "The pooled empirical mean is the explicit null model.",
            "Stimulation optimization remains blocked until unseen spatial group effects pass.",
        ],
        "config": asdict(config),
    }
    (fit_dir / "fit_summary.json").write_text(
        json.dumps(json_value(summary), indent=2), encoding="utf-8"
    )
    return table
