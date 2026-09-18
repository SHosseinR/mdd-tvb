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
    add_observation_backgrounds,
    fit_spectral_transformer,
    load_cross_spectral_collection,
    save_spectral_transformer,
)
from .config import load_config
from .connectome import load_connectome
from .eeg import build_eeg_monitor, regularize_analytic_eeg_gain
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


def _posterior_weight(
    data_cost: np.ndarray, prior_cost: np.ndarray, temperature: float
) -> np.ndarray:
    """Return a regularized finite-bank kernel posterior.

    ``prior_cost`` is deliberately part of the penalized objective before
    tempering.  This gives the Gaussian shrinkage terms the same meaning for
    every cross-validated temperature and prevents them from disappearing as
    the kernel becomes sharper.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return _softmax(-0.5 * (np.asarray(data_cost) + prior_cost) / temperature)


def _select_temperature(
    state_features: np.ndarray,
    state_mean: np.ndarray,
    fit_features: np.ndarray,
    validation_features: np.ndarray,
    train_indices: np.ndarray,
    pooled_feature: np.ndarray,
    prior_cost: np.ndarray,
    base_temperature: float,
    multipliers: tuple[float, ...],
    minimum_temperature: float,
) -> tuple[float, pd.DataFrame]:
    """Select one global kernel temperature without touching held-out subjects."""

    temperatures = np.unique(
        np.maximum(
            minimum_temperature,
            base_temperature * np.asarray(multipliers, dtype=float),
        )
    )
    rows: list[dict[str, float]] = []
    for temperature in temperatures:
        ratios: list[float] = []
        for subject_index in train_indices:
            residual = (
                state_features
                - fit_features[subject_index][np.newaxis, np.newaxis, :]
            )
            data_cost = np.mean(np.sum(residual**2, axis=2), axis=1)
            weight = _posterior_weight(data_cost, prior_cost, float(temperature))
            prediction = weight @ state_mean
            numerator = float(
                np.sum((prediction - validation_features[subject_index]) ** 2)
            )
            denominator = float(
                np.sum((pooled_feature - validation_features[subject_index]) ** 2)
            )
            ratios.append(numerator / denominator if denominator > 0.0 else np.nan)
        ratio_array = np.asarray(ratios)
        rows.append(
            {
                "temperature": float(temperature),
                "temperature_multiplier": float(temperature / base_temperature),
                "training_unseen_median_cost_ratio_to_pooled_null": float(
                    np.nanmedian(ratio_array)
                ),
                "training_unseen_mean_cost_ratio_to_pooled_null": float(
                    np.nanmean(ratio_array)
                ),
                "training_unseen_fraction_beating_pooled_null": float(
                    np.nanmean(ratio_array < 1.0)
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        [
            "training_unseen_median_cost_ratio_to_pooled_null",
            "training_unseen_mean_cost_ratio_to_pooled_null",
        ],
        kind="stable",
    )
    return float(table.iloc[0]["temperature"]), table


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
    source_covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    diagonal_rows: list[np.ndarray] = []
    candidate_indices: list[int] = []
    fractions: list[float] = []
    exponents: list[float] = []
    source_fractions: list[float] = []
    source_exponents: list[float] = []
    source_power_rows: list[np.ndarray] = []
    for candidate_index in range(len(bank.parameters)):
        for fraction in config.posterior.observation_noise_fractions:
            for exponent in config.posterior.observation_noise_exponents:
                for source_fraction in config.posterior.source_background_fractions:
                    for source_exponent in config.posterior.source_background_exponents:
                        adjusted, diagonal_increment, source_power = add_observation_backgrounds(
                            bank.csd_replicates[candidate_index],
                            bank.frequency_hz,
                            fraction,
                            exponent,
                            source_fraction,
                            source_exponent,
                            source_covariance,
                        )
                        feature_rows.append(transformer.transform(adjusted))
                        # Compact increments avoid a dense CSD per nuisance state.
                        diagonal_rows.append(diagonal_increment.mean(axis=0))
                        source_power_rows.append(source_power.mean(axis=0))
                        candidate_indices.append(candidate_index)
                        fractions.append(fraction)
                        exponents.append(exponent)
                        source_fractions.append(source_fraction)
                        source_exponents.append(source_exponent)
    return {
        "features": np.stack(feature_rows),
        "candidate_csd": bank.csd_replicates.mean(axis=1),
        "diagonal_noise": np.stack(diagonal_rows),
        "source_power": np.stack(source_power_rows),
        "source_covariance": source_covariance,
        "candidate_index": np.asarray(candidate_indices, dtype=int),
        "observation_noise_fraction": np.asarray(fractions),
        "observation_noise_exponent": np.asarray(exponents),
        "source_background_fraction": np.asarray(source_fractions),
        "source_background_exponent": np.asarray(source_exponents),
    }


def _posterior_csd(weight: np.ndarray, expanded: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct a posterior CSD without expanding every dense state CSD."""

    candidate_weight = np.bincount(
        expanded["candidate_index"],
        weights=weight,
        minlength=len(expanded["candidate_csd"]),
    )
    prediction = np.einsum(
        "c,cfij->fij", candidate_weight, expanded["candidate_csd"], optimize=True
    )
    diagonal = np.einsum(
        "s,sfc->fc", weight, expanded["diagonal_noise"], optimize=True
    )
    indices = np.arange(prediction.shape[-1])
    prediction[..., indices, indices] += diagonal
    source_power = np.einsum(
        "s,sf->f", weight, expanded["source_power"], optimize=True
    )
    prediction += source_power[:, None, None] * expanded["source_covariance"]
    return prediction


def _source_background_covariance(config: SpectralM5Config) -> np.ndarray:
    """Fixed group-blind covariance of independent parcels through the lead field."""

    baseline = load_config(config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    monitor, _ = build_eeg_monitor(
        baseline.monitor,
        baseline.connectivity.expected_regions,
        baseline.simulation.monitor_period_ms,
    )
    monitor.configure()
    regularize_analytic_eeg_gain(
        monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        baseline.monitor.minimum_source_sensor_distance_mm,
    )
    gain = monitor.gain.copy()
    reference = baseline.monitor.reference
    if reference:
        if reference.lower() == "average":
            gain -= gain.mean(axis=0, keepdims=True)
        else:
            gain -= gain[baseline.monitor.channels.index(reference)][None, :]
    covariance = gain @ gain.T
    covariance = 0.5 * (covariance + covariance.T)
    covariance /= np.mean(np.diag(covariance))
    return covariance


def _synthetic_recovery(
    config: SpectralM5Config,
    bank: SpectralSimulationBank,
    expanded: dict[str, np.ndarray],
    prior_cost: np.ndarray,
    temperature: float,
    output_path: Path,
) -> dict[str, Any]:
    """Recover held-out simulation seeds using the remaining bank seeds."""

    features = expanded["features"]
    if features.shape[1] < 2:
        return {"status": "not_run", "reason": "at least two seeds are required"}
    recovery_mean = features[:, 1:].mean(axis=1)
    fractions = expanded["observation_noise_fraction"]
    exponents = expanded["observation_noise_exponent"]
    target_fraction = min(
        config.posterior.observation_noise_fractions, key=lambda value: abs(value - 0.5)
    )
    target_exponent = min(
        config.posterior.observation_noise_exponents, key=lambda value: abs(value - 1.0)
    )
    target_source_fraction = min(
        config.posterior.source_background_fractions,
        key=lambda value: abs(value - 0.5),
    )
    target_source_exponent = min(
        config.posterior.source_background_exponents,
        key=lambda value: abs(value - 1.0),
    )
    normalized = normalized_spectral_parameters(bank.parameters, config.design)
    state_parameters = normalized[expanded["candidate_index"]]
    rows: list[dict[str, Any]] = []
    recovered: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for candidate_index in range(len(bank.parameters)):
        matches = np.flatnonzero(
            (expanded["candidate_index"] == candidate_index)
            & np.isclose(fractions, target_fraction)
            & np.isclose(exponents, target_exponent)
            & np.isclose(
                expanded["source_background_fraction"], target_source_fraction
            )
            & np.isclose(
                expanded["source_background_exponent"], target_source_exponent
            )
        )
        if matches.size != 1:
            raise RuntimeError("Synthetic recovery state is not unique")
        target = features[int(matches[0]), 0]
        cost = np.sum((recovery_mean - target) ** 2, axis=1)
        weight = _posterior_weight(cost, prior_cost, temperature)
        estimate = weight @ state_parameters
        truth = normalized[candidate_index]
        recovered.append(estimate)
        truths.append(truth)
        row: dict[str, Any] = {
            "candidate_index": candidate_index,
            "normalized_parameter_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
            "posterior_effective_sample_size_states": float(1.0 / np.sum(weight**2)),
        }
        for column, name in enumerate(PARAMETER_NAMES):
            row[f"{name}_truth_normalized"] = float(truth[column])
            row[f"{name}_posterior_mean_normalized"] = float(estimate[column])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    truth_matrix = np.stack(truths)
    estimate_matrix = np.stack(recovered)
    correlations = {
        name: _safe_correlation(truth_matrix[:, column], estimate_matrix[:, column])
        for column, name in enumerate(PARAMETER_NAMES)
    }
    return {
        "status": "completed",
        "fit_seed": 0,
        "bank_seeds_used_for_prediction": int(features.shape[1] - 1),
        "observation_noise_fraction": float(target_fraction),
        "observation_noise_exponent": float(target_exponent),
        "source_background_fraction": float(target_source_fraction),
        "source_background_exponent": float(target_source_exponent),
        "median_normalized_parameter_rmse": float(
            np.median(table["normalized_parameter_rmse"])
        ),
        "parameter_recovery_correlations": correlations,
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
    holdout_indices: np.ndarray,
) -> None:
    groups = list(dict.fromkeys(fitting.groups.astype(str).tolist()))
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    holdout_indices = np.asarray(holdout_indices, dtype=int)
    holdout_mask = np.zeros(len(fitting.subject_ids), dtype=bool)
    holdout_mask[holdout_indices] = True
    frequency = fitting.frequency_hz
    fit_power = _channel_log_power(fitting.csd).mean(axis=2)
    validation_power = _channel_log_power(validation.csd).mean(axis=2)
    prediction_power = _channel_log_power(predictions).mean(axis=2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for group in groups:
        mask = (fitting.groups == group) & holdout_mask
        axes[0, 0].plot(frequency, fit_power[mask].mean(axis=0), color=colors[group], label=f"{group} empirical fit")
        axes[0, 0].plot(frequency, prediction_power[mask].mean(axis=0), color=colors[group], linestyle="--", label=f"{group} posterior prediction")
        axes[0, 1].plot(frequency, validation_power[mask].mean(axis=0), color=colors[group], label=f"{group} empirical unseen")
        axes[0, 1].plot(frequency, prediction_power[mask].mean(axis=0), color=colors[group], linestyle="--", label=f"{group} same prediction")
        for split, marker, size, alpha in (
            ("train", "o", 16, 0.40),
            ("holdout", "D", 28, 0.85),
        ):
            selected = table[
                (table["group"] == group) & (table["subject_split"] == split)
            ]
            axes[1, 0].scatter(
                selected["fit_cost_ratio_to_pooled_null"],
                selected["validation_cost_ratio_to_pooled_null"],
                s=size,
                alpha=alpha,
                marker=marker,
                color=colors[group],
                edgecolors="black" if split == "holdout" else "none",
                linewidths=0.4,
                label=f"{group} {split}",
            )
    axes[0, 0].set_title("Subject holdout, first half: individual spectral fit")
    axes[0, 1].set_title(
        "Subject holdout, second half: untouched individual validation"
    )
    for axis in axes[0]:
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Centered mean log power")
        axis.legend(fontsize=8)
    axes[1, 0].axhline(1.0, color="black", linestyle=":")
    axes[1, 0].axvline(1.0, color="black", linestyle=":")
    axes[1, 0].set_xlabel("Fit cost / pooled empirical null")
    axes[1, 0].set_ylabel("Unseen cost / pooled empirical null")
    axes[1, 0].set_title("Each point is one independently fitted subject")
    axes[1, 0].legend(fontsize=8)

    if len(groups) == 2:
        selected_groups = validation.groups[holdout_indices]
        actual = _group_difference(
            validation_power[holdout_indices], selected_groups, groups[0], groups[1]
        )
        predicted = _group_difference(
            prediction_power[holdout_indices], selected_groups, groups[0], groups[1]
        )
        axes[1, 1].scatter(actual.ravel(), predicted.ravel(), s=10, alpha=0.5)
        correlation = _safe_correlation(actual, predicted)
        axes[1, 1].set_title(
            f"Subject-holdout unseen channel-mean spectral effect: r={correlation:.3f}"
        )
        axes[1, 1].set_xlabel(f"Empirical {groups[1]} - {groups[0]}")
        axes[1, 1].set_ylabel("Posterior-predicted difference")
    else:
        axes[1, 1].axis("off")
    fig.suptitle(
        "M5 spectral posterior validation (group labels used only after fitting)",
        fontsize=14,
    )
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


def _plot_heldout_parameter_effects(
    path: Path,
    table: pd.DataFrame,
    groups: tuple[str, ...],
    recovery_correlations: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Report diagnosis effects only after fitting, with recovery guardrails."""

    selected = table[table["subject_split"] == "holdout"]
    if len(groups) != 2:
        return pd.DataFrame()
    first, second = groups
    rng = np.random.default_rng(seed)

    def standardized_difference(first_values: np.ndarray, second_values: np.ndarray) -> float:
        denominator_dof = len(first_values) + len(second_values) - 2
        if denominator_dof <= 0:
            return np.nan
        pooled_variance = (
            (len(first_values) - 1) * np.var(first_values, ddof=1)
            + (len(second_values) - 1) * np.var(second_values, ddof=1)
        ) / denominator_dof
        if pooled_variance <= np.finfo(float).tiny:
            return np.nan
        return float(
            (np.mean(second_values) - np.mean(first_values))
            / np.sqrt(pooled_variance)
        )

    rows: list[dict[str, Any]] = []
    structural = {
        "default_dorsattn_weight_contrast",
        "default_salventattn_weight_contrast",
    }
    for name in PARAMETER_NAMES:
        column = f"{name}_posterior_mean"
        first_values = selected.loc[selected["group"] == first, column].to_numpy()
        second_values = selected.loc[selected["group"] == second, column].to_numpy()
        bootstrapped = np.asarray(
            [
                standardized_difference(
                    rng.choice(first_values, size=len(first_values), replace=True),
                    rng.choice(second_values, size=len(second_values), replace=True),
                )
                for _ in range(1000)
            ]
        )
        recovery = float(recovery_correlations.get(name, np.nan))
        threshold = 0.30 if name in structural else 0.50
        rows.append(
            {
                "parameter": name,
                "contrast": f"{second} - {first}",
                "n_first": int(len(first_values)),
                "n_second": int(len(second_values)),
                "raw_posterior_mean_difference": float(
                    np.mean(second_values) - np.mean(first_values)
                ),
                "standardized_posterior_mean_difference": standardized_difference(
                    first_values, second_values
                ),
                "bootstrap_ci_low": float(np.nanquantile(bootstrapped, 0.025)),
                "bootstrap_ci_high": float(np.nanquantile(bootstrapped, 0.975)),
                "synthetic_recovery_correlation": recovery,
                "recovery_threshold": threshold,
                "passes_recovery_gate": bool(recovery >= threshold),
            }
        )
    result = pd.DataFrame(rows)
    y = np.arange(len(result))
    estimate = result["standardized_posterior_mean_difference"].to_numpy()
    lower = result["bootstrap_ci_low"].to_numpy()
    upper = result["bootstrap_ci_high"].to_numpy()
    passed = result["passes_recovery_gate"].to_numpy(dtype=bool)
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for mask, color, alpha, label in (
        (passed, "C0", 0.95, "passes synthetic-recovery gate"),
        (~passed, "0.55", 0.45, "fails synthetic-recovery gate"),
    ):
        axis.errorbar(
            estimate[mask],
            y[mask],
            xerr=np.vstack((estimate[mask] - lower[mask], upper[mask] - estimate[mask])),
            fmt="o",
            color=color,
            alpha=alpha,
            capsize=2,
            label=label,
        )
    axis.axvline(0.0, color="black", linestyle=":")
    axis.set_yticks(y)
    axis.set_yticklabels(result["parameter"])
    axis.invert_yaxis()
    axis.set_xlabel(f"Standardized posterior-mean difference ({second} - {first})")
    axis.set_title(
        "Subject-holdout parameter effects—descriptive, not diagnosis fitting"
    )
    axis.legend(fontsize=8)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=170)
    plt.close(fig)
    temporary.replace(path)
    return result


def fit_spectral_subjects(
    config: SpectralM5Config,
    fitting: CrossSpectralCollection | None = None,
    validation: CrossSpectralCollection | None = None,
    bank: SpectralSimulationBank | None = None,
    reliability_first: CrossSpectralCollection | None = None,
    reliability_second: CrossSpectralCollection | None = None,
) -> pd.DataFrame:
    empirical_dir = config.paths.output_dir / "empirical"
    bank_dir = config.paths.output_dir / "bank"
    fitting = fitting or load_cross_spectral_collection(empirical_dir / "cross_spectra_fit.npz")
    validation = validation or load_cross_spectral_collection(empirical_dir / "cross_spectra_validation.npz")
    bank = bank or load_spectral_simulation_bank(bank_dir / "spectral_simulation_bank.npz")
    reliability_first = reliability_first or load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_a.npz"
    )
    reliability_second = reliability_second or load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_b.npz"
    )
    if not np.array_equal(fitting.subject_ids, validation.subject_ids):
        raise ValueError("Fitting and validation subject order differs")
    if not np.array_equal(fitting.frequency_hz, bank.frequency_hz):
        raise ValueError("Empirical and simulated frequency grids differ")
    if not np.array_equal(fitting.channel_names, bank.channel_names):
        raise ValueError("Empirical and simulated channel orders differ")
    if tuple(bank.parameter_names.astype(str)) != PARAMETER_NAMES:
        raise ValueError(
            "Simulation-bank parameter schema differs from the current model"
        )
    train_indices, holdout_indices = _stratified_split(
        fitting.groups, config.spectral.holdout_fraction, config.spectral.split_seed
    )
    source_covariance = _source_background_covariance(config)
    sensor_basis = None
    if config.spectral.sensor_basis_method == "leadfield":
        eigenvalues, eigenvectors = np.linalg.eigh(source_covariance)
        order = np.argsort(eigenvalues)[::-1]
        sensor_basis = eigenvectors[:, order[: config.spectral.sensor_modes]]
    transformer = fit_spectral_transformer(
        fitting,
        train_indices,
        config.spectral,
        reliability_first,
        reliability_second,
        sensor_basis=sensor_basis,
    )
    fit_dir = config.paths.output_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    save_spectral_transformer(fit_dir / "spectral_transformer.npz", transformer)
    fit_features = transformer.transform(fitting.csd)
    validation_features = transformer.transform(validation.csd)
    expanded = _expand_bank(config, transformer, bank, source_covariance)
    state_features = expanded["features"]
    state_mean = state_features.mean(axis=1)
    empirical_repeat_cost = np.sum(
        (fit_features[train_indices] - validation_features[train_indices]) ** 2,
        axis=1,
    )
    base_temperature = float(np.median(empirical_repeat_cost))
    pooled_feature = fit_features[train_indices].mean(axis=0)
    normalized = normalized_spectral_parameters(bank.parameters, config.design)
    neural_state_parameters = normalized[expanded["candidate_index"]]
    prior_cost = config.posterior.prior_strength * np.mean(neural_state_parameters**2, axis=1)
    spatial_columns = [
        PARAMETER_NAMES.index(name)
        for name in (
            "dorsattn_time_contrast",
            "visual_time_contrast",
            "default_noise_contrast",
            "visual_noise_contrast",
            "network_noise_mode_1",
            "network_noise_mode_2",
        )
    ]
    structural_columns = [
        PARAMETER_NAMES.index(name)
        for name in (
            "default_dorsattn_weight_contrast",
            "default_salventattn_weight_contrast",
        )
    ]
    prior_cost += config.posterior.spatial_prior_strength * np.mean(
        neural_state_parameters[:, spatial_columns] ** 2, axis=1
    )
    prior_cost += config.posterior.structural_prior_strength * np.mean(
        neural_state_parameters[:, structural_columns] ** 2, axis=1
    )

    temperature, temperature_table = _select_temperature(
        state_features,
        state_mean,
        fit_features,
        validation_features,
        train_indices,
        pooled_feature,
        prior_cost,
        base_temperature,
        config.posterior.temperature_multipliers,
        config.posterior.minimum_temperature,
    )
    temperature_table.to_csv(
        fit_dir / "training_only_temperature_selection.csv", index=False
    )

    recovery_summary = _synthetic_recovery(
        config,
        bank,
        expanded,
        prior_cost,
        temperature,
        fit_dir / "synthetic_recovery.csv",
    )

    population_cost = np.sum((state_mean - pooled_feature) ** 2, axis=1)
    population_penalized_cost = population_cost + prior_cost
    population_order = np.argsort(population_penalized_cost)
    population_rows: list[dict[str, Any]] = []
    for rank, state in enumerate(population_order, start=1):
        candidate_index = int(expanded["candidate_index"][state])
        row: dict[str, Any] = {
            "rank": rank,
            "candidate_index": candidate_index,
            "pooled_training_feature_cost": float(population_cost[state]),
            "pooled_training_penalized_cost": float(
                population_penalized_cost[state]
            ),
            "gaussian_prior_cost": float(prior_cost[state]),
            "observation_noise_fraction": float(expanded["observation_noise_fraction"][state]),
            "observation_noise_exponent": float(expanded["observation_noise_exponent"][state]),
            "source_background_fraction": float(expanded["source_background_fraction"][state]),
            "source_background_exponent": float(expanded["source_background_exponent"][state]),
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
        weight = _posterior_weight(cost, prior_cost, temperature)
        map_state = int(np.argmax(weight))
        map_candidate = int(expanded["candidate_index"][map_state])
        prediction_feature = weight @ state_mean
        prediction_csd = _posterior_csd(weight, expanded)
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
            "source_background_fraction_posterior_mean": float(weight @ expanded["source_background_fraction"]),
            "source_background_exponent_posterior_mean": float(weight @ expanded["source_background_exponent"]),
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
                    "source_background_fraction": float(expanded["source_background_fraction"][state]),
                    "source_background_exponent": float(expanded["source_background_exponent"][state]),
                    "posterior_weight": float(weight[state]),
                }
            )
    prediction_array = np.stack(predictions)
    table = pd.DataFrame(records)
    fit_auto, fit_cross, fit_topography = transformer.transform_blocks(fitting.csd)
    validation_auto, validation_cross, validation_topography = (
        transformer.transform_blocks(validation.csd)
    )
    prediction_auto, prediction_cross, prediction_topography = (
        transformer.transform_blocks(prediction_array)
    )
    for name, fit_block, validation_block, prediction_block in (
        ("auto_spectrum", fit_auto, validation_auto, prediction_auto),
        ("complex_coherency", fit_cross, validation_cross, prediction_cross),
        (
            "alpha_topography",
            fit_topography,
            validation_topography,
            prediction_topography,
        ),
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
        holdout_indices,
    )
    group_effect_metrics = _plot_group_effects(
        fit_dir / "m5_spectral_group_effects.png",
        validation,
        prediction_array,
        holdout_indices,
    )
    recovery_correlations = recovery_summary.get(
        "parameter_recovery_correlations", {}
    )
    parameter_effects = _plot_heldout_parameter_effects(
        fit_dir / "m5_heldout_parameter_effects.png",
        table,
        config.empirical.groups,
        recovery_correlations,
        config.spectral.split_seed + 1,
    )
    parameter_effects.to_csv(
        fit_dir / "heldout_group_parameter_effects.csv", index=False
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
            "median_alpha_topography_validation_cost_ratio_to_pooled_null": float(selected["validation_alpha_topography_cost_ratio_to_pooled_null"].median()),
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

    observation_boundary = bool(
        population_rows[0]["observation_noise_fraction"]
        == max(config.posterior.observation_noise_fractions)
    )
    source_boundary = bool(
        len(config.posterior.source_background_fractions) > 1
        and population_rows[0]["source_background_fraction"]
        == max(config.posterior.source_background_fractions)
    )
    structural_recovery = [
        float(recovery_correlations.get(name, 0.0))
        for name in (
            "default_dorsattn_weight_contrast",
            "default_salventattn_weight_contrast",
        )
    ]
    holdout_summary = split_summary["holdout"]
    acceptance_gates = {
        "holdout_unseen_total_cost_below_pooled_null": bool(
            holdout_summary["median_validation_cost_ratio_to_pooled_null"] < 1.0
        ),
        "holdout_majority_of_subjects_beat_pooled_null": bool(
            holdout_summary["fraction_beating_pooled_null_on_validation"] > 0.5
        ),
        "holdout_auto_spectrum_cost_below_pooled_null": bool(
            holdout_summary[
                "median_auto_spectrum_validation_cost_ratio_to_pooled_null"
            ]
            < 1.0
        ),
        "holdout_complex_coherency_cost_below_pooled_null": bool(
            holdout_summary[
                "median_complex_coherency_validation_cost_ratio_to_pooled_null"
            ]
            < 1.0
        ),
        "holdout_alpha_topography_cost_below_pooled_null": bool(
            holdout_summary[
                "median_alpha_topography_validation_cost_ratio_to_pooled_null"
            ]
            < 1.0
        ),
        "holdout_power_group_effect_preserved": bool(
            group_effect_metrics.get(
                "channel_frequency_log_power_effect_correlation", 0.0
            )
            >= 0.30
        ),
        "holdout_alpha_topography_group_effect_preserved": bool(
            group_effect_metrics.get("alpha_topography_effect_correlation", 0.0)
            >= 0.30
        ),
        "holdout_coherency_group_effect_preserved": bool(
            group_effect_metrics.get(
                "complex_coherency_effect_correlation", 0.0
            )
            >= 0.10
        ),
        "population_observation_nuisance_not_at_boundary": bool(
            not observation_boundary and not source_boundary
        ),
        "at_least_three_parameters_recoverable_in_synthetic_data": bool(
            sum(float(value) >= 0.50 for value in recovery_correlations.values())
            >= 3
        ),
        "both_structural_modes_recoverable_in_synthetic_data": bool(
            len(structural_recovery) == 2
            and min(structural_recovery) >= 0.30
        ),
    }
    accepted = all(acceptance_gates.values())
    summary = {
        "status": "accepted" if accepted else "not_accepted",
        "run_scale": "pilot" if len(bank.parameters) < 64 or config.design.replicates < 3 else "production",
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
                "direct_alpha_topography_coordinates": int(
                    transformer.topography_components.shape[0]
                    if transformer.topography_components.shape[0]
                    else transformer.topography_mean.size
                ),
                "reliability_calibration": (
                    "two non-overlapping quarters inside the fitting half; "
                    "training-subject final halves select one global posterior "
                    "temperature, while subject-holdout final halves remain unseen"
                ),
            },
            "periodic_aperiodic": "per-sensor-mode log-linear background from 2-7 and 30-40 Hz; fit retains residual spectra and explicit exponents",
            "absolute_amplitude": "per-sensor-mode intercepts removed as observation-gain nuisances",
            "dc": "excluded by demeaning and the 2-Hz lower fit bound",
            "structural_weights": (
                "two symmetric mean-strength-preserving network-pair modes, bounded "
                "to +/-10% and given a stronger Gaussian shrinkage prior"
            ),
            "spatial_physiology": (
                "visual/default noise and visual/dorsal-attention time contrasts, "
                "plus two fixed diagnosis-blind network-noise modes learned from "
                "training-subject alpha-topography variation"
            ),
        },
        "posterior": {
            "kind": "finite-bank kernel posterior, not an exact posterior and not yet amortized SBI",
            "base_temperature_from_median_train_subject_fit_vs_unseen_discrepancy": base_temperature,
            "selected_temperature_training_subjects_only": temperature,
            "temperature_selection_file": "training_only_temperature_selection.csv",
            "observation_noise_grid_size": int(
                len(config.posterior.observation_noise_fractions)
                * len(config.posterior.observation_noise_exponents)
                * len(config.posterior.source_background_fractions)
                * len(config.posterior.source_background_exponents)
            ),
            "gaussian_regularization": {
                "global_parameter_strength": config.posterior.prior_strength,
                "spatial_physiology_strength": config.posterior.spatial_prior_strength,
                "structural_weight_strength": config.posterior.structural_prior_strength,
                "definition": (
                    "quadratic Gaussian penalty on range-normalized deviations; "
                    "added to the data objective before kernel tempering"
                ),
            },
        },
        "synthetic_recovery": recovery_summary,
        "population_calibration": {
            "target": "label-blind pooled fitting-half spectrum from training subjects",
            "best_state": population_rows[0],
            "best_observation_noise_fraction_at_grid_maximum": observation_boundary,
            "best_source_background_fraction_at_grid_maximum": source_boundary,
            "interpretation": (
                "An observation nuisance reaches the tested boundary, indicating "
                "unresolved neural/forward-model mismatch."
                if observation_boundary or source_boundary
                else "Observation nuisances are interior to the tested grid."
            ),
        },
        "simulation": {
            "candidates": int(len(bank.parameters)),
            "replicates": int(bank.csd_replicates.shape[1]),
            "analyzed_seconds_per_replicate": float((config.design.duration_ms - config.design.transient_ms) / 1000.0),
            "common_random_numbers_across_candidates": True,
        },
        "validation": split_summary,
        "validation_roles": {
            "train_subject_second_half": (
                "used only to select one global posterior temperature and therefore "
                "reported as internal validation, not an unbiased test"
            ),
            "holdout_subject_first_half": (
                "used for that subject's diagnosis-blind parameter fit"
            ),
            "holdout_subject_second_half": (
                "untouched by feature, objective-resolution, nuisance-grid, and "
                "temperature selection; primary subject-level test"
            ),
        },
        "subject_holdout_unseen_group_effect_preservation": group_effect_metrics,
        "subject_holdout_parameter_effects": {
            "table": "heldout_group_parameter_effects.csv",
            "figure": "m5_heldout_parameter_effects.png",
            "interpretation": (
                "Descriptive effects of independently fitted posterior means; "
                "parameters failing synthetic recovery are not biological findings."
            ),
        },
        "acceptance_gates": acceptance_gates,
        "guardrails": [
            "Diagnosis labels never enter subject fitting or feature reduction.",
            "The two empirical spatial modes use training subjects only and no diagnosis labels.",
            "Group summaries are descriptive second-level outputs, not group-level fits.",
            "The pooled empirical mean is the explicit null model.",
            "Only the subject-holdout second halves are an unbiased final test after temperature selection.",
            "Stimulation optimization remains blocked until unseen spatial group effects pass.",
        ],
        "config": asdict(config),
    }
    (fit_dir / "fit_summary.json").write_text(
        json.dumps(json_value(summary), indent=2), encoding="utf-8"
    )
    (fit_dir / "acceptance_report.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "gates": acceptance_gates,
                "note": (
                    "No stimulation optimization should use this fit until all "
                    "gates required by the scientific question pass."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_lines = [
        f"# M5 acceptance: {'PASS' if accepted else 'FAIL'}",
        "",
        "## Key held-out metrics",
        "",
        (
            "- median unseen total cost / pooled null: "
            f"{holdout_summary['median_validation_cost_ratio_to_pooled_null']:.3f}"
        ),
        (
            "- held-out subjects beating pooled null: "
            f"{holdout_summary['fraction_beating_pooled_null_on_validation']:.1%}"
        ),
        (
            "- median auto / coherency / alpha-topography ratios: "
            f"{holdout_summary['median_auto_spectrum_validation_cost_ratio_to_pooled_null']:.3f} / "
            f"{holdout_summary['median_complex_coherency_validation_cost_ratio_to_pooled_null']:.3f} / "
            f"{holdout_summary['median_alpha_topography_validation_cost_ratio_to_pooled_null']:.3f}"
        ),
        (
            "- power / alpha-topography / coherency group-effect correlations: "
            f"{group_effect_metrics.get('channel_frequency_log_power_effect_correlation', np.nan):.3f} / "
            f"{group_effect_metrics.get('alpha_topography_effect_correlation', np.nan):.3f} / "
            f"{group_effect_metrics.get('complex_coherency_effect_correlation', np.nan):.3f}"
        ),
        (
            "- structural-mode synthetic-recovery correlations: "
            + " / ".join(f"{value:.3f}" for value in structural_recovery)
        ),
        "",
        "## Gates",
        "",
        *[
            f"- {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}"
            for name, passed in acceptance_gates.items()
        ],
        "",
        "Stimulation optimization remains blocked while required gates fail.",
    ]
    (fit_dir / "ACCEPTANCE.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return table
