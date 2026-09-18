"""Audit whether the production M5 bank can identify each fitted parameter.

This is a diagnosis-label-blind development diagnostic.  It uses only the
declared training subjects when fitting the feature transformer and never
uses Healthy/MDD labels.  The report separates model sensitivity from Monte
Carlo noise, empirical feature coverage, bank resolution, and prior shrinkage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import (
    fit_spectral_transformer,
    load_cross_spectral_collection,
)
from mdd_tvb.spectral_fit import (
    _expand_bank,
    _source_background_covariance,
    _stratified_split,
)
from mdd_tvb.spectral_parameterization import (
    PARAMETER_NAMES,
    normalized_spectral_parameters,
)


SPATIAL_NAMES = PARAMETER_NAMES[9:]
STRUCTURAL_NAMES = PARAMETER_NAMES[-2:]


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _softmax_rows(log_weight: np.ndarray) -> np.ndarray:
    shifted = log_weight - np.max(log_weight, axis=1, keepdims=True)
    weight = np.exp(shifted)
    return weight / weight.sum(axis=1, keepdims=True)


def _block_slices(transformer: object) -> dict[str, slice]:
    auto_count = (
        transformer.auto_components.shape[0]
        if transformer.auto_components.shape[0]
        else transformer.auto_mean.size
    )
    cross_count = (
        transformer.cross_components.shape[0]
        if transformer.cross_components.shape[0]
        else transformer.cross_mean.size
    )
    topography_count = (
        transformer.topography_components.shape[0]
        if transformer.topography_components.shape[0]
        else transformer.topography_mean.size
    )
    return {
        "auto_spectrum": slice(0, auto_count),
        "complex_coherency": slice(auto_count, auto_count + cross_count),
        "alpha_topography": slice(
            auto_count + cross_count,
            auto_count + cross_count + topography_count,
        ),
    }


def _prior_cost(
    state_parameters: np.ndarray,
    global_strength: float,
    spatial_strength: float,
    structural_strength: float,
) -> np.ndarray:
    global_columns = np.arange(0, 9)
    spatial_columns = np.arange(9, 15)
    structural_columns = np.arange(15, 17)
    result = global_strength * np.mean(
        state_parameters[:, global_columns] ** 2, axis=1
    )
    result += spatial_strength * np.mean(
        state_parameters[:, spatial_columns] ** 2, axis=1
    )
    result += structural_strength * np.mean(
        state_parameters[:, structural_columns] ** 2, axis=1
    )
    return result


def run_audit(config_path: Path) -> dict[str, object]:
    config = load_spectral_m5_config(config_path)
    empirical_dir = config.paths.output_dir / "empirical"
    bank_dir = config.paths.output_dir / "bank"
    output_dir = config.paths.output_dir / "observability"
    output_dir.mkdir(parents=True, exist_ok=True)

    fitting = load_cross_spectral_collection(empirical_dir / "cross_spectra_fit.npz")
    reliability_a = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_a.npz"
    )
    reliability_b = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_b.npz"
    )
    bank = load_spectral_simulation_bank(bank_dir / "spectral_simulation_bank.npz")
    train_indices, _ = _stratified_split(
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
        reliability_a,
        reliability_b,
        sensor_basis=sensor_basis,
    )
    expanded = _expand_bank(config, transformer, bank, source_covariance)
    block_slices = _block_slices(transformer)

    zero_state = np.flatnonzero(
        np.isclose(expanded["observation_noise_fraction"], 0.0)
        & np.isclose(expanded["source_background_fraction"], 0.0)
    )
    # Exponents are irrelevant at a zero fraction and therefore create exact
    # duplicates. Retain one state per neural candidate.
    retained: list[int] = []
    for candidate_index in range(len(bank.parameters)):
        matches = zero_state[
            expanded["candidate_index"][zero_state] == candidate_index
        ]
        retained.append(int(matches[0]))
    candidate_seed_features = expanded["features"][np.asarray(retained)]
    candidate_mean = candidate_seed_features.mean(axis=1)
    empirical_features = transformer.transform(fitting.csd[train_indices])
    normalized = normalized_spectral_parameters(bank.parameters, config.design)

    x = normalized - normalized.mean(axis=0, keepdims=True)
    y = candidate_mean - candidate_mean.mean(axis=0, keepdims=True)
    ridge = 1e-6 * np.eye(x.shape[1])
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    seed_residual = candidate_seed_features - candidate_mean[:, np.newaxis, :]

    sensitivity_rows: list[dict[str, object]] = []
    for parameter_index, name in enumerate(PARAMETER_NAMES):
        for block, selected in block_slices.items():
            effect = beta[parameter_index, selected]
            noise = seed_residual[:, :, selected]
            sensitivity_rows.append(
                {
                    "parameter": name,
                    "block": block,
                    "linear_effect_rms": float(np.sqrt(np.mean(effect**2))),
                    "replicate_noise_rms": float(np.sqrt(np.mean(noise**2))),
                    "effect_to_replicate_noise": float(
                        np.sqrt(np.mean(effect**2))
                        / max(np.sqrt(np.mean(noise**2)), np.finfo(float).tiny)
                    ),
                    "maximum_absolute_marginal_correlation": float(
                        max(
                            abs(
                                _safe_correlation(
                                    normalized[:, parameter_index],
                                    candidate_mean[:, coordinate],
                                )
                            )
                            for coordinate in range(selected.start, selected.stop)
                        )
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "parameter_feature_sensitivity.csv", index=False)

    coverage_rows: list[dict[str, object]] = []
    for block, selected in block_slices.items():
        model_scale = np.std(candidate_mean[:, selected], axis=0, ddof=1)
        empirical_scale = np.std(empirical_features[:, selected], axis=0, ddof=1)
        replicate_scale = np.sqrt(
            np.mean(seed_residual[:, :, selected] ** 2, axis=(0, 1))
        )
        ratio = model_scale / np.maximum(empirical_scale, np.finfo(float).tiny)
        reliability = model_scale**2 / np.maximum(
            model_scale**2 + replicate_scale**2, np.finfo(float).tiny
        )
        coverage_rows.append(
            {
                "block": block,
                "coordinates": selected.stop - selected.start,
                "median_model_to_empirical_sd": float(np.median(ratio)),
                "minimum_model_to_empirical_sd": float(np.min(ratio)),
                "median_candidate_replicate_reliability": float(
                    np.median(reliability)
                ),
                "minimum_candidate_replicate_reliability": float(
                    np.min(reliability)
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "feature_block_coverage.csv", index=False)

    fit_summary_path = config.paths.output_dir / "fit" / "fit_summary.json"
    with fit_summary_path.open("r", encoding="utf-8") as handle:
        fit_summary = json.load(handle)
    temperature = float(
        fit_summary["posterior"]["selected_temperature_training_subjects_only"]
    )
    feature_states = expanded["features"]
    prediction = feature_states[:, 1:].mean(axis=1)
    target_fraction = min(
        config.posterior.observation_noise_fractions,
        key=lambda value: abs(value - 0.5),
    )
    target_exponent = min(
        config.posterior.observation_noise_exponents,
        key=lambda value: abs(value - 1.0),
    )
    target_source_fraction = min(
        config.posterior.source_background_fractions,
        key=lambda value: abs(value - 0.5),
    )
    target_source_exponent = min(
        config.posterior.source_background_exponents,
        key=lambda value: abs(value - 1.0),
    )
    target_state = np.empty(len(bank.parameters), dtype=int)
    for candidate_index in range(len(bank.parameters)):
        matches = np.flatnonzero(
            (expanded["candidate_index"] == candidate_index)
            & np.isclose(
                expanded["observation_noise_fraction"], target_fraction
            )
            & np.isclose(
                expanded["observation_noise_exponent"], target_exponent
            )
            & np.isclose(
                expanded["source_background_fraction"], target_source_fraction
            )
            & np.isclose(
                expanded["source_background_exponent"], target_source_exponent
            )
        )
        target_state[candidate_index] = int(matches[0])
    target = feature_states[target_state, 0]
    target_norm = np.sum(target**2, axis=1, keepdims=True)
    prediction_norm = np.sum(prediction**2, axis=1)[np.newaxis, :]
    cost = np.maximum(
        target_norm + prediction_norm - 2.0 * target @ prediction.T,
        0.0,
    )
    state_parameters = normalized[expanded["candidate_index"]]
    recovery_rows: list[dict[str, object]] = []
    for structural_strength in (0.0, 0.02, 0.05, 0.10, 0.20):
        prior = _prior_cost(
            state_parameters,
            config.posterior.prior_strength,
            config.posterior.spatial_prior_strength,
            structural_strength,
        )
        weight = _softmax_rows(-0.5 * (cost + prior[np.newaxis, :]) / temperature)
        estimate = weight @ state_parameters
        for parameter_index, name in enumerate(PARAMETER_NAMES):
            recovery_rows.append(
                {
                    "structural_prior_strength": structural_strength,
                    "parameter": name,
                    "recovery_correlation": _safe_correlation(
                        normalized[:, parameter_index], estimate[:, parameter_index]
                    ),
                    "normalized_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    normalized[:, parameter_index]
                                    - estimate[:, parameter_index]
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
    recovery = pd.DataFrame(recovery_rows)
    recovery.to_csv(output_dir / "prior_recovery_sweep.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    total_sensitivity = (
        sensitivity.groupby("parameter", sort=False)["effect_to_replicate_noise"]
        .mean()
        .reindex(PARAMETER_NAMES)
    )
    colors = ["C3" if name in STRUCTURAL_NAMES else "C0" for name in PARAMETER_NAMES]
    axes[0].barh(PARAMETER_NAMES, total_sensitivity, color=colors)
    axes[0].axvline(1.0, color="black", linestyle=":")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean linear effect / stochastic replicate noise")
    axes[0].set_title("Bank observability (diagnosis-label blind)")

    structural = recovery[recovery["parameter"].isin(STRUCTURAL_NAMES)]
    for name in STRUCTURAL_NAMES:
        selected = structural[structural["parameter"] == name]
        axes[1].plot(
            selected["structural_prior_strength"],
            selected["recovery_correlation"],
            marker="o",
            label=name,
        )
    axes[1].axhline(0.30, color="black", linestyle=":", label="acceptance threshold")
    axes[1].set_xlabel("structural prior strength")
    axes[1].set_ylabel("synthetic recovery correlation")
    axes[1].set_title("Prior versus structural recovery")
    axes[1].legend(fontsize=8)
    figure.savefig(output_dir / "m5_observability_audit.png", dpi=180)
    plt.close(figure)

    selected_recovery = recovery[
        np.isclose(
            recovery["structural_prior_strength"],
            config.posterior.structural_prior_strength,
        )
    ]
    result: dict[str, object] = {
        "status": "completed",
        "training_subjects_only": True,
        "candidates": int(len(bank.parameters)),
        "spatial_samples": int(config.design.spatial_samples),
        "spatial_dimensions": len(SPATIAL_NAMES),
        "selected_temperature": temperature,
        "feature_block_coverage": coverage.to_dict(orient="records"),
        "selected_prior_recovery": {
            str(row["parameter"]): float(row["recovery_correlation"])
            for _, row in selected_recovery.iterrows()
        },
        "structural_recovery_without_structural_prior": {
            str(row["parameter"]): float(row["recovery_correlation"])
            for _, row in recovery[
                np.isclose(recovery["structural_prior_strength"], 0.0)
                & recovery["parameter"].isin(STRUCTURAL_NAMES)
            ].iterrows()
        },
        "outputs": {
            "sensitivity": "parameter_feature_sensitivity.csv",
            "coverage": "feature_block_coverage.csv",
            "prior_sweep": "prior_recovery_sweep.csv",
            "figure": "m5_observability_audit.png",
        },
    }
    with (output_dir / "observability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m5_spectral.toml"))
    args = parser.parse_args()
    result = run_audit(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
