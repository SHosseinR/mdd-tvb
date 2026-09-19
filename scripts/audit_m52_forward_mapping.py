"""Exploratory, development-only audit of regional EEG forward geometries.

This is not a subject fit or a replacement for the fixed M5.2 eligibility
gates. A seven-network nonnegative alpha-power oracle asks whether each gain
can even express empirical sensor-power profiles using the same seven network
degrees of freedom, fitted on the first temporal half and scored on the next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.eeg import build_eeg_monitor, regularize_analytic_eeg_gain
from mdd_tvb.heterogeneity import network_labels
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import load_cross_spectral_collection
from mdd_tvb.template_bem import load_template_bem_gain


def network_power_basis(gain: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build seven nonnegative sensor-power templates from a regional gain."""

    names = sorted(set(labels.astype(str)))
    matrix = np.stack(
        [np.sum(np.square(gain[:, labels == name]), axis=1) for name in names],
        axis=1,
    )
    if np.any(matrix.mean(axis=0) <= 0.0):
        raise ValueError("At least one network has zero sensor power")
    return matrix / matrix.mean(axis=0, keepdims=True), names


def alpha_power_profile(csd: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    alpha = (frequency_hz >= 8.0) & (frequency_hz < 13.0)
    if not alpha.any():
        raise ValueError("No alpha frequencies in empirical CSD")
    power = np.real(np.diagonal(csd[:, alpha].mean(axis=1), axis1=-2, axis2=-1))
    if np.any(power <= 0.0) or not np.isfinite(power).all():
        raise ValueError("Invalid empirical alpha power")
    return power / power.mean(axis=1, keepdims=True)


def evaluate_network_oracle(
    basis: np.ndarray, fit_profile: np.ndarray, unseen_profile: np.ndarray
) -> np.ndarray:
    if basis.shape[0] != fit_profile.shape[1]:
        raise ValueError("Sensor axes do not match")
    design = np.column_stack([basis, np.ones(len(basis))])
    predictions = []
    for target in fit_profile:
        weights, _ = nnls(design, target)
        prediction = design @ weights
        predictions.append(prediction / prediction.mean())
    predictions = np.stack(predictions)
    return np.sqrt(np.mean(np.square(predictions - unseen_profile), axis=1))


def centered_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    return float(
        np.dot(first, second)
        / max(np.linalg.norm(first) * np.linalg.norm(second), np.finfo(float).tiny)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corrected-gain",
        type=Path,
        default=Path(
            "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/m52_forward_mapping_audit")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_spectral_m5_config(root / "configs" / "m5_spectral.toml")
    baseline = load_config(config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    labels = network_labels(connectome.region_labels)
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
    analytic = np.asarray(monitor.gain, dtype=float)
    analytic -= analytic.mean(axis=0, keepdims=True)
    old, old_channels, old_regions = load_template_bem_gain(
        root / "data/forward/template_bem/template_bem_schaefer200_gain.npz"
    )
    corrected_path = args.corrected_gain
    if not corrected_path.is_absolute():
        corrected_path = root / corrected_path
    corrected, corrected_channels, corrected_regions = load_template_bem_gain(
        corrected_path
    )
    expected_channels = np.asarray(baseline.monitor.channels).astype(str)
    expected_regions = np.asarray(connectome.region_labels).astype(str)
    for channels, regions in (
        (old_channels, old_regions),
        (corrected_channels, corrected_regions),
    ):
        if not np.array_equal(channels.astype(str), expected_channels):
            raise ValueError("BEM sensor order mismatch")
        if not np.array_equal(regions.astype(str), expected_regions):
            raise ValueError("BEM parcel order mismatch")

    empirical_dir = root / "outputs/m5_spectral_m51_production/empirical"
    fitting = load_cross_spectral_collection(empirical_dir / "cross_spectra_fit.npz")
    unseen = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_validation.npz"
    )
    if not np.array_equal(fitting.subject_ids, unseen.subject_ids):
        raise ValueError("Empirical temporal halves are not subject-aligned")
    if not np.array_equal(fitting.channel_names.astype(str), expected_channels):
        raise ValueError("Empirical channel order mismatch")
    split = pd.read_csv(root / "configs/m52_nested_splits.csv")
    development_ids = set(split.subject_id.astype(str))
    selected = np.asarray(
        [
            index
            for index, subject_id in enumerate(fitting.subject_ids.astype(str))
            if subject_id in development_ids
        ],
        dtype=int,
    )
    if len(selected) != 262:
        raise ValueError("Audit must use exactly 262 development subjects")
    fit_profile = alpha_power_profile(fitting.csd[selected], fitting.frequency_hz)
    unseen_profile = alpha_power_profile(unseen.csd[selected], unseen.frequency_hz)
    pooled_profile = fit_profile.mean(axis=0)
    pooled_rmse = np.sqrt(
        np.mean(np.square(unseen_profile - pooled_profile), axis=1)
    )

    gains = {
        "analytic": analytic,
        "old_misregistered_bem": old,
        "corrected_bem": corrected,
    }
    models = {}
    bases = {}
    for name, gain in gains.items():
        basis, networks = network_power_basis(gain, labels)
        bases[name] = basis
        errors = evaluate_network_oracle(basis, fit_profile, unseen_profile)
        models[name] = {
            "median_unseen_alpha_profile_rmse": float(np.median(errors)),
            "median_ratio_to_pooled_null": float(np.median(errors / pooled_rmse)),
            "fraction_below_pooled_null": float(np.mean(errors < pooled_rmse)),
            "median_fit_to_unseen_profile_difference": float(
                np.median(
                    np.sqrt(
                        np.mean(np.square(fit_profile - unseen_profile), axis=1)
                    )
                )
            ),
        }
        models[name]["per_subject_unseen_alpha_profile_rmse"] = errors.tolist()

    mode_cosines = {}
    for index, mode in enumerate(
        (
            config.design.network_noise_mode_1,
            config.design.network_noise_mode_2,
        ),
        start=1,
    ):
        coefficients_by_network = dict(mode)
        coefficients = np.asarray(
            [coefficients_by_network[name] for name in networks]
        )
        analytic_map = bases["analytic"] @ coefficients
        mode_cosines[f"mode_{index}"] = {
            name: centered_cosine(analytic_map, basis @ coefficients)
            for name, basis in bases.items()
        }

    output = {
        "interpretation": (
            "Exploratory seven-network alpha-power expressivity diagnostic only; "
            "not a neural-model fit or M5.2 acceptance result."
        ),
        "development_subjects": len(selected),
        "consumed_m51_holdout_read": False,
        "networks": networks,
        "pooled_null_median_rmse": float(np.median(pooled_rmse)),
        "model": models,
        "inherited_spatial_mode_map_cosines_to_analytic": mode_cosines,
    }
    (output_dir / "forward_mapping_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].boxplot(
        [np.asarray(models[name]["per_subject_unseen_alpha_profile_rmse"]) for name in gains],
        tick_labels=list(gains),
        showfliers=False,
    )
    axes[0].axhline(np.median(pooled_rmse), color="black", linestyle="--")
    axes[0].set_ylabel("Unseen normalized alpha-profile RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].plot(expected_channels, pooled_profile, label="empirical development mean")
    for name, basis in bases.items():
        weights, _ = nnls(np.column_stack([basis, np.ones(len(basis))]), pooled_profile)
        prediction = np.column_stack([basis, np.ones(len(basis))]) @ weights
        axes[1].plot(expected_channels, prediction / prediction.mean(), label=name)
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_ylabel("Mean normalized alpha power")
    axes[1].legend(fontsize=8)
    figure.savefig(output_dir / "forward_mapping_audit.png", dpi=170)
    plt.close(figure)
    concise = {key: value for key, value in output.items() if key != "model"}
    concise["model"] = {
        name: {
            key: value
            for key, value in result.items()
            if key != "per_subject_unseen_alpha_profile_rmse"
        }
        for name, result in models.items()
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
