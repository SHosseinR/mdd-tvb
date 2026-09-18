"""Relate empirical alpha-topography effects to Schaefer-network lead fields."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.eeg import build_eeg_monitor, regularize_analytic_eeg_gain
from mdd_tvb.heterogeneity import network_labels
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import load_cross_spectral_collection
from mdd_tvb.spectral_fit import _channel_log_power, _group_difference, _safe_correlation


def main() -> None:
    config = load_spectral_m5_config("configs/m5_spectral_calibration.toml")
    empirical = load_cross_spectral_collection(
        config.paths.output_dir / "empirical" / "cross_spectra_fit.npz"
    )
    split = pd.read_csv(config.paths.output_dir / "fit" / "data_split.csv")
    train = np.flatnonzero(split["subject_split"].to_numpy() == "train")
    groups = empirical.groups[train]
    power = _channel_log_power(empirical.csd[train])
    alpha = (empirical.frequency_hz >= 8.0) & (empirical.frequency_hz < 13.0)
    subject_topography = power[:, alpha].mean(axis=1)
    subject_topography -= subject_topography.mean(axis=1, keepdims=True)
    effect = _group_difference(subject_topography, groups, "Healthy", "MDD")

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
    if baseline.monitor.reference.lower() == "average":
        gain -= gain.mean(axis=0, keepdims=True)
    networks = network_labels(connectome.region_labels)
    names = np.unique(networks)
    network_power = np.stack(
        [np.sum(gain[:, networks == name] ** 2, axis=1) for name in names],
        axis=1,
    )
    fractions = network_power / np.maximum(
        network_power.sum(axis=1, keepdims=True), np.finfo(float).tiny
    )
    fractions -= fractions.mean(axis=0, keepdims=True)
    gram = fractions.T @ fractions
    ridge = 0.05 * float(np.trace(gram)) / len(names)
    subject_coefficients = (
        subject_topography
        @ fractions
        @ np.linalg.inv(gram + ridge * np.eye(len(names)))
    )
    subject_coefficients -= subject_coefficients.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(
        subject_coefficients, full_matrices=False
    )
    network_modes: list[dict[str, object]] = []
    explained = singular_values**2 / np.sum(singular_values**2)
    for mode_index in range(2):
        mode = right[mode_index].copy()
        mode -= np.average(
            mode,
            weights=np.asarray([np.sum(networks == name) for name in names]),
        )
        mode /= np.max(np.abs(mode))
        if mode[np.argmax(np.abs(mode))] < 0.0:
            mode *= -1.0
        network_modes.append(
            {
                "mode": mode_index + 1,
                "explained_coefficient_variance": float(explained[mode_index]),
                "coefficients": {
                    str(name): float(mode[column])
                    for column, name in enumerate(names)
                },
            }
        )
    coefficients, *_ = np.linalg.lstsq(fractions, effect, rcond=None)
    fitted = fractions @ coefficients
    rows = [
        {
            "network": str(name),
            "univariate_effect_correlation": _safe_correlation(
                fractions[:, column], effect
            ),
            "multivariate_coefficient": float(coefficients[column]),
        }
        for column, name in enumerate(names)
    ]
    output = {
        "training_subjects_only": True,
        "multivariate_effect_correlation": _safe_correlation(fitted, effect),
        "networks": rows,
        "label_blind_training_alpha_topography_network_modes": network_modes,
    }
    path = config.project_root / "outputs" / "m5_network_topography_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
