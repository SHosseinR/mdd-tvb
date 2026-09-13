from dataclasses import replace
from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.heterogeneity import build_regional_parameters


def test_regional_parameters_are_reproducible_and_bounded() -> None:
    config = load_config(Path("configs/baseline.toml"))
    connectome = load_connectome(config.paths, config.connectivity)
    first = build_regional_parameters(
        config.model, config.simulation.noise_nsig,
        connectome.region_labels, config.heterogeneity,
    )
    second = build_regional_parameters(
        config.model, config.simulation.noise_nsig,
        connectome.region_labels, config.heterogeneity,
    )
    assert np.array_equal(first.mu, second.mu)
    assert np.array_equal(first.a, second.a)
    assert np.unique(first.network_labels).size == 7
    assert first.time_scale_multiplier.min() >= 0.85
    assert first.time_scale_multiplier.max() <= 1.15
    visual = first.network_labels == "Vis"
    assert first.noise_multiplier[visual].mean() > first.noise_multiplier[~visual].mean()


def test_declared_network_multipliers_affect_only_the_requested_network() -> None:
    config = load_config(Path("configs/baseline.toml"))
    connectome = load_connectome(config.paths, config.connectivity)
    baseline = build_regional_parameters(
        config.model, config.simulation.noise_nsig,
        connectome.region_labels, config.heterogeneity,
    )
    modified = build_regional_parameters(
        config.model,
        config.simulation.noise_nsig,
        connectome.region_labels,
        replace(
            config.heterogeneity,
            network_drive_multipliers={"Default": 1.10},
            network_time_scale_multipliers={"SomMot": 0.90},
            network_noise_multipliers={"Cont": 1.05},
        ),
    )
    default = baseline.network_labels == "Default"
    somatomotor = baseline.network_labels == "SomMot"
    control = baseline.network_labels == "Cont"
    assert np.allclose(modified.mu[default], baseline.mu[default] * 1.10)
    assert np.allclose(modified.mu[~default], baseline.mu[~default])
    assert np.allclose(modified.a[somatomotor], baseline.a[somatomotor] * 0.90)
    assert np.allclose(modified.b[somatomotor], baseline.b[somatomotor] * 0.90)
    assert np.allclose(modified.noise_nsig[control], baseline.noise_nsig[control] * 1.05)
