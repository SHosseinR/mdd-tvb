from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import (
    load_connectome,
    with_network_endpoint_gains,
    with_network_pair_gains,
)
from mdd_tvb.features import extract_eeg_features
from mdd_tvb.fit_config import load_m5_config
from mdd_tvb.heterogeneity import network_labels
from mdd_tvb.parameterization import NETWORK_ORDER, make_design


def test_m5_config_and_bounded_design() -> None:
    config = load_m5_config(Path("configs/m5_fit.toml"))
    design = make_design(config.design)
    assert len(design) == config.design.samples
    assert np.array_equal(design[0].network_gains, np.ones(7))
    assert design[0].a_scale == config.design.reference_a_scale
    assert design[0].b_scale == config.design.reference_b_scale
    for candidate in design:
        assert np.isclose(candidate.network_gains.mean(), 1.0)
        assert candidate.network_gains.min() >= 0.9 - 1e-12
        assert candidate.network_gains.max() <= 1.1 + 1e-12


def test_network_endpoint_gains_preserve_symmetry_and_support() -> None:
    baseline = load_config(Path("configs/baseline.toml"))
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    gains = {name: 1.0 for name in NETWORK_ORDER}
    gains["Vis"] = 1.1
    gains["Cont"] = 0.9
    modified = with_network_endpoint_gains(connectome, networks, gains)
    assert np.allclose(modified.weights, modified.weights.T)
    assert np.array_equal(modified.weights > 0, connectome.weights > 0)
    assert not np.array_equal(modified.weights, connectome.weights)


def test_network_pair_gains_preserve_support_symmetry_and_total_strength() -> None:
    baseline = load_config(Path("configs/baseline.toml"))
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    modified = with_network_pair_gains(
        connectome,
        networks,
        {("Default", "Cont"): 1.1, ("Limbic", "Limbic"): 0.9},
    )
    assert np.allclose(modified.weights, modified.weights.T)
    assert np.array_equal(modified.weights > 0, connectome.weights > 0)
    assert np.isclose(modified.weights.sum(), connectome.weights.sum())
    assert not np.array_equal(modified.weights, connectome.weights)


def test_m5_features_ignore_dc_and_absolute_amplitude() -> None:
    config = load_m5_config(Path("configs/m5_fit.toml"))
    sfreq = 500.0
    time = np.arange(0.0, 10.0, 1.0 / sfreq)
    channels = np.arange(26)
    rng = np.random.default_rng(3)
    eeg = (
        np.sin(2.0 * np.pi * 10.0 * time[:, None] + channels[None, :] / 7.0)
        + 0.05 * rng.normal(size=(time.size, channels.size))
    )
    first = extract_eeg_features(eeg, sfreq, config.features)
    second = extract_eeg_features(17.0 * eeg + channels[None, :] * 100.0, sfreq, config.features)
    assert np.allclose(first.psd_shape, second.psd_shape, atol=1e-10)
    assert np.allclose(first.alpha_topography, second.alpha_topography, atol=1e-10)
    assert np.allclose(first.coherence, second.coherence, atol=1e-10)
    assert first.alpha_peak_hz == 10.0
    assert np.isclose(first.alpha_power_fraction, second.alpha_power_fraction)
