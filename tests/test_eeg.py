from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.eeg import (
    add_colored_observation_noise,
    build_eeg_monitor,
    mne_sensor_montage,
    project_regional_psp_to_eeg,
    regularize_analytic_eeg_gain,
    sensor_coordinate_audit,
    sensor_locations,
)
from mdd_tvb.reporting import _highpass_for_visualization


def test_exact_tdbrain_montage() -> None:
    config = load_config(Path("configs/baseline.toml"))
    locations = sensor_locations(config.monitor)
    assert locations.shape == (26, 3)
    assert np.allclose(np.linalg.norm(locations, axis=1), 1.0)
    expected_fp1 = np.array([-26.81, 84.06, -10.56])
    expected_fp1 /= np.linalg.norm(expected_fp1)
    assert np.allclose(locations[0], expected_fp1)
    assert config.monitor.channels == (
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
        "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
        "O1", "Oz", "O2",
    )
    monitor, _ = build_eeg_monitor(config.monitor, 200, 2.0)
    assert monitor.obsnoise is None
    montage = mne_sensor_montage(config.monitor)
    assert set(config.monitor.channels).issubset(montage.get_positions()["ch_pos"])
    audit = sensor_coordinate_audit(config.monitor)
    assert audit["median_angular_difference_from_mne_degrees"] < 2.0
    assert audit["maximum_angular_difference_from_mne_degrees"] < 4.0


def test_centroid_gain_regularization_and_average_reference() -> None:
    config = load_config(Path("configs/baseline.toml"))
    connectome = load_connectome(config.paths, config.connectivity)
    monitor, _ = build_eeg_monitor(config.monitor, 200, 2.0)
    audit = regularize_analytic_eeg_gain(
        monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        config.monitor.minimum_source_sensor_distance_mm,
    )
    assert audit["minimum_unregularized_distance_mm"] < 20.0
    assert audit["regularized_source_sensor_pairs"] == 2
    assert monitor.gain.shape == (26, 200)
    regional = np.arange(400, dtype=float).reshape(2, 200)
    eeg = project_regional_psp_to_eeg(
        regional, monitor.gain, config.monitor.reference, config.monitor.channels
    )
    assert np.max(np.abs(eeg.mean(axis=1))) < 1e-15


def test_visualization_highpass_removes_slow_offset_without_erasing_alpha() -> None:
    sfreq = 500.0
    time = np.arange(0.0, 10.0, 1.0 / sfreq)
    slow = np.sin(2.0 * np.pi * 0.2 * time)
    alpha = np.sin(2.0 * np.pi * 10.0 * time)
    filtered_slow = _highpass_for_visualization(slow[:, None], sfreq, 1.0)[:, 0]
    filtered_alpha = _highpass_for_visualization(alpha[:, None], sfreq, 1.0)[:, 0]
    interior = slice(int(sfreq), -int(sfreq))
    assert np.std(filtered_slow[interior]) < 0.01 * np.std(slow[interior])
    assert np.std(filtered_alpha[interior]) > 0.95 * np.std(alpha[interior])


def test_colored_observation_noise_is_reproducible_and_average_referenced() -> None:
    rng = np.random.default_rng(4)
    eeg = rng.normal(size=(2000, 6))
    first = add_colored_observation_noise(eeg, 500.0, 0.4, 0.5, 123)
    second = add_colored_observation_noise(eeg, 500.0, 0.4, 0.5, 123)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, eeg)
    assert np.max(np.abs(first.mean(axis=1))) < 1e-12
