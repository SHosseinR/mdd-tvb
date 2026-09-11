from dataclasses import replace
from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.simulation import run_baseline


def test_short_tvb_simulation_and_eeg_monitor() -> None:
    config = load_config(Path("configs/baseline.toml"))
    short_simulation = replace(
        config.simulation,
        duration_ms=120.0,
        transient_ms=40.0,
        monitor_period_ms=2.0,
    )
    no_csd_monitor = replace(config.monitor, surface_laplacian=False)
    config = replace(config, simulation=short_simulation, monitor=no_csd_monitor)
    connectome = load_connectome(config.paths, config.connectivity)
    result = run_baseline(config, connectome)
    assert result.region_psp.shape == (40, 200)
    assert result.eeg.shape == (40, 26)
    assert result.gain_matrix.shape == (26, 200)
    assert np.isfinite(result.region_psp).all()
    assert np.isfinite(result.eeg).all()
    assert np.max(np.abs(result.eeg.mean(axis=1))) < 1e-9
    assert result.metadata["observation_noise"] == "disabled"
    assert result.metadata["noise_target_state"] == "y4_only"
    assert result.metadata["analytic_gain_regularization"]["regularized_source_sensor_pairs"] == 2
    assert result.regional_parameters.mu.shape == (200,)
    assert np.ptp(result.regional_parameters.mu) > 0
