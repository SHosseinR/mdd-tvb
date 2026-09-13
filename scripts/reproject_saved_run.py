"""Reproject saved regional activity after an EEG forward-model change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.eeg import (
    apply_surface_laplacian,
    build_eeg_monitor,
    project_regional_psp_to_eeg,
    regularize_analytic_eeg_gain,
)
from mdd_tvb.heterogeneity import build_regional_parameters
from mdd_tvb.reporting import save_run
from mdd_tvb.simulation import SimulationResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or config.paths.output_dir / "baseline_timeseries.npz"
    metadata_path = input_path.parent / "run_metadata.json"
    with np.load(input_path) as arrays:
        time_ms = arrays["time_ms"].copy()
        region_psp = arrays["region_psp"].copy()

    connectome = load_connectome(config.paths, config.connectivity)
    monitor, _ = build_eeg_monitor(
        config.monitor,
        config.connectivity.expected_regions,
        config.simulation.monitor_period_ms,
    )
    gain_audit = regularize_analytic_eeg_gain(
        monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        config.monitor.minimum_source_sensor_distance_mm,
    )
    eeg = project_regional_psp_to_eeg(
        region_psp, monitor.gain, config.monitor.reference, config.monitor.channels
    )
    sfreq_hz = 1000.0 / float(np.median(np.diff(time_ms)))
    eeg_csd = (
        apply_surface_laplacian(
            eeg, config.monitor.channels, sfreq_hz, config.monitor
        )
        if config.monitor.surface_laplacian
        else None
    )
    regional = build_regional_parameters(
        config.model,
        config.simulation.noise_nsig,
        connectome.region_labels,
        config.heterogeneity,
    )
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["simulation"]
    simulation_metadata = {
        **old_metadata,
        "eeg_reprojected_from_saved_regional_psp": True,
        "analytic_gain_regularization": gain_audit,
        "average_reference_max_abs_mean": float(np.max(np.abs(eeg.mean(axis=1)))),
        "eeg_standard_deviation": float(eeg.std()),
        "gain_shape": list(monitor.gain.shape),
    }
    result = SimulationResult(
        time_ms=time_ms,
        region_psp=region_psp,
        eeg=eeg,
        eeg_surface_laplacian=eeg_csd,
        gain_matrix=monitor.gain.copy(),
        channel_names=config.monitor.channels,
        region_labels=connectome.region_labels,
        regional_parameters=regional,
        metadata=simulation_metadata,
    )
    output_dir = save_run(config, connectome, result)
    print(f"Reprojected and saved {output_dir.resolve()}")


if __name__ == "__main__":
    main()
