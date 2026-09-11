"""Regenerate the baseline QC figure without rerunning the TVB simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.heterogeneity import build_regional_parameters
from mdd_tvb.reporting import _plot_summary, _spectral_summary
from mdd_tvb.simulation import SimulationResult


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or config.paths.output_dir / "baseline_timeseries.npz"
    output_path = args.output or config.paths.output_dir / "baseline_summary.png"
    connectome = load_connectome(config.paths, config.connectivity)
    regional = build_regional_parameters(
        config.model,
        config.simulation.noise_nsig,
        connectome.region_labels,
        config.heterogeneity,
    )

    with np.load(input_path) as arrays:
        surface_laplacian = (
            arrays["eeg_surface_laplacian"]
            if "eeg_surface_laplacian" in arrays.files
            else None
        )
        result = SimulationResult(
            time_ms=arrays["time_ms"],
            region_psp=arrays["region_psp"],
            eeg=arrays["eeg"],
            eeg_surface_laplacian=surface_laplacian,
            gain_matrix=arrays["gain_matrix"],
            channel_names=config.monitor.channels,
            region_labels=connectome.region_labels,
            regional_parameters=regional,
            metadata={},
        )

    sfreq_hz = 1000.0 / float(np.median(np.diff(result.time_ms)))
    frequencies, psd, _ = _spectral_summary(
        result.eeg, sfreq_hz, result.channel_names
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_summary(
        output_path,
        result,
        connectome,
        frequencies,
        psd,
        config.monitor.visualization_highpass_hz,
    )
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
