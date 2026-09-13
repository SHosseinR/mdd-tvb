"""Persist reproducible numerical outputs and baseline quality-control plots."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, sosfiltfilt

from .config import MonitorConfig, RunConfig
from .connectome import ConnectomeBundle
from .eeg import mne_sensor_montage, sensor_coordinate_audit
from .simulation import SimulationResult


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


def _highpass_for_visualization(
    eeg: np.ndarray, sfreq_hz: float, cutoff_hz: float
) -> np.ndarray:
    """Zero-phase high-pass used only for readable EEG trace display."""

    sos = butter(4, cutoff_hz, btype="highpass", fs=sfreq_hz, output="sos")
    return sosfiltfilt(sos, eeg, axis=0)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _spectral_summary(
    eeg: np.ndarray,
    sfreq_hz: float,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Remove the channel equilibrium potential before tapering so that a large
    # model DC term cannot leak into the low-frequency spectrum.
    demeaned_eeg = eeg - eeg.mean(axis=0, keepdims=True)
    psd_channels, frequencies = psd_array_multitaper(
        demeaned_eeg.T,
        sfreq=sfreq_hz,
        fmin=0.5,
        fmax=min(80.0, sfreq_hz / 2.0),
        bandwidth=1.0,
        adaptive=True,
        normalization="full",
        verbose="ERROR",
    )
    psd = psd_channels.T
    mean_psd = psd.mean(axis=1)
    valid_peak = (frequencies >= 1.0) & (frequencies <= 45.0)
    peak_frequency = float(frequencies[valid_peak][np.argmax(mean_psd[valid_peak])])
    band_power: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_power[name] = float(np.trapezoid(mean_psd[mask], frequencies[mask]))
    spectrum = mean_psd[valid_peak]
    normalized = spectrum / spectrum.sum()
    spectral_entropy = float(
        -(normalized * np.log(normalized + np.finfo(float).tiny)).sum()
        / np.log(normalized.size)
    )
    alpha_channel_power = np.trapezoid(
        psd[(frequencies >= 8.0) & (frequencies < 13.0)],
        frequencies[(frequencies >= 8.0) & (frequencies < 13.0)],
        axis=0,
    )
    alpha_spectrum = mean_psd[(frequencies >= 8.0) & (frequencies < 13.0)]
    alpha_frequencies = frequencies[(frequencies >= 8.0) & (frequencies < 13.0)]
    alpha_peak_index = int(np.argmax(alpha_spectrum))
    half_max = alpha_spectrum[alpha_peak_index] / 2.0
    above_half = alpha_frequencies[alpha_spectrum >= half_max]
    peak_width = float(above_half.max() - above_half.min()) if above_half.size > 1 else 0.0

    primary_power = float(mean_psd[np.argmin(np.abs(frequencies - peak_frequency))])
    harmonic_ratios: dict[str, float | None] = {}
    for order in (2, 3, 4):
        harmonic_frequency = order * peak_frequency
        if harmonic_frequency > 45.0:
            harmonic_ratios[f"{order}x"] = None
            continue
        harmonic_window = np.abs(frequencies - harmonic_frequency) <= 0.75
        harmonic_power = float(np.max(mean_psd[harmonic_window]))
        harmonic_ratios[f"{order}x"] = harmonic_power / primary_power

    name_to_index = {name: index for index, name in enumerate(channel_names)}
    posterior_names = ("P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")
    anterior_names = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")
    posterior_power = np.mean([alpha_channel_power[name_to_index[name]] for name in posterior_names])
    anterior_power = np.mean([alpha_channel_power[name_to_index[name]] for name in anterior_names])
    return frequencies, psd, {
        "dominant_frequency_1_45_hz": peak_frequency,
        "mean_band_power": band_power,
        "normalized_spectral_entropy_1_45": spectral_entropy,
        "alpha_peak_half_max_width_hz": peak_width,
        "harmonic_to_fundamental_power_ratio": harmonic_ratios,
        "posterior_to_anterior_alpha_power_ratio": float(posterior_power / anterior_power),
        "alpha_channel_power": alpha_channel_power.tolist(),
    }


def _dynamics_summary(result: SimulationResult) -> dict[str, float]:
    regional_correlation = np.corrcoef(result.region_psp.T)
    upper = np.abs(regional_correlation[np.triu_indices_from(regional_correlation, k=1)])
    centered_eeg = result.eeg - result.eeg.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered_eeg, compute_uv=False)
    variance = singular**2
    return {
        "median_absolute_regional_correlation": float(np.median(upper)),
        "p95_absolute_regional_correlation": float(np.percentile(upper, 95)),
        "eeg_pc1_variance_fraction": float(variance[0] / variance.sum()),
        "eeg_pc1_to_pc5_variance_fraction": float(variance[:5].sum() / variance.sum()),
    }


def _plot_summary(
    output_path: Path,
    result: SimulationResult,
    connectome: ConnectomeBundle,
    frequencies: np.ndarray,
    psd: np.ndarray,
    monitor_settings: MonitorConfig,
    visualization_highpass_hz: float = 1.0,
) -> None:
    time_s = result.time_ms / 1000.0
    view = time_s <= min(2.0, time_s[-1])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    image = axes[0, 0].imshow(connectome.weights, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("Scaled Schaefer-200 structural weights")
    axes[0, 0].set_xlabel("Source region")
    axes[0, 0].set_ylabel("Target region")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.8)

    offsets = np.arange(min(8, result.region_psp.shape[1])) * 15.0
    axes[0, 1].plot(
        time_s[view], result.region_psp[view, : offsets.size] + offsets, linewidth=0.7
    )
    axes[0, 1].set_title("Regional pyramidal PSPs (first 8 regions)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Arbitrary units + offset")

    sfreq_hz = 1000.0 / float(np.median(np.diff(result.time_ms)))
    eeg_display = _highpass_for_visualization(
        result.eeg, sfreq_hz, visualization_highpass_hz
    )
    eeg_scale = float(np.median(np.std(eeg_display, axis=0)))
    eeg_scale = eeg_scale if eeg_scale > 0 else 1.0
    sensor_offsets = np.arange(len(result.channel_names)) * 5.0
    axes[1, 0].plot(
        time_s[view], eeg_display[view] / eeg_scale + sensor_offsets, linewidth=0.45
    )
    axes[1, 0].set_title(
        f"Template EEG monitor ({visualization_highpass_hz:g} Hz high-pass for display)"
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_yticks(sensor_offsets[::3])
    axes[1, 0].set_yticklabels(result.channel_names[::3])

    mean_psd = psd.mean(axis=1)
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    axes[0, 2].semilogy(frequencies[mask], mean_psd[mask])
    axes[0, 2].axvspan(8.0, 13.0, color="tab:orange", alpha=0.15, label="alpha")
    axes[0, 2].set_title("Multitaper mean sensor PSD")
    axes[0, 2].set_xlabel("Frequency (Hz)")
    axes[0, 2].set_ylabel("PSD (arbitrary units²/Hz)")
    axes[0, 2].legend(loc="best")

    alpha_mask = (frequencies >= 8.0) & (frequencies < 13.0)
    alpha_power = np.trapezoid(psd[alpha_mask], frequencies[alpha_mask], axis=0)
    import mne
    info = mne.create_info(list(result.channel_names), sfreq=sfreq_hz, ch_types="eeg")
    info.set_montage(mne_sensor_montage(monitor_settings), on_missing="raise")
    mne.viz.plot_topomap(
        alpha_power,
        info,
        axes=axes[1, 1],
        show=False,
        contours=6,
        cmap="magma",
    )
    axes[1, 1].set_title("8–13 Hz sensor power")

    network_names = np.unique(result.regional_parameters.network_labels)
    network_alpha = []
    region_psd, region_frequencies = psd_array_multitaper(
        result.region_psp.T,
        sfreq=sfreq_hz,
        fmin=0.5,
        fmax=45.0,
        bandwidth=1.0,
        adaptive=False,
        normalization="full",
        verbose="ERROR",
    )
    region_alpha_mask = (region_frequencies >= 8.0) & (region_frequencies < 13.0)
    region_alpha = np.trapezoid(
        region_psd[:, region_alpha_mask], region_frequencies[region_alpha_mask], axis=1
    )
    for network_name in network_names:
        network_alpha.append(
            float(np.mean(region_alpha[result.regional_parameters.network_labels == network_name]))
        )
    axes[1, 2].bar(network_names, network_alpha, color="tab:purple", alpha=0.8)
    axes[1, 2].set_title("Source alpha power by Schaefer network")
    axes[1, 2].set_ylabel("Mean power (arbitrary units²)")
    axes[1, 2].tick_params(axis="x", rotation=45)

    fig.suptitle("MDD-TVB baseline simulation QC", fontsize=14)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary_path, dpi=160)
    plt.close(fig)
    os.replace(temporary_path, output_path)


def save_run(config: RunConfig, connectome: ConnectomeBundle, result: SimulationResult) -> Path:
    """Save arrays, metadata, spectra, labels, and a compact QC figure."""

    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sfreq_hz = 1000.0 / config.simulation.monitor_period_ms
    frequencies, psd, spectral = _spectral_summary(
        result.eeg, sfreq_hz, result.channel_names
    )

    arrays: dict[str, np.ndarray] = {
        "time_ms": result.time_ms,
        "region_psp": result.region_psp,
        "eeg": result.eeg,
        "gain_matrix": result.gain_matrix,
    }
    if result.eeg_surface_laplacian is not None:
        arrays["eeg_surface_laplacian"] = result.eeg_surface_laplacian
    np.savez_compressed(output_dir / "baseline_timeseries.npz", **arrays)
    np.save(output_dir / "scaled_connectome.npy", connectome.weights)
    np.save(output_dir / "tract_lengths_mm.npy", connectome.tract_lengths)

    pd.DataFrame(
        {"region_index": np.arange(result.region_labels.size), "label": result.region_labels}
    ).to_csv(output_dir / "regions.csv", index=False)
    pd.DataFrame({
        "region_index": np.arange(result.region_labels.size),
        "label": result.region_labels,
        "network": result.regional_parameters.network_labels,
        "mu": result.regional_parameters.mu,
        "a": result.regional_parameters.a,
        "b": result.regional_parameters.b,
        "noise_nsig": result.regional_parameters.noise_nsig,
        "drive_multiplier": result.regional_parameters.drive_multiplier,
        "time_scale_multiplier": result.regional_parameters.time_scale_multiplier,
        "noise_multiplier": result.regional_parameters.noise_multiplier,
    }).to_csv(output_dir / "regional_parameters.csv", index=False)
    pd.DataFrame(
        {"sensor_index": np.arange(len(result.channel_names)), "label": result.channel_names}
    ).to_csv(output_dir / "sensors.csv", index=False)
    psd_table = pd.DataFrame(psd, columns=result.channel_names)
    psd_table.insert(0, "frequency_hz", frequencies)
    psd_table.to_csv(output_dir / "eeg_psd.csv", index=False)

    metadata = {
        "software": {
            "tvb-library": version("tvb-library"),
            "mne": version("mne"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "config": _json_value(asdict(config)),
        "connectome_audit": connectome.audit,
        "sensor_coordinate_audit": sensor_coordinate_audit(config.monitor),
        "simulation": result.metadata,
        "spectral_qc": spectral,
        "dynamics_qc": _dynamics_summary(result),
        "reporting": {
            "stored_eeg": "raw average-referenced model observation",
            "spectral_preprocessing": "per-channel temporal mean removed",
            "trace_visualization_highpass_hz": config.monitor.visualization_highpass_hz,
            "trace_visualization_filter": "fourth-order zero-phase Butterworth",
        },
        "limitations": [
            "The structural connectome is common to all future subjects.",
            "The EEG gain matrix uses TVB's analytic single-sphere approximation.",
            "Sensor coordinates are the TDBRAIN publication-level Table 3 positions, not per-subject digitization.",
            "Regional orientations are radial approximations derived from parcel centroids.",
            f"The centroid-based analytic gain uses a declared {config.monitor.minimum_source_sensor_distance_mm:g} mm near-field distance floor; it is not a substitute for a surface BEM/FEM lead field.",
            "Signal amplitude is in model/arbitrary units and is not calibrated to microvolts.",
            "No FEM field, parameter fit, plasticity, or protocol optimization is included.",
            "The headerless connectome matrix order is assumed to match the official Schaefer ordering used by its source project; the files themselves cannot prove this mapping.",
        ],
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metadata), handle, indent=2)

    _plot_summary(
        output_dir / "baseline_summary.png",
        result,
        connectome,
        frequencies,
        psd,
        config.monitor,
        config.monitor.visualization_highpass_hz,
    )
    return output_dir
