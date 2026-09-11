"""Persist reproducible numerical outputs and baseline quality-control plots."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from .config import RunConfig
from .connectome import ConnectomeBundle
from .simulation import SimulationResult


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


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


def _spectral_summary(eeg: np.ndarray, sfreq_hz: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    nperseg = min(eeg.shape[0], int(round(2.0 * sfreq_hz)))
    frequencies, psd = welch(eeg, fs=sfreq_hz, axis=0, nperseg=nperseg)
    mean_psd = psd.mean(axis=1)
    valid_peak = (frequencies >= 1.0) & (frequencies <= 45.0)
    peak_frequency = float(frequencies[valid_peak][np.argmax(mean_psd[valid_peak])])
    band_power: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        band_power[name] = float(np.trapezoid(mean_psd[mask], frequencies[mask]))
    return frequencies, psd, {
        "dominant_frequency_1_45_hz": peak_frequency,
        "mean_band_power": band_power,
    }


def _plot_summary(
    output_path: Path,
    result: SimulationResult,
    connectome: ConnectomeBundle,
    frequencies: np.ndarray,
    psd: np.ndarray,
) -> None:
    time_s = result.time_ms / 1000.0
    view = time_s <= min(2.0, time_s[-1])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

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

    eeg_scale = np.std(result.eeg)
    eeg_scale = eeg_scale if eeg_scale > 0 else 1.0
    sensor_offsets = np.arange(len(result.channel_names)) * 5.0
    axes[1, 0].plot(
        time_s[view], result.eeg[view] / eeg_scale + sensor_offsets, linewidth=0.45
    )
    axes[1, 0].set_title("Template EEG monitor, average referenced")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_yticks(sensor_offsets[::3])
    axes[1, 0].set_yticklabels(result.channel_names[::3])

    mean_psd = psd.mean(axis=1)
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    axes[1, 1].semilogy(frequencies[mask], mean_psd[mask])
    axes[1, 1].axvspan(8.0, 13.0, color="tab:orange", alpha=0.15, label="alpha")
    axes[1, 1].set_title("Mean sensor PSD")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("PSD (arbitrary units²/Hz)")
    axes[1, 1].legend(loc="best")

    fig.suptitle("MDD-TVB baseline simulation QC", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_run(config: RunConfig, connectome: ConnectomeBundle, result: SimulationResult) -> Path:
    """Save arrays, metadata, spectra, labels, and a compact QC figure."""

    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sfreq_hz = 1000.0 / config.simulation.monitor_period_ms
    frequencies, psd, spectral = _spectral_summary(result.eeg, sfreq_hz)

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
        "simulation": result.metadata,
        "spectral_qc": spectral,
        "limitations": [
            "The structural connectome is common to all future subjects.",
            "The EEG gain matrix uses TVB's analytic single-sphere approximation.",
            "Regional orientations are radial approximations derived from parcel centroids.",
            "Signal amplitude is in model/arbitrary units and is not calibrated to microvolts.",
            "No FEM field, parameter fit, plasticity, or protocol optimization is included.",
            "The headerless connectome matrix order is assumed to match the official Schaefer ordering used by its source project; the files themselves cannot prove this mapping.",
        ],
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(metadata), handle, indent=2)

    _plot_summary(
        output_dir / "baseline_summary.png", result, connectome, frequencies, psd
    )
    return output_dir
