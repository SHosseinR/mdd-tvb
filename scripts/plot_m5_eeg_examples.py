"""Plot representative empirical EEG and rerun fitted TVB MAP simulations."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.empirical import load_empirical_eeg
from mdd_tvb.heterogeneity import network_labels
from mdd_tvb.multifrequency import run_dual_jansen_rit
from mdd_tvb.spectral_bank import (
    build_spectral_run_config,
    connectome_for_spectral_candidate,
    load_spectral_simulation_bank,
)
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import load_cross_spectral_collection
from mdd_tvb.spectral_parameterization import PARAMETER_NAMES, SpectralCandidate


DISPLAY_CHANNELS = ("Fp1", "Fz", "Cz", "Pz", "O1", "O2")


def _display_signal(eeg: np.ndarray, sfreq_hz: float) -> np.ndarray:
    sos = butter(4, 1.0, btype="highpass", fs=sfreq_hz, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(eeg, dtype=float), axis=0)
    scale = np.std(filtered, axis=0, ddof=1)
    return filtered / np.maximum(scale, np.finfo(float).tiny)


def _add_colored_observation_noise(
    eeg: np.ndarray,
    sfreq_hz: float,
    fraction: float,
    exponent: float,
    seed: int,
) -> np.ndarray:
    """Create a time-domain view of the fitted diagonal CSD nuisance."""

    signal = np.asarray(eeg, dtype=float)
    if fraction <= 0.0:
        return signal.copy()
    rng = np.random.default_rng(seed)
    frequency = np.fft.rfftfreq(signal.shape[0], d=1.0 / sfreq_hz)
    floor = max(1.0 / (signal.shape[0] / sfreq_hz), 0.25)
    amplitude = np.maximum(frequency, floor) ** (-0.5 * exponent)
    amplitude[0] = 0.0
    spectrum = (
        rng.normal(size=(frequency.size, signal.shape[1]))
        + 1j * rng.normal(size=(frequency.size, signal.shape[1]))
    ) * amplitude[:, np.newaxis]
    noise = np.fft.irfft(spectrum, n=signal.shape[0], axis=0)
    noise -= noise.mean(axis=0, keepdims=True)
    noise /= np.maximum(noise.std(axis=0, ddof=1), np.finfo(float).tiny)
    centered = signal - signal.mean(axis=0, keepdims=True)
    signal_variance = float(np.mean(centered**2))
    ratio = fraction / (1.0 - fraction)
    noise *= np.sqrt(max(signal_variance, np.finfo(float).tiny) * ratio)
    return signal + noise


def _stacked(
    axis: plt.Axes,
    time_s: np.ndarray,
    eeg: np.ndarray,
    indices: list[int],
    title: str,
) -> None:
    spacing = 5.0
    for row, index in enumerate(indices):
        axis.plot(time_s, eeg[:, index] - row * spacing, linewidth=0.65)
    axis.set_yticks(-np.arange(len(indices)) * spacing)
    axis.set_yticklabels(DISPLAY_CHANNELS)
    axis.set_xlim(time_s[0], time_s[-1])
    axis.set_xlabel("Time (s)")
    axis.set_title(title)


def main(config_path: str = "configs/m5_spectral.toml") -> None:
    config = load_spectral_m5_config(config_path)
    fit_dir = config.paths.output_dir / "fit"
    table = pd.read_csv(fit_dir / "subject_posteriors.csv")
    bank = load_spectral_simulation_bank(
        config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    empirical_collection = load_cross_spectral_collection(
        config.paths.output_dir / "empirical" / "cross_spectra_fit.npz"
    )
    baseline = load_config(config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    channel_indices = [baseline.monitor.channels.index(name) for name in DISPLAY_CHANNELS]
    examples: list[dict[str, object]] = []
    signals: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
    example_config = replace(
        config,
        design=replace(
            config.design,
            duration_ms=12000.0,
            transient_ms=2000.0,
        ),
    )

    for group in config.empirical.groups:
        selected = table[
            (table["group"] == group) & (table["subject_split"] == "holdout")
        ].copy()
        median = float(selected["validation_cost_ratio_to_pooled_null"].median())
        row = selected.iloc[
            np.argmin(
                np.abs(
                    selected["validation_cost_ratio_to_pooled_null"].to_numpy()
                    - median
                )
            )
        ]
        subject_id = str(row["subject_id"])
        empirical_index = int(
            np.flatnonzero(empirical_collection.subject_ids == subject_id)[0]
        )
        source = Path(str(empirical_collection.source_files[empirical_index]))
        empirical, empirical_sfreq = load_empirical_eeg(
            str(source),
            str(config.paths.baseline_config),
            12.0,
            config.empirical.apply_surface_laplacian,
        )
        candidate_index = int(row["map_candidate_index"])
        values = bank.parameters[candidate_index]
        candidate = SpectralCandidate(
            candidate_index=candidate_index,
            **{
                name: float(values[column])
                for column, name in enumerate(PARAMETER_NAMES)
            },
        )
        run_config = build_spectral_run_config(
            baseline, networks, candidate, example_config, replicate=0
        )
        fitted_connectome = connectome_for_spectral_candidate(
            connectome, networks, candidate
        )
        result = run_dual_jansen_rit(
            run_config,
            fitted_connectome,
            fast_ratio=candidate.fast_ratio,
            fast_fraction=candidate.fast_fraction,
            inhibitory_scale=1.0,
            speed_mm_per_ms=candidate.speed_mm_per_ms,
            cross_coupling=example_config.design.cross_coupling,
        )
        simulated_sfreq = 1000.0 / run_config.simulation.monitor_period_ms
        duration = 10.0
        empirical = empirical[: int(round(duration * empirical_sfreq))]
        simulated = result.eeg[: int(round(duration * simulated_sfreq))]
        observation_fraction = float(
            row["observation_noise_fraction_posterior_mean"]
        )
        observation_exponent = float(
            row["observation_noise_exponent_posterior_mean"]
        )
        simulated = _add_colored_observation_noise(
            simulated,
            simulated_sfreq,
            observation_fraction,
            observation_exponent,
            seed=98173 + candidate_index,
        )
        signals[group] = (
            _display_signal(empirical, empirical_sfreq),
            _display_signal(simulated, simulated_sfreq),
            empirical_sfreq,
            simulated_sfreq,
        )
        examples.append(
            {
                "group": group,
                "subject_id": subject_id,
                "source_file": str(source.resolve()),
                "candidate_index": candidate_index,
                "validation_cost_ratio_to_pooled_null": float(
                    row["validation_cost_ratio_to_pooled_null"]
                ),
                "observation_noise_fraction_posterior_mean": observation_fraction,
                "observation_noise_exponent_posterior_mean": observation_exponent,
            }
        )

    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    for column, group in enumerate(config.empirical.groups):
        empirical, simulated, empirical_sfreq, simulated_sfreq = signals[group]
        empirical_time = np.arange(len(empirical)) / empirical_sfreq
        simulated_time = np.arange(len(simulated)) / simulated_sfreq
        _stacked(
            axes[0, column],
            empirical_time,
            empirical,
            channel_indices,
            f"{group}: representative empirical EEG (1-Hz HP for display)",
        )
        _stacked(
            axes[1, column],
            simulated_time,
            simulated,
            channel_indices,
            f"{group}: fitted TVB MAP + posterior-mean observation noise",
        )
        for label, signal, sample_rate in (
            ("empirical", empirical, empirical_sfreq),
            ("TVB + observation", simulated, simulated_sfreq),
        ):
            frequency, power = welch(
                signal,
                fs=sample_rate,
                nperseg=int(4 * sample_rate),
                noverlap=int(2 * sample_rate),
                axis=0,
            )
            mask = (frequency >= 2.0) & (frequency <= 40.0)
            curve = np.log(np.maximum(power[mask].mean(axis=1), np.finfo(float).tiny))
            curve -= curve.mean()
            axes[2, column].plot(frequency[mask], curve, label=label)
        axes[2, column].set_title(f"{group}: centered mean log PSD")
        axes[2, column].set_xlabel("Frequency (Hz)")
        axes[2, column].set_ylabel("Centered log power")
        axes[2, column].legend()
    fig.suptitle(
        "M5 empirical versus fitted stationary EEG examples—waveforms are not time-aligned",
        fontsize=14,
    )
    temporary = fit_dir / ".m5_eeg_examples.tmp.png"
    fig.savefig(temporary, dpi=170)
    plt.close(fig)
    temporary.replace(fit_dir / "m5_eeg_examples.png")
    (fit_dir / "m5_eeg_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/m5_spectral.toml")
    arguments = parser.parse_args()
    main(arguments.config)
