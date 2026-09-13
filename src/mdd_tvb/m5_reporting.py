"""Transparent waveform examples for the completed M5 fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .empirical import load_empirical_eeg
from .features import load_feature_collection
from .fit_config import M5Config
from .m5 import load_simulation_bank, simulate_candidate_eeg
from .parameterization import CandidateParameters


DISPLAY_CHANNELS: tuple[str, ...] = ("Fz", "Cz", "Pz", "Oz")


def _candidate(index: int, values: np.ndarray) -> CandidateParameters:
    return CandidateParameters(
        candidate_index=index,
        global_coupling=float(values[0]),
        mu=float(values[1]),
        a_scale=float(values[2]),
        b_scale=float(values[3]),
        noise_nsig=float(values[4]),
        regional_time_log_sd=float(values[5]),
        network_gains=np.asarray(values[6:], dtype=float),
    )


def _display_segment(
    eeg: np.ndarray,
    sfreq_hz: float,
    channel_indices: list[int],
    start_seconds: float,
    duration_seconds: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    sos = butter(4, 1.0, btype="highpass", fs=sfreq_hz, output="sos")
    filtered = sosfiltfilt(sos, eeg, axis=0)
    start = int(round(start_seconds * sfreq_hz))
    count = int(round(duration_seconds * sfreq_hz))
    segment = filtered[start:start + count, channel_indices]
    if len(segment) != count:
        raise ValueError("EEG is too short for the requested display segment")
    segment -= segment.mean(axis=0, keepdims=True)
    scale = segment.std(axis=0, ddof=1, keepdims=True)
    segment /= np.maximum(scale, np.finfo(float).eps)
    offsets = np.arange(len(channel_indices), dtype=float) * 4.0
    stacked = segment + offsets[np.newaxis, :]
    time = np.arange(count) / sfreq_hz
    return time, stacked


def create_eeg_example_figure(config: M5Config) -> pd.DataFrame:
    """Rerun representative fitted candidates and plot raw-looking EEG examples."""

    output_dir = config.paths.output_dir
    fit_dir = output_dir / "fit"
    subjects = pd.read_csv(fit_dir / "subject_fits.csv")
    collection = load_feature_collection(
        output_dir / "empirical" / "empirical_features.npz"
    )
    bank = load_simulation_bank(output_dir / "bank" / "simulation_bank.npz")
    source_by_subject = dict(
        zip(collection.subject_ids.astype(str), collection.source_files.astype(str))
    )

    representatives: list[pd.Series] = []
    for group in config.empirical.groups:
        available = subjects[
            (subjects["group"] == group) & (subjects["split"] == "holdout")
        ]
        median_ratio = float(available["validation_cost_ratio"].median())
        row = available.iloc[
            np.argmin(np.abs(available["validation_cost_ratio"] - median_ratio))
        ]
        representatives.append(row)

    simulated_cache: dict[int, tuple[np.ndarray, float, tuple[str, ...]]] = {}
    figure, axes = plt.subplots(
        3, len(representatives), figsize=(15, 10), sharex=True,
        constrained_layout=True,
    )
    audit_rows: list[dict[str, object]] = []
    for column, row in enumerate(representatives):
        subject_id = str(row["subject_id"])
        group = str(row["group"])
        candidate_index = int(row["candidate_index"])
        empirical, empirical_sfreq = load_empirical_eeg(
            source_by_subject[subject_id],
            str(config.paths.baseline_config),
            config.empirical.max_duration_s,
            config.empirical.apply_surface_laplacian,
        )
        if candidate_index not in simulated_cache:
            simulated_cache[candidate_index] = simulate_candidate_eeg(
                str(config.paths.baseline_config),
                _candidate(candidate_index, bank.parameters[candidate_index]),
                0,
                config,
            )
        simulated, simulated_sfreq, channel_names = simulated_cache[candidate_index]
        empirical_channel_names = channel_names
        channel_indices = [
            empirical_channel_names.index(name) for name in DISPLAY_CHANNELS
        ]
        midpoint_seconds = empirical.shape[0] / empirical_sfreq / 2.0
        panels = (
            (empirical, empirical_sfreq, 5.0, "Empirical fitting half"),
            (
                empirical,
                empirical_sfreq,
                midpoint_seconds + 5.0,
                "Empirical unseen half",
            ),
            (simulated, simulated_sfreq, 2.0, "Selected TVB simulation"),
        )
        for row_index, (data, sfreq, start, label) in enumerate(panels):
            time, stacked = _display_segment(
                data, sfreq, channel_indices, start
            )
            axis = axes[row_index, column]
            axis.plot(time, stacked, color=f"C{column}", linewidth=0.7)
            axis.set_yticks(np.arange(len(DISPLAY_CHANNELS)) * 4.0)
            axis.set_yticklabels(DISPLAY_CHANNELS)
            axis.set_ylabel(f"{label}\n(z-score + offset)")
            axis.grid(axis="x", alpha=0.2)
        axes[0, column].set_title(
            f"{group}: {subject_id}\ncandidate {candidate_index}; "
            f"unseen/reference cost ratio={row['validation_cost_ratio']:.3f}"
        )
        axes[-1, column].set_xlabel("Time within displayed segment (s)")
        audit_rows.append({
            "group": group,
            "subject_id": subject_id,
            "split": str(row["split"]),
            "candidate_index": candidate_index,
            "fit_cost_ratio": float(row["fit_cost_ratio"]),
            "validation_cost_ratio": float(row["validation_cost_ratio"]),
            "source_file": source_by_subject[subject_id],
            "display_filter": "zero-phase 1 Hz high-pass; per-channel z-score",
        })

    figure.suptitle(
        "Representative held-out M5 EEG: empirical halves vs fitted TVB model",
        fontsize=14,
    )
    output_path = fit_dir / "m5_eeg_examples.png"
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    temporary.replace(output_path)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(fit_dir / "m5_eeg_examples.csv", index=False)
    return audit
