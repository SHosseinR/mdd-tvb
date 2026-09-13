"""One-factor sensitivity of network-local Jansen--Rit physiology."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import logging
import numpy as np
import pandas as pd

from .config import load_config
from .connectome import load_connectome, with_network_endpoint_gains
from .connectome_sensitivity import _candidate_from_values, _cosine, _exact_blocks
from .eeg import add_colored_observation_noise, apply_surface_laplacian
from .features import extract_eeg_features, load_feature_collection, load_feature_transformer
from .fit_config import load_m5_config
from .heterogeneity import network_labels
from .m5 import build_candidate_run_config
from .parameterization import NETWORK_ORDER
from .simulation import run_baseline


FAMILIES: tuple[str, ...] = ("drive", "time_scale", "neural_noise")


def _mean_preserving_network_map(
    networks: np.ndarray, target: str, target_gain: float
) -> dict[str, float]:
    target_count = int(np.sum(networks == target))
    other_count = int(networks.size - target_count)
    other_gain = (networks.size - target_count * target_gain) / other_count
    return {
        name: float(target_gain if name == target else other_gain)
        for name in NETWORK_ORDER
    }


def _simulate_regional_mode(
    config_path: str,
    candidate_index: int,
    candidate_values: np.ndarray,
    family: str,
    network: str,
    gain: float,
) -> dict[str, Any]:
    logging.raiseExceptions = False
    m5_config = load_m5_config(config_path)
    baseline = load_config(m5_config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    candidate = _candidate_from_values(candidate_index, candidate_values)
    connectome = with_network_endpoint_gains(
        connectome,
        networks,
        {
            name: float(value)
            for name, value in zip(NETWORK_ORDER, candidate.network_gains, strict=True)
        },
    )
    run_config = build_candidate_run_config(baseline, candidate, m5_config)
    mapping = _mean_preserving_network_map(networks, network, gain)
    updates: dict[str, dict[str, float]] = {
        "network_drive_multipliers": {},
        "network_time_scale_multipliers": {},
        "network_noise_multipliers": {},
    }
    key = {
        "drive": "network_drive_multipliers",
        "time_scale": "network_time_scale_multipliers",
        "neural_noise": "network_noise_multipliers",
    }[family]
    updates[key] = mapping
    run_config = replace(
        run_config,
        heterogeneity=replace(run_config.heterogeneity, **updates),
    )
    result = run_baseline(run_config, connectome)
    sfreq_hz = 1000.0 / run_config.simulation.monitor_period_ms
    eeg = add_colored_observation_noise(
        result.eeg,
        sfreq_hz,
        m5_config.design.observation_noise_fraction,
        m5_config.design.observation_noise_exponent,
        run_config.simulation.seed + 700001,
    )
    if m5_config.empirical.apply_surface_laplacian:
        eeg = apply_surface_laplacian(
            eeg, result.channel_names, sfreq_hz, run_config.monitor
        )
    feature = extract_eeg_features(eeg, sfreq_hz, m5_config.features)
    return {
        "family": family,
        "network": network,
        "gain": gain,
        "psd_shape": feature.psd_shape,
        "alpha_topography": feature.alpha_topography,
        "coherence": feature.coherence,
        "alpha_peak_hz": feature.alpha_peak_hz,
        "alpha_power_fraction": feature.alpha_power_fraction,
        "spectral_entropy": feature.spectral_entropy,
    }


def _plot(path: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    blocks = ["PSD shape", "Alpha topography", "Coherence alpha", "Coherence beta"]
    modes = [f"{family}: {network}" for family in FAMILIES for network in NETWORK_ORDER]
    sensitivity = np.asarray([
        [
            summary[(summary["mode"] == mode) & (summary["block"] == block)][
                "standardized_sensitivity_per_unit_gain"
            ].iloc[0]
            for block in blocks
        ]
        for mode in modes
    ])
    alignment = np.asarray([
        [
            summary[(summary["mode"] == mode) & (summary["block"] == block)][
                "cosine_to_empirical_mdd_direction"
            ].iloc[0]
            for block in blocks
        ]
        for mode in modes
    ])
    figure, axes = plt.subplots(1, 2, figsize=(13, 10), constrained_layout=True)
    first = axes[0].imshow(sensitivity, aspect="auto", cmap="viridis")
    axes[0].set_title("Observable sensitivity")
    axes[0].set_xticks(range(len(blocks)), blocks, rotation=25, ha="right")
    axes[0].set_yticks(range(len(modes)), modes)
    figure.colorbar(first, ax=axes[0], label="standardized RMS / unit gain")
    second = axes[1].imshow(alignment, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Direction relative to empirical MDD contrast")
    axes[1].set_xticks(range(len(blocks)), blocks, rotation=25, ha="right")
    axes[1].set_yticks(range(len(modes)), modes)
    figure.colorbar(second, ax=axes[1], label="cosine (positive target-network gain)")
    figure.suptitle("Network-local physiology sensitivity around the revised M5 working point")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_regional_sensitivity(
    config_path: Path,
    output_dir: Path,
    *,
    candidate_index: int = 3,
    deviation: float = 0.10,
    n_jobs: int = 6,
) -> pd.DataFrame:
    config = load_m5_config(config_path)
    fitting = load_feature_collection(
        config.paths.output_dir / "empirical" / "empirical_features.npz"
    )
    with np.load(config.paths.output_dir / "bank" / "simulation_bank.npz") as arrays:
        bank = {name: arrays[name] for name in arrays.files}
    transformer = load_feature_transformer(
        config.paths.output_dir / "fit" / "feature_transformer.npz"
    )
    split = pd.read_csv(config.paths.output_dir / "fit" / "data_split.csv")
    train = np.flatnonzero(split["split"].to_numpy() == "train")
    raw_mean = fitting.coherence[train].mean(axis=0)
    raw_scale = fitting.coherence[train].std(axis=0, ddof=1)
    positive = raw_scale[raw_scale > 0]
    raw_scale = np.maximum(raw_scale, float(np.median(positive) * 1e-3))
    candidate_values = np.asarray(bank["parameters"][candidate_index], dtype=float)

    jobs = [
        (family, network, 1.0 + sign * deviation)
        for family in FAMILIES
        for network in NETWORK_ORDER
        for sign in (-1.0, 1.0)
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                _simulate_regional_mode,
                str(config_path),
                candidate_index,
                candidate_values,
                family,
                network,
                gain,
            ): (family, network, gain)
            for family, network, gain in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            family, network, gain = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - expensive runtime path
                failures.append({
                    "family": family,
                    "network": network,
                    "gain": gain,
                    "error": repr(exc),
                })
            if completed % max(n_jobs, 1) == 0 or completed == len(futures):
                print(f"regional sensitivity: {completed}/{len(futures)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(output_dir / "regional_sensitivity_failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} regional-sensitivity simulations failed")

    empirical_blocks = _exact_blocks(fitting, transformer, raw_mean, raw_scale)
    baseline_source = {
        name: bank[name][candidate_index]
        for name in (
            "psd_shape", "alpha_topography", "coherence", "alpha_peak_hz",
            "alpha_power_fraction", "spectral_entropy",
        )
    }
    baseline_blocks = _exact_blocks(baseline_source, transformer, raw_mean, raw_scale)
    labels = (fitting.groups == "MDD").astype(int)
    contrasts = {
        block: values[train][labels[train] == 1].mean(axis=0)
        - values[train][labels[train] == 0].mean(axis=0)
        for block, values in empirical_blocks.items()
    }
    by_key = {
        (str(row["family"]), str(row["network"]), float(row["gain"])): row
        for row in results
    }
    rows: list[dict[str, float | str]] = []
    for family in FAMILIES:
        for network in NETWORK_ORDER:
            minus = _exact_blocks(
                by_key[(family, network, 1.0 - deviation)], transformer, raw_mean, raw_scale
            )
            plus = _exact_blocks(
                by_key[(family, network, 1.0 + deviation)], transformer, raw_mean, raw_scale
            )
            for block in baseline_blocks:
                derivative = (plus[block][0] - minus[block][0]) / (2.0 * deviation)
                alignment = _cosine(derivative, contrasts[block])
                rows.append({
                    "mode": f"{family}: {network}",
                    "family": family,
                    "network": network,
                    "block": block,
                    "standardized_sensitivity_per_unit_gain": float(
                        np.linalg.norm(derivative) / np.sqrt(derivative.size)
                    ),
                    "cosine_to_empirical_mdd_direction": alignment,
                    "preferred_gain_direction": "increase" if alignment >= 0 else "decrease",
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "regional_parameter_sensitivity.csv", index=False)
    _plot(output_dir / "regional_parameter_sensitivity.png", summary)
    return summary
