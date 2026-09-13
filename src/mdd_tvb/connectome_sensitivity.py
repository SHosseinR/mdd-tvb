"""One-factor sensitivity of Yeo network-pair connectome perturbations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import logging
import numpy as np
import pandas as pd

from .config import load_config
from .connectome import (
    load_connectome,
    with_network_endpoint_gains,
    with_network_pair_gains,
)
from .features import (
    FeatureCollection,
    FeatureTransformer,
    extract_eeg_features,
    load_feature_collection,
    load_feature_transformer,
)
from .fit_config import M5Config, load_m5_config
from .heterogeneity import network_labels
from .m5 import build_candidate_run_config
from .parameterization import CandidateParameters, NETWORK_ORDER
from .simulation import run_baseline


def _candidate_from_values(index: int, values: np.ndarray) -> CandidateParameters:
    values = np.asarray(values, dtype=float)
    return CandidateParameters(
        candidate_index=index,
        global_coupling=float(values[0]),
        mu=float(values[1]),
        a_scale=float(values[2]),
        b_scale=float(values[3]),
        noise_nsig=float(values[4]),
        regional_time_log_sd=float(values[5]),
        network_gains=values[6:].copy(),
    )


def _simulate_pair_mode(
    config_path: str,
    candidate_index: int,
    candidate_values: np.ndarray,
    first_network: str,
    second_network: str,
    gain: float,
) -> dict[str, Any]:
    logging.raiseExceptions = False
    m5_config = load_m5_config(config_path)
    baseline = load_config(m5_config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    candidate = _candidate_from_values(candidate_index, candidate_values)
    endpoint_gains = {
        name: float(value)
        for name, value in zip(NETWORK_ORDER, candidate.network_gains, strict=True)
    }
    connectome = with_network_endpoint_gains(connectome, networks, endpoint_gains)
    connectome = with_network_pair_gains(
        connectome,
        networks,
        {(first_network, second_network): gain},
        preserve_total_strength=True,
    )
    run_config = build_candidate_run_config(baseline, candidate, m5_config)
    result = run_baseline(run_config, connectome)
    eeg = (
        result.eeg_surface_laplacian
        if m5_config.empirical.apply_surface_laplacian
        else result.eeg
    )
    if eeg is None:
        raise RuntimeError("Surface-Laplacian simulation output is missing")
    feature = extract_eeg_features(
        eeg,
        1000.0 / run_config.simulation.monitor_period_ms,
        m5_config.features,
    )
    return {
        "first_network": first_network,
        "second_network": second_network,
        "gain": gain,
        "psd_shape": feature.psd_shape,
        "alpha_topography": feature.alpha_topography,
        "coherence": feature.coherence,
        "alpha_peak_hz": feature.alpha_peak_hz,
        "alpha_power_fraction": feature.alpha_power_fraction,
        "spectral_entropy": feature.spectral_entropy,
    }


def _source_values(source: FeatureCollection | dict[str, np.ndarray], name: str) -> np.ndarray:
    return np.asarray(
        getattr(source, name) if isinstance(source, FeatureCollection) else source[name],
        dtype=float,
    )


def _exact_blocks(
    source: FeatureCollection | dict[str, np.ndarray],
    transformer: FeatureTransformer,
    raw_coherence_mean: np.ndarray,
    raw_coherence_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    psd = np.atleast_2d(_source_values(source, "psd_shape"))
    alpha_peak = np.atleast_1d(_source_values(source, "alpha_peak_hz"))
    alpha_fraction = np.atleast_1d(_source_values(source, "alpha_power_fraction"))
    entropy = np.atleast_1d(_source_values(source, "spectral_entropy"))
    topography = np.atleast_2d(_source_values(source, "alpha_topography"))
    coherence = _source_values(source, "coherence")
    if coherence.ndim == 2:
        coherence = coherence[np.newaxis, :, :]
    transformed = transformer.transform_blocks(
        psd, alpha_peak, alpha_fraction, entropy, topography, coherence
    )
    raw_z = (coherence - raw_coherence_mean) / raw_coherence_scale
    return {
        "PSD shape": transformed[0],
        "Spectral summary": transformed[1],
        "Alpha topography": transformed[2],
        "Coherence PCA12": transformed[3],
        "Coherence alpha": raw_z[:, 2, :],
        "Coherence beta": raw_z[:, 3, :],
        "Weighted objective": transformer.transform(
            psd, alpha_peak, alpha_fraction, entropy, topography, coherence
        ),
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.dot(first, second) / denominator)


def _plot_sensitivity(path: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    blocks = ["PSD shape", "Alpha topography", "Coherence alpha", "Coherence beta"]
    modes = list(dict.fromkeys(summary["mode"]))
    sensitivity = np.asarray(
        [
            [
                summary[(summary["mode"] == mode) & (summary["block"] == block)][
                    "standardized_sensitivity_per_unit_gain"
                ].iloc[0]
                for block in blocks
            ]
            for mode in modes
        ]
    )
    alignment = np.asarray(
        [
            [
                summary[(summary["mode"] == mode) & (summary["block"] == block)][
                    "cosine_to_empirical_mdd_direction"
                ].iloc[0]
                for block in blocks
            ]
            for mode in modes
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 12), constrained_layout=True)
    first = axes[0].imshow(sensitivity, aspect="auto", cmap="viridis")
    axes[0].set_title("Feature sensitivity to pair gain")
    axes[0].set_xticks(range(len(blocks)), blocks, rotation=28, ha="right")
    axes[0].set_yticks(range(len(modes)), modes)
    figure.colorbar(first, ax=axes[0], label="standardized RMS / unit gain")
    second = axes[1].imshow(alignment, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Direction relative to empirical MDD contrast")
    axes[1].set_xticks(range(len(blocks)), blocks, rotation=28, ha="right")
    axes[1].set_yticks(range(len(modes)), modes)
    figure.colorbar(second, ax=axes[1], label="cosine (positive gain)")
    figure.suptitle("Yeo network-pair connectome sensitivity around the M5 working point")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_connectome_sensitivity(
    config_path: Path,
    output_dir: Path,
    *,
    candidate_index: int = 99,
    deviation: float = 0.10,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """Simulate every Yeo within/between-network block at plus/minus deviation."""

    config: M5Config = load_m5_config(config_path)
    fitting = load_feature_collection(config.paths.output_dir / "empirical" / "empirical_features.npz")
    with np.load(config.paths.output_dir / "bank" / "simulation_bank.npz") as arrays:
        bank = {name: arrays[name] for name in arrays.files}
    transformer = load_feature_transformer(config.paths.output_dir / "fit" / "feature_transformer.npz")
    split = pd.read_csv(config.paths.output_dir / "fit" / "data_split.csv")
    train = np.flatnonzero(split["split"].to_numpy() == "train")
    raw_coherence_mean = fitting.coherence[train].mean(axis=0)
    raw_coherence_scale = fitting.coherence[train].std(axis=0, ddof=1)
    positive = raw_coherence_scale[raw_coherence_scale > 0]
    floor = float(np.median(positive) * 1e-3) if positive.size else 1.0
    raw_coherence_scale = np.maximum(raw_coherence_scale, floor)

    candidate_values = np.asarray(bank["parameters"][candidate_index], dtype=float)
    pairs = [
        (first, second)
        for first_index, first in enumerate(NETWORK_ORDER)
        for second in NETWORK_ORDER[first_index:]
    ]
    jobs = [
        (first, second, 1.0 + sign * deviation)
        for first, second in pairs
        for sign in (-1.0, 1.0)
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                _simulate_pair_mode,
                str(config_path),
                candidate_index,
                candidate_values,
                first,
                second,
                gain,
            ): (first, second, gain)
            for first, second, gain in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            first, second, gain = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - expensive runtime path
                failures.append(
                    {
                        "first_network": first,
                        "second_network": second,
                        "gain": gain,
                        "error": repr(exc),
                    }
                )
            if completed % max(n_jobs, 1) == 0 or completed == len(futures):
                print(f"connectome sensitivity: {completed}/{len(futures)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(output_dir / "connectome_sensitivity_failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} connectome-sensitivity simulations failed")

    empirical_blocks = _exact_blocks(
        fitting, transformer, raw_coherence_mean, raw_coherence_scale
    )
    baseline_source = {
        name: bank[name][candidate_index]
        for name in (
            "psd_shape",
            "alpha_topography",
            "coherence",
            "alpha_peak_hz",
            "alpha_power_fraction",
            "spectral_entropy",
        )
    }
    baseline_blocks = _exact_blocks(
        baseline_source, transformer, raw_coherence_mean, raw_coherence_scale
    )
    labels = (fitting.groups == "MDD").astype(int)
    contrasts = {
        block: values[train][labels[train] == 1].mean(axis=0)
        - values[train][labels[train] == 0].mean(axis=0)
        for block, values in empirical_blocks.items()
    }
    by_key = {
        (str(row["first_network"]), str(row["second_network"]), float(row["gain"])): row
        for row in results
    }
    rows: list[dict[str, float | str]] = []
    for first, second in pairs:
        minus = _exact_blocks(
            by_key[(first, second, 1.0 - deviation)],
            transformer,
            raw_coherence_mean,
            raw_coherence_scale,
        )
        plus = _exact_blocks(
            by_key[(first, second, 1.0 + deviation)],
            transformer,
            raw_coherence_mean,
            raw_coherence_scale,
        )
        for block in baseline_blocks:
            derivative = (plus[block][0] - minus[block][0]) / (2.0 * deviation)
            alignment = _cosine(derivative, contrasts[block])
            rows.append(
                {
                    "mode": f"{first}--{second}",
                    "first_network": first,
                    "second_network": second,
                    "block": block,
                    "standardized_sensitivity_per_unit_gain": float(
                        np.linalg.norm(derivative) / np.sqrt(derivative.size)
                    ),
                    "cosine_to_empirical_mdd_direction": alignment,
                    "preferred_gain_direction": "increase" if alignment >= 0 else "decrease",
                    "minus_distance_from_working_point": float(
                        np.linalg.norm(minus[block][0] - baseline_blocks[block][0])
                        / np.sqrt(derivative.size)
                    ),
                    "plus_distance_from_working_point": float(
                        np.linalg.norm(plus[block][0] - baseline_blocks[block][0])
                        / np.sqrt(derivative.size)
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "connectome_pair_sensitivity.csv", index=False)
    _plot_sensitivity(output_dir / "connectome_pair_sensitivity.png", summary)
    return summary
