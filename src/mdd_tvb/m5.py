"""M5 direct simulation-bank fitting for groups and individual subjects."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .connectome import load_connectome, with_network_endpoint_gains
from .empirical import extract_empirical_collection
from .eeg import add_colored_observation_noise, apply_surface_laplacian
from .features import (
    COHERENCE_BANDS,
    EEGFeatureVector,
    FeatureCollection,
    FeatureTransformer,
    extract_eeg_features,
    fit_feature_transformer,
    load_feature_collection,
    save_feature_transformer,
)
from .fit_config import M5Config, load_m5_config
from .heterogeneity import network_labels
from .parameterization import (
    NETWORK_ORDER,
    PARAMETER_NAMES,
    CandidateParameters,
    make_design,
    normalized_parameter_matrix,
)
from .simulation import run_baseline


@dataclass(frozen=True)
class SimulationBank:
    parameters: np.ndarray
    parameter_names: np.ndarray
    psd_shape: np.ndarray
    alpha_topography: np.ndarray
    coherence: np.ndarray
    frequency_hz: np.ndarray
    alpha_peak_hz: np.ndarray
    alpha_power_fraction: np.ndarray
    spectral_entropy: np.ndarray
    total_log_power: np.ndarray


_SIMULATION_CACHE: dict[str, tuple[Any, Any, np.ndarray]] = {}


def _simulation_inputs(baseline_config_path: str) -> tuple[Any, Any, np.ndarray]:
    if baseline_config_path not in _SIMULATION_CACHE:
        baseline = load_config(baseline_config_path)
        connectome = load_connectome(baseline.paths, baseline.connectivity)
        networks = network_labels(connectome.region_labels)
        _SIMULATION_CACHE[baseline_config_path] = (baseline, connectome, networks)
    return _SIMULATION_CACHE[baseline_config_path]


def simulate_candidate_eeg(
    baseline_config_path: str,
    candidate: CandidateParameters,
    replicate: int,
    m5_config: M5Config,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    # TVB uses one daily rotating user log. On Windows, concurrent workers can
    # race at midnight because another process holds the file open. Logging
    # already catches that non-model exception; suppress its verbose traceback.
    logging.raiseExceptions = False
    baseline, base_connectome, networks = _simulation_inputs(baseline_config_path)
    gains = {
        name: float(value)
        for name, value in zip(NETWORK_ORDER, candidate.network_gains, strict=True)
    }
    connectome = with_network_endpoint_gains(
        base_connectome, networks, gains
    )
    run_config = build_candidate_run_config(
        baseline, candidate, m5_config, replicate=replicate
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
    if eeg is None:
        raise RuntimeError("Surface-Laplacian simulation output is missing")
    return (
        eeg,
        sfreq_hz,
        result.channel_names,
    )


def build_candidate_run_config(
    baseline: Any,
    candidate: CandidateParameters,
    m5_config: M5Config,
    *,
    replicate: int = 0,
) -> Any:
    """Apply one M5 parameter vector to the shared baseline configuration."""

    design = m5_config.design
    simulation = replace(
        baseline.simulation,
        duration_ms=design.duration_ms,
        transient_ms=design.transient_ms,
        dt_ms=design.dt_ms,
        noise_nsig=candidate.noise_nsig,
        noise_tau_ms=design.noise_tau_ms,
        # Common random numbers reduce stochastic differences between candidates.
        seed=design.simulation_seed + replicate * 100003,
    )
    model = replace(
        baseline.model,
        mu=candidate.mu,
        a=baseline.model.a * candidate.a_scale,
        b=baseline.model.b * candidate.b_scale,
    )
    monitor = replace(
        baseline.monitor,
        surface_laplacian=m5_config.empirical.apply_surface_laplacian,
    )
    return replace(
        baseline,
        model=model,
        coupling=replace(
            baseline.coupling, global_gain=candidate.global_coupling
        ),
        simulation=simulation,
        heterogeneity=replace(
            baseline.heterogeneity,
            regional_time_scale_log_sd=candidate.regional_time_log_sd,
            max_parameter_deviation=design.max_parameter_deviation,
        ),
        monitor=monitor,
    )


def _simulate_candidate(
    baseline_config_path: str,
    candidate: CandidateParameters,
    replicate: int,
    m5_config: M5Config,
) -> dict[str, Any]:
    eeg, sfreq_hz, _ = simulate_candidate_eeg(
        baseline_config_path, candidate, replicate, m5_config
    )
    features: EEGFeatureVector = extract_eeg_features(
        eeg,
        sfreq_hz,
        m5_config.features,
    )
    return {
        "candidate_index": candidate.candidate_index,
        "replicate": replicate,
        "psd_shape": features.psd_shape,
        "alpha_topography": features.alpha_topography,
        "coherence": features.coherence,
        "frequency_hz": features.frequency_hz,
        "alpha_peak_hz": features.alpha_peak_hz,
        "alpha_power_fraction": features.alpha_power_fraction,
        "spectral_entropy": features.spectral_entropy,
        "total_log_power": features.total_log_power,
    }


def _save_bank(path: Path, bank: SimulationBank) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.__dict__)


def load_simulation_bank(path: Path) -> SimulationBank:
    with np.load(path) as arrays:
        return SimulationBank(
            **{name: arrays[name] for name in SimulationBank.__dataclass_fields__}
        )


def build_simulation_bank(
    config: M5Config,
    design_samples: int | None = None,
) -> SimulationBank:
    design_settings = (
        config.design
        if design_samples is None
        else replace(config.design, samples=design_samples)
    )
    working_config = replace(config, design=design_settings)
    candidates = make_design(design_settings)
    tasks = [
        (candidate, replicate)
        for candidate in candidates
        for replicate in range(design_settings.replicates)
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str | int]] = []
    if design_settings.n_jobs == 1:
        for index, (candidate, replicate) in enumerate(tasks, start=1):
            try:
                rows.append(_simulate_candidate(
                    str(config.paths.baseline_config), candidate, replicate, working_config
                ))
            except Exception as error:
                failures.append({
                    "candidate_index": candidate.candidate_index,
                    "replicate": replicate,
                    "error": repr(error),
                })
            print(f"Simulation bank: {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=design_settings.n_jobs) as pool:
            futures = {
                pool.submit(
                    _simulate_candidate,
                    str(config.paths.baseline_config),
                    candidate,
                    replicate,
                    working_config,
                ): (candidate.candidate_index, replicate)
                for candidate, replicate in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                candidate_index, replicate = futures[future]
                try:
                    rows.append(future.result())
                except Exception as error:
                    failures.append({
                        "candidate_index": candidate_index,
                        "replicate": replicate,
                        "error": repr(error),
                    })
                if index % design_settings.n_jobs == 0 or index == len(futures):
                    print(f"Simulation bank: {index}/{len(futures)}", flush=True)

    bank_dir = config.paths.output_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["candidate_index", "replicate", "error"]
    ).to_csv(bank_dir / "simulation_failures.csv", index=False)
    if failures:
        raise RuntimeError(
            f"Simulation failed for {len(failures)} design evaluations; see "
            f"{bank_dir / 'simulation_failures.csv'}"
        )
    rows.sort(key=lambda row: (row["candidate_index"], row["replicate"]))
    frequency = rows[0]["frequency_hz"]
    aggregated: list[dict[str, Any]] = []
    for candidate in candidates:
        selected = [
            row for row in rows if row["candidate_index"] == candidate.candidate_index
        ]
        if len(selected) != design_settings.replicates:
            raise RuntimeError("Simulation bank replicate count is incomplete")
        aggregated.append({
            "psd_shape": np.mean([row["psd_shape"] for row in selected], axis=0),
            "alpha_topography": np.mean(
                [row["alpha_topography"] for row in selected], axis=0
            ),
            "coherence": np.mean([row["coherence"] for row in selected], axis=0),
            "alpha_peak_hz": np.mean([row["alpha_peak_hz"] for row in selected]),
            "alpha_power_fraction": np.mean(
                [row["alpha_power_fraction"] for row in selected]
            ),
            "spectral_entropy": np.mean(
                [row["spectral_entropy"] for row in selected]
            ),
            "total_log_power": np.mean(
                [row["total_log_power"] for row in selected]
            ),
        })
    bank = SimulationBank(
        parameters=np.stack([candidate.numeric_vector() for candidate in candidates]),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="U48"),
        psd_shape=np.stack([row["psd_shape"] for row in aggregated]),
        alpha_topography=np.stack(
            [row["alpha_topography"] for row in aggregated]
        ),
        coherence=np.stack([row["coherence"] for row in aggregated]),
        frequency_hz=frequency,
        alpha_peak_hz=np.asarray([row["alpha_peak_hz"] for row in aggregated]),
        alpha_power_fraction=np.asarray(
            [row["alpha_power_fraction"] for row in aggregated]
        ),
        spectral_entropy=np.asarray(
            [row["spectral_entropy"] for row in aggregated]
        ),
        total_log_power=np.asarray(
            [row["total_log_power"] for row in aggregated]
        ),
    )
    _save_bank(bank_dir / "simulation_bank.npz", bank)
    parameter_table = pd.DataFrame(bank.parameters, columns=bank.parameter_names)
    parameter_table.insert(0, "candidate_index", np.arange(len(candidates)))
    parameter_table.to_csv(bank_dir / "candidate_parameters.csv", index=False)
    return bank


def _stratified_split(
    groups: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    holdout: list[int] = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * holdout_fraction)))
        holdout.extend(indices[:count])
        train.extend(indices[count:])
    return np.asarray(sorted(train)), np.asarray(sorted(holdout))


def _feature_matrix(
    transformer: FeatureTransformer,
    source: FeatureCollection | SimulationBank,
) -> np.ndarray:
    return transformer.transform(
        source.psd_shape,
        source.alpha_peak_hz,
        source.alpha_power_fraction,
        source.spectral_entropy,
        source.alpha_topography,
        source.coherence,
    )


def _parameter_record(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(value) for name, value in zip(PARAMETER_NAMES, values, strict=True)
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=float).ravel()
    second_values = np.asarray(second, dtype=float).ravel()
    if np.std(first_values) == 0 or np.std(second_values) == 0:
        return 0.0
    return float(np.corrcoef(first_values, second_values)[0, 1])


def _synthetic_recovery(
    bank_features: np.ndarray,
    normalized_parameters: np.ndarray,
) -> dict[str, Any]:
    feature_distance = np.sum(
        (bank_features[:, np.newaxis, :] - bank_features[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    np.fill_diagonal(feature_distance, np.inf)
    recovered = np.argmin(feature_distance, axis=1)
    error = normalized_parameters[recovered] - normalized_parameters
    return {
        "method": "leave-one-design-point-out nearest simulated feature vector",
        "median_normalized_parameter_rmse": float(
            np.median(np.sqrt(np.mean(error**2, axis=1)))
        ),
        "p90_normalized_parameter_rmse": float(
            np.percentile(np.sqrt(np.mean(error**2, axis=1)), 90)
        ),
        "per_parameter_mean_absolute_error": {
            name: float(value)
            for name, value in zip(
                PARAMETER_NAMES, np.mean(np.abs(error), axis=0), strict=True
            )
        },
    }


def _plot_fit_summary(
    output_path: Path,
    collection: FeatureCollection,
    validation_collection: FeatureCollection,
    bank: SimulationBank,
    train_indices: np.ndarray,
    group_rows: list[dict[str, Any]],
    subject_table: pd.DataFrame,
) -> None:
    """Plot group averages of independently fitted subject simulations."""

    groups = list(dict.fromkeys(collection.groups.tolist()))
    colors = {group: f"C{index}" for index, group in enumerate(groups)}
    ordered_subjects = subject_table.set_index("subject_id").loc[
        collection.subject_ids.astype(str)
    ]
    selected_candidates = ordered_subjects["candidate_index"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    channels = np.arange(collection.alpha_topography.shape[1])
    for group in groups:
        subject_indices = np.flatnonzero(collection.groups == group)
        candidates = selected_candidates[subject_indices]
        unique_count = np.unique(candidates).size
        fit_empirical_psd = collection.psd_shape[subject_indices].mean(axis=0)
        unseen_empirical_psd = validation_collection.psd_shape[
            subject_indices
        ].mean(axis=0)
        fitted_psd = bank.psd_shape[candidates].mean(axis=0)
        fit_empirical_topography = collection.alpha_topography[
            subject_indices
        ].mean(axis=0)
        unseen_empirical_topography = validation_collection.alpha_topography[
            subject_indices
        ].mean(axis=0)
        fitted_topography = bank.alpha_topography[candidates].mean(axis=0)

        for axis, empirical_psd, half_name in (
            (axes[0, 0], fit_empirical_psd, "fitting half"),
            (axes[0, 1], unseen_empirical_psd, "unseen half"),
        ):
            axis.plot(
                collection.frequency_hz,
                np.exp(empirical_psd),
                color=colors[group],
                linewidth=2,
                label=f"{group} empirical {half_name}",
            )
            axis.plot(
                collection.frequency_hz,
                np.exp(fitted_psd),
                color=colors[group],
                linestyle="--",
                linewidth=1.7,
                label=(
                    f"{group} mean of subject fits "
                    f"({unique_count} candidates)"
                ),
            )
        for axis, empirical_topography, half_name in (
            (axes[1, 0], fit_empirical_topography, "fitting half"),
            (axes[1, 1], unseen_empirical_topography, "unseen half"),
        ):
            axis.plot(
                channels,
                empirical_topography,
                color=colors[group],
                linewidth=2,
                label=f"{group} empirical {half_name}",
            )
            axis.plot(
                channels,
                fitted_topography,
                color=colors[group],
                linestyle="--",
                linewidth=1.7,
                label=f"{group} mean of subject fits",
            )

    for axis, title in (
        (axes[0, 0], "PSD: fitting half vs independent subject fits"),
        (axes[0, 1], "PSD: unseen half vs same subject fits"),
    ):
        axis.set_title(title)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Relative PSD shape")
        axis.set_yscale("log")
        axis.legend(fontsize=8)
    for axis, title in (
        (axes[1, 0], "Alpha topography: fitting half vs subject fits"),
        (axes[1, 1], "Alpha topography: unseen half vs same subject fits"),
    ):
        axis.set_title(title)
        axis.set_xlabel("TDBRAIN channel index")
        axis.set_ylabel("Centered log alpha power")
        axis.legend(fontsize=8)
    fig.suptitle(
        "M5 group summaries of independently fitted subject models\n"
        "Dashed curves are averages of subject-selected simulations, not group MAP fits",
        fontsize=14,
    )
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary, dpi=160)
    subject_summary_path = output_path.with_name("m5_subject_fit_summary.png")
    fig.savefig(subject_summary_path, dpi=160)
    plt.close(fig)
    temporary.replace(output_path)

    _plot_individual_validation(
        output_path.with_name("m5_individual_validation.png"),
        subject_table,
        groups,
        colors,
    )
    _plot_subject_parameter_distributions(
        output_path.with_name("m5_subject_parameter_distributions.png"),
        subject_table,
        bank,
        groups,
        colors,
    )


def _plot_individual_validation(
    output_path: Path,
    subject_table: pd.DataFrame,
    groups: list[str],
    colors: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for group in groups:
        selected = subject_table[subject_table["group"] == group]
        axes[0].scatter(
            selected["fit_cost_ratio"], selected["validation_cost_ratio"],
            color=colors[group], alpha=0.55, s=18, label=group,
        )
    limits = [
        0.0,
        float(
            max(
                subject_table["fit_cost_ratio"].max(),
                subject_table["validation_cost_ratio"].max(),
                1.05,
            )
        ),
    ]
    axes[0].plot(limits, limits, color="black", linestyle=":", label="equal ratio")
    axes[0].axhline(1.0, color="red", linewidth=1, linestyle="--")
    axes[0].axvline(1.0, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_title("Fit-half vs unseen-half cost ratio")
    axes[0].set_xlabel("Fit-half fitted / reference cost")
    axes[0].set_ylabel("Unseen-half fitted / reference cost")
    axes[0].legend()

    for group in groups:
        selected = subject_table[subject_table["group"] == group]
        axes[1].scatter(
            selected["validation_alpha_peak_hz"],
            selected["fitted_alpha_peak_hz"],
            color=colors[group], alpha=0.55, s=18, label=group,
        )
    iaf_limits = [
        min(subject_table["validation_alpha_peak_hz"].min(),
            subject_table["fitted_alpha_peak_hz"].min()),
        max(subject_table["validation_alpha_peak_hz"].max(),
            subject_table["fitted_alpha_peak_hz"].max()),
    ]
    axes[1].plot(iaf_limits, iaf_limits, color="black", linestyle=":")
    axes[1].set_title("Selected-model vs unseen-half IAF")
    axes[1].set_xlabel("Unseen empirical IAF (Hz)")
    axes[1].set_ylabel("Selected simulation IAF (Hz)")
    axes[1].legend()

    width = 0.8 / len(groups)
    metric_names = ["PSD", "Alpha topo.", "Coherence"]
    positions = np.arange(len(metric_names))
    for group_index, group in enumerate(groups):
        selected = subject_table[subject_table["group"] == group]
        values = [
            selected["validation_psd_correlation"].median(),
            selected["validation_alpha_topography_correlation"].median(),
            selected["validation_coherence_correlation"].median(),
        ]
        axes[2].bar(
            positions + group_index * width, values, width, label=group
        )
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_xticks(positions + width * (len(groups) - 1) / 2)
    axes[2].set_xticklabels(metric_names)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].set_ylabel("Median correlation on unseen half")
    axes[2].set_title("Feature-specific validation agreement")
    axes[2].legend()
    fig.suptitle("M5 individual-fit validation", fontsize=14)
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(output_path)


def _plot_subject_parameter_distributions(
    output_path: Path,
    subject_table: pd.DataFrame,
    bank: SimulationBank,
    groups: list[str],
    colors: dict[str, str],
) -> None:
    selected = subject_table["candidate_index"].to_numpy(dtype=int)
    parameter_values = bank.parameters[selected]
    dynamic_lower = bank.parameters[:, :6].min(axis=0)
    dynamic_upper = bank.parameters[:, :6].max(axis=0)
    dynamic_values = 2.0 * (
        (parameter_values[:, :6] - dynamic_lower)
        / np.maximum(dynamic_upper - dynamic_lower, np.finfo(float).eps)
    ) - 1.0
    connectivity_values = parameter_values[:, 6:] - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    panels = (
        (
            axes[0], dynamic_values, PARAMETER_NAMES[:6],
            "Subject dynamic parameters (normalized to bank range)",
        ),
        (
            axes[1], connectivity_values,
            tuple(name.removeprefix("network_gain_") for name in PARAMETER_NAMES[6:]),
            "Subject effective-connectivity gains",
        ),
    )
    for axis, panel_values, labels, title in panels:
        base_positions = np.arange(panel_values.shape[1], dtype=float)
        for group_index, group in enumerate(groups):
            group_mask = subject_table["group"].to_numpy() == group
            values = panel_values[group_mask]
            positions = base_positions + (group_index - 0.5) * 0.28
            boxes = axis.boxplot(
                [values[:, column] for column in range(values.shape[1])],
                positions=positions,
                widths=0.24,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for box in boxes["boxes"]:
                box.set_facecolor(colors[group])
                box.set_alpha(0.55)
            axis.plot([], [], color=colors[group], linewidth=8, alpha=0.55, label=group)
        axis.axhline(0.0, color="black", linestyle=":")
        axis.set_xticks(base_positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.legend()
    axes[0].set_ylabel("Normalized bank coordinate")
    axes[1].set_ylabel("Endpoint gain minus one")
    fig.suptitle(
        "Distributions of independently selected subject parameters\n"
        "These are discrete-bank estimates; overlap is expected",
        fontsize=14,
    )
    temporary = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(output_path)


def fit_groups_and_subjects(
    config: M5Config,
    collection: FeatureCollection | None = None,
    validation_collection: FeatureCollection | None = None,
    bank: SimulationBank | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    empirical_path = config.paths.output_dir / "empirical" / "empirical_features.npz"
    validation_path = (
        config.paths.output_dir
        / "empirical"
        / "empirical_validation_features.npz"
    )
    bank_path = config.paths.output_dir / "bank" / "simulation_bank.npz"
    collection = collection or load_feature_collection(empirical_path)
    validation_collection = validation_collection or load_feature_collection(
        validation_path
    )
    bank = bank or load_simulation_bank(bank_path)
    if not np.array_equal(collection.frequency_hz, bank.frequency_hz):
        raise ValueError("Empirical and simulated feature frequency grids differ")
    if not np.array_equal(collection.subject_ids, validation_collection.subject_ids):
        raise ValueError("Fit and validation subject orders differ")
    train_indices, holdout_indices = _stratified_split(
        collection.groups,
        config.features.holdout_fraction,
        config.features.split_seed,
    )
    transformer = fit_feature_transformer(
        collection,
        train_indices,
        config.features,
        validation_collection=validation_collection,
    )
    fit_dir = config.paths.output_dir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    save_feature_transformer(fit_dir / "feature_transformer.npz", transformer)
    empirical_features = _feature_matrix(transformer, collection)
    validation_features = _feature_matrix(transformer, validation_collection)
    bank_features = _feature_matrix(transformer, bank)
    normalized_parameters = normalized_parameter_matrix(
        bank.parameters, config.design
    )
    baseline_parameter = normalized_parameters[0]

    pooled_target = empirical_features[train_indices].mean(axis=0)
    pooled_data_cost = np.sum((bank_features - pooled_target) ** 2, axis=1)
    pooled_prior_cost = config.fit.group_prior_strength * np.mean(
        (normalized_parameters - baseline_parameter) ** 2, axis=1
    )
    pooled_candidate = int(np.argmin(pooled_data_cost + pooled_prior_cost))
    pd.DataFrame(
        [
            {
                "candidate_index": pooled_candidate,
                "training_data_cost": float(pooled_data_cost[pooled_candidate]),
                "training_reference_cost": float(pooled_data_cost[0]),
                **_parameter_record(bank.parameters[pooled_candidate]),
            }
        ]
    ).to_csv(fit_dir / "pooled_label_blind_fit.csv", index=False)

    group_rows: list[dict[str, Any]] = []
    group_candidate: dict[str, int] = {}
    for group in config.empirical.groups:
        group_train = train_indices[collection.groups[train_indices] == group]
        group_holdout = holdout_indices[collection.groups[holdout_indices] == group]
        target = empirical_features[group_train].mean(axis=0)
        data_cost = np.sum((bank_features - target) ** 2, axis=1)
        prior_cost = config.fit.group_prior_strength * np.mean(
            (normalized_parameters - baseline_parameter) ** 2, axis=1
        )
        selected = int(np.argmin(data_cost + prior_cost))
        group_candidate[group] = selected
        holdout_target = validation_features[group_holdout].mean(axis=0)
        holdout_data_cost = float(
            np.sum((bank_features[selected] - holdout_target) ** 2)
        )
        holdout_reference_cost = float(
            np.sum((bank_features[0] - holdout_target) ** 2)
        )
        training_cost = float(data_cost[selected])
        training_reference_cost = float(data_cost[0])
        group_train_psd = collection.psd_shape[group_train].mean(axis=0)
        group_holdout_psd = validation_collection.psd_shape[
            group_holdout
        ].mean(axis=0)
        group_train_topography = collection.alpha_topography[
            group_train
        ].mean(axis=0)
        group_holdout_topography = validation_collection.alpha_topography[
            group_holdout
        ].mean(axis=0)
        record: dict[str, Any] = {
            "group": group,
            "n_train": len(group_train),
            "n_holdout": len(group_holdout),
            "candidate_index": selected,
            "training_data_cost": training_cost,
            "training_reference_cost": training_reference_cost,
            "training_weighted_standardized_rmse": float(np.sqrt(training_cost)),
            "training_reference_weighted_standardized_rmse": float(
                np.sqrt(training_reference_cost)
            ),
            "training_cost_ratio": (
                training_cost / training_reference_cost
                if training_reference_cost > 0 else np.nan
            ),
            "holdout_data_cost": holdout_data_cost,
            "holdout_reference_cost": holdout_reference_cost,
            "holdout_weighted_standardized_rmse": float(
                np.sqrt(holdout_data_cost)
            ),
            "holdout_reference_weighted_standardized_rmse": float(
                np.sqrt(holdout_reference_cost)
            ),
            "holdout_cost_ratio": (
                holdout_data_cost / holdout_reference_cost
                if holdout_reference_cost > 0 else np.nan
            ),
            "training_psd_correlation": _correlation(
                group_train_psd, bank.psd_shape[selected]
            ),
            "holdout_psd_correlation": _correlation(
                group_holdout_psd, bank.psd_shape[selected]
            ),
            "training_alpha_topography_correlation": _correlation(
                group_train_topography, bank.alpha_topography[selected]
            ),
            "holdout_alpha_topography_correlation": _correlation(
                group_holdout_topography, bank.alpha_topography[selected]
            ),
            "fitted_alpha_peak_hz": float(bank.alpha_peak_hz[selected]),
        }
        record.update(_parameter_record(bank.parameters[selected]))
        group_rows.append(record)
    group_table = pd.DataFrame(group_rows)
    group_table.to_csv(fit_dir / "group_fits.csv", index=False)

    subject_rows: list[dict[str, Any]] = []
    for subject_index in range(len(collection.subject_ids)):
        group = str(collection.groups[subject_index])
        data_cost = np.sum(
            (bank_features - empirical_features[subject_index]) ** 2, axis=1
        )
        if config.fit.subject_prior_center == "pooled":
            prior_candidate = pooled_candidate
        elif config.fit.subject_prior_center == "reference":
            prior_candidate = 0
        else:
            prior_candidate = group_candidate[group]
        center = normalized_parameters[prior_candidate]
        prior_cost = config.fit.subject_prior_strength * np.mean(
            (normalized_parameters - center) ** 2, axis=1
        )
        selected = int(np.argmin(data_cost + prior_cost))
        baseline_cost = float(data_cost[0])
        selected_cost = float(data_cost[selected])
        validation_cost = np.sum(
            (bank_features - validation_features[subject_index]) ** 2, axis=1
        )
        reference_validation_cost = float(validation_cost[0])
        selected_validation_cost = float(validation_cost[selected])
        fit_cost_ratio = (
            selected_cost / baseline_cost if baseline_cost > 0 else np.nan
        )
        validation_cost_ratio = (
            selected_validation_cost / reference_validation_cost
            if reference_validation_cost > 0 else np.nan
        )
        record = {
            "subject_id": str(collection.subject_ids[subject_index]),
            "group": group,
            "split": "holdout" if subject_index in set(holdout_indices) else "train",
            "candidate_index": selected,
            "group_candidate_index": group_candidate[group],
            "subject_prior_candidate_index": prior_candidate,
            "subject_prior_center": config.fit.subject_prior_center,
            "data_fit_cost": selected_cost,
            "reference_data_cost": baseline_cost,
            "fit_weighted_standardized_rmse": float(np.sqrt(selected_cost)),
            "reference_fit_weighted_standardized_rmse": float(
                np.sqrt(baseline_cost)
            ),
            "data_fit_improvement_fraction": (
                (baseline_cost - selected_cost) / baseline_cost
                if baseline_cost > 0 else 0.0
            ),
            "fit_cost_ratio": fit_cost_ratio,
            "validation_data_cost": selected_validation_cost,
            "reference_validation_cost": reference_validation_cost,
            "validation_weighted_standardized_rmse": float(
                np.sqrt(selected_validation_cost)
            ),
            "reference_validation_weighted_standardized_rmse": float(
                np.sqrt(reference_validation_cost)
            ),
            "validation_improvement_fraction": (
                (reference_validation_cost - selected_validation_cost)
                / reference_validation_cost
                if reference_validation_cost > 0 else 0.0
            ),
            "validation_cost_ratio": validation_cost_ratio,
            "generalization_gap_cost_ratio": (
                validation_cost_ratio - fit_cost_ratio
            ),
            "fit_psd_correlation": _correlation(
                collection.psd_shape[subject_index], bank.psd_shape[selected]
            ),
            "validation_psd_correlation": _correlation(
                validation_collection.psd_shape[subject_index],
                bank.psd_shape[selected],
            ),
            "fit_alpha_topography_correlation": _correlation(
                collection.alpha_topography[subject_index],
                bank.alpha_topography[selected],
            ),
            "validation_alpha_topography_correlation": _correlation(
                validation_collection.alpha_topography[subject_index],
                bank.alpha_topography[selected],
            ),
            "fit_coherence_correlation": _correlation(
                collection.coherence[subject_index], bank.coherence[selected]
            ),
            "validation_coherence_correlation": _correlation(
                validation_collection.coherence[subject_index],
                bank.coherence[selected],
            ),
            "empirical_alpha_peak_hz": float(collection.alpha_peak_hz[subject_index]),
            "validation_alpha_peak_hz": float(
                validation_collection.alpha_peak_hz[subject_index]
            ),
            "fitted_alpha_peak_hz": float(bank.alpha_peak_hz[selected]),
            "fit_alpha_peak_absolute_error_hz": float(
                abs(collection.alpha_peak_hz[subject_index] - bank.alpha_peak_hz[selected])
            ),
            "validation_alpha_peak_absolute_error_hz": float(
                abs(
                    validation_collection.alpha_peak_hz[subject_index]
                    - bank.alpha_peak_hz[selected]
                )
            ),
            "observation_log_amplitude_offset_nuisance": float(
                0.5
                * (
                    collection.total_log_power[subject_index]
                    - bank.total_log_power[selected]
                )
            ),
        }
        record.update(_parameter_record(bank.parameters[selected]))
        subject_rows.append(record)
    subject_table = pd.DataFrame(subject_rows)
    subject_table.to_csv(fit_dir / "subject_fits.csv", index=False)
    parameter_summary_rows: list[dict[str, Any]] = []
    fitted_parameter_names = (
        PARAMETER_NAMES
        if config.design.fit_network_endpoint_gains
        else PARAMETER_NAMES[:6]
    )
    for group in config.empirical.groups:
        selected_group = subject_table[subject_table["group"] == group]
        for parameter in fitted_parameter_names:
            values = selected_group[parameter].to_numpy(dtype=float)
            parameter_summary_rows.append({
                "group": group,
                "parameter": parameter,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            })
    pd.DataFrame(parameter_summary_rows).to_csv(
        fit_dir / "group_summary_of_subject_parameters.csv", index=False
    )

    baseline = load_config(config.paths.baseline_config)
    base_connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(base_connectome.region_labels)
    for row in group_rows:
        gains = {
            name: row[f"network_gain_{name}"] for name in NETWORK_ORDER
        }
        group_connectome = with_network_endpoint_gains(
            base_connectome, networks, gains
        )
        np.save(
            fit_dir / f"group_connectome_{row['group']}.npy",
            group_connectome.weights,
        )

    recovery = _synthetic_recovery(bank_features, normalized_parameters)
    subject_validation: dict[str, Any] = {}
    for group in config.empirical.groups:
        group_subjects = subject_table[subject_table["group"] == group]
        split_records: dict[str, Any] = {}
        for split_name in ("train", "holdout"):
            selected_subjects = group_subjects[
                group_subjects["split"] == split_name
            ]
            improvement = selected_subjects[
                "validation_improvement_fraction"
            ].to_numpy()
            validation_ratio = selected_subjects[
                "validation_cost_ratio"
            ].to_numpy()
            split_records[split_name] = {
                "n": int(len(selected_subjects)),
                "median_validation_improvement_fraction": float(
                    np.median(improvement)
                ),
                "mean_validation_improvement_fraction": float(
                    np.mean(improvement)
                ),
                "positive_validation_fraction": float(
                    np.mean(improvement > 0)
                ),
                "median_validation_cost_ratio": float(
                    np.median(validation_ratio)
                ),
                "median_validation_psd_correlation": float(
                    selected_subjects["validation_psd_correlation"].median()
                ),
                "median_validation_alpha_topography_correlation": float(
                    selected_subjects[
                        "validation_alpha_topography_correlation"
                    ].median()
                ),
                "median_validation_coherence_correlation": float(
                    selected_subjects[
                        "validation_coherence_correlation"
                    ].median()
                ),
            }
        subject_validation[group] = {
            "unique_selected_candidates": int(
                group_subjects["candidate_index"].nunique()
            ),
            "group_candidate_selection_fraction": float(
                np.mean(
                    group_subjects["candidate_index"]
                    == group_candidate[group]
                )
            ),
            "by_subject_split": split_records,
        }
    empirical_entropy_percentiles = np.percentile(
        collection.spectral_entropy, [5, 50, 95]
    )
    empirical_iaf_percentiles = np.percentile(
        collection.alpha_peak_hz, [5, 50, 95]
    )
    summary = {
        "method": (
            "direct simulation-bank MAP selection; individual fits use the declared "
            f"{config.fit.subject_prior_center} prior centre"
        ),
        "parameterization": {
            "dynamic_parameters": list(PARAMETER_NAMES[:6]),
            "connectivity_parameters": (
                list(PARAMETER_NAMES[6:])
                if config.design.fit_network_endpoint_gains
                else []
            ),
            "fixed_connectivity_parameters": (
                []
                if config.design.fit_network_endpoint_gains
                else list(PARAMETER_NAMES[6:])
            ),
            "network_gain_bounds": [
                (
                    1.0 - config.design.network_weight_deviation
                    if config.design.fit_network_endpoint_gains
                    else 1.0
                ),
                (
                    1.0 + config.design.network_weight_deviation
                    if config.design.fit_network_endpoint_gains
                    else 1.0
                ),
            ],
            "connectivity_constraint": (
                (
                    "seven endpoint gains with mean one; symmetric edge multiplier "
                    "is the geometric mean of endpoint gains"
                )
                if config.design.fit_network_endpoint_gains
                else "fixed to the shared baseline connectome after identifiability audit"
            ),
        },
        "objective": {
            "absolute_amplitude_used": False,
            "dc_used": False,
            "frequency_range_hz": [
                config.features.frequency_min_hz,
                config.features.frequency_max_hz,
            ],
            "feature_blocks": {
                "normalized_psd_shape": config.features.psd_weight,
                "spectral_summary": config.features.spectral_summary_weight,
                "centered_alpha_topography": config.features.alpha_topography_weight,
                f"coherence_{config.features.coherence_transform}": (
                    config.features.coherence_weight
                ),
            },
            "coherence_bands": (
                list(config.features.coherence_fit_bands)
                if config.features.coherence_transform == "reliable_edges"
                else [name for name, _, _ in COHERENCE_BANDS]
            ),
        },
        "evaluation": {
            "weighted_standardized_rmse": (
                "square root of the weighted mean squared z-feature mismatch; "
                "lower is better and zero is exact feature agreement"
            ),
            "fit_cost_ratio": (
                "selected-candidate feature cost on the fitting half divided "
                "by reference-candidate cost on that same half"
            ),
            "validation_cost_ratio": (
                "selected-candidate feature cost on the unseen half divided "
                "by reference-candidate cost on that same unseen half"
            ),
            "interpretation": (
                "ratios below one beat the declared reference; absolute ratios "
                "and feature-specific correlations are reported alongside them"
            ),
        },
        "split": {
            "train_subjects": int(len(train_indices)),
            "holdout_subjects": int(len(holdout_indices)),
            "seed": config.features.split_seed,
            "within_subject_validation": (
                "fit on first temporal half; evaluated on second temporal half"
            ),
        },
        "coverage": {
            "bank_alpha_peak_hz_range": [
                float(bank.alpha_peak_hz.min()),
                float(bank.alpha_peak_hz.max()),
            ],
            "empirical_alpha_peak_hz_p5_median_p95": (
                empirical_iaf_percentiles.tolist()
            ),
            "bank_spectral_entropy_range": [
                float(bank.spectral_entropy.min()),
                float(bank.spectral_entropy.max()),
            ],
            "empirical_spectral_entropy_p5_median_p95": (
                empirical_entropy_percentiles.tolist()
            ),
        },
        "group_results": group_rows,
        "subject_validation": subject_validation,
        "simulation_bank": {
            "candidates": int(len(bank.parameters)),
            "replicates": config.design.replicates,
            "duration_ms": config.design.duration_ms,
            "dt_ms": config.design.dt_ms,
            "common_random_numbers": True,
        },
        "synthetic_recovery": recovery,
        "limitations": [
            "Every reported fit selects a directly simulated bank member; parameter resolution is limited by bank coverage.",
            "The common structural connectome is modulated only by strongly bounded network-level gains, not subject tractography.",
            "The primary subject prior is label-blind; diagnosis labels are used only for post-fit group summaries and descriptive group-centroid fits.",
            "One stochastic replicate per design point is a pilot setting and should be increased before inference.",
            "A common regional heterogeneity realization is used across candidates; its dispersion, not its spatial draw, is fitted.",
            "Absolute EEG amplitude is treated as a nuisance observation gain and is not used to identify neural parameters.",
            "The upper empirical IAF and high-entropy tails extend beyond this finite simulation bank.",
            "Synthetic recovery is weak for individual network gains; treat those gains as regularized effective hypotheses, not anatomical estimates.",
        ],
        "config": asdict(config),
    }

    def json_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    (fit_dir / "fit_summary.json").write_text(
        json.dumps(json_value(summary), indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "subject_id": collection.subject_ids,
        "group": collection.groups,
        "split": [
            "holdout" if index in set(holdout_indices) else "train"
            for index in range(len(collection.subject_ids))
        ],
    }).to_csv(fit_dir / "data_split.csv", index=False)
    _plot_fit_summary(
        fit_dir / "m5_fit_summary.png",
        collection,
        validation_collection,
        bank,
        train_indices,
        group_rows,
        subject_table,
    )
    from .m5_reporting import create_eeg_example_figure

    create_eeg_example_figure(config)
    return group_table, subject_table


def run_m5(
    config_path: str | Path,
    stages: tuple[str, ...] = ("extract", "bank", "fit"),
    max_subjects_per_group: int | None = None,
    design_samples: int | None = None,
) -> None:
    config = load_m5_config(config_path)
    collection: FeatureCollection | None = None
    bank: SimulationBank | None = None
    if "extract" in stages:
        collection = extract_empirical_collection(config, max_subjects_per_group)
    if "bank" in stages:
        bank = build_simulation_bank(config, design_samples)
    if "fit" in stages:
        fit_groups_and_subjects(config, collection, None, bank)
