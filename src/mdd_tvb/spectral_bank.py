"""Multi-seed TVB simulation bank for cross-spectral M5."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .connectome import load_connectome, with_network_pair_gains
from .heterogeneity import network_labels
from .multifrequency import run_dual_jansen_rit
from .spectral_config import SpectralM5Config
from .spectral_features import estimate_cross_spectrum
from .spectral_parameterization import (
    PARAMETER_NAMES,
    SpectralCandidate,
    make_spectral_design,
)


@dataclass(frozen=True)
class SpectralSimulationBank:
    parameters: np.ndarray
    parameter_names: np.ndarray
    frequency_hz: np.ndarray
    channel_names: np.ndarray
    csd_replicates: np.ndarray

    @property
    def csd(self) -> np.ndarray:
        return np.mean(self.csd_replicates, axis=1)


_CACHE: dict[str, tuple[Any, Any, np.ndarray]] = {}


def _inputs(path: str) -> tuple[Any, Any, np.ndarray]:
    if path not in _CACHE:
        baseline = load_config(path)
        connectome = load_connectome(baseline.paths, baseline.connectivity)
        _CACHE[path] = (
            baseline,
            connectome,
            network_labels(connectome.region_labels),
        )
    return _CACHE[path]


def _network_contrast_map(
    networks: np.ndarray, contrasts: dict[str, float]
) -> dict[str, float]:
    """Return positive, parcel-mean-one network multipliers."""

    labels = networks.astype(str)
    names = np.unique(labels)
    multipliers = np.ones(len(labels), dtype=float)
    for name, contrast in contrasts.items():
        if name not in names:
            raise ValueError(f"Unknown network contrast target: {name}")
        if 1.0 + contrast <= 0.0:
            raise ValueError(f"Network contrast for {name} is non-positive")
        multipliers[labels == name] *= 1.0 + contrast
    multipliers /= multipliers.mean()
    return {
        str(name): float(multipliers[labels == name][0])
        for name in names
    }


def connectome_for_spectral_candidate(
    connectome: Any,
    networks: np.ndarray,
    candidate: SpectralCandidate,
) -> Any:
    """Apply two broad, bounded and total-strength-preserving weight modes.

    The previous two modes touched only the Default--DorsAttn and
    Default--SalVentAttn blocks.  Their effect was below stochastic replicate
    noise in the better-sampled calibration bank.  Each mode retains the
    requested +/-10% coefficient bound while acting on enough of the
    connectome to be potentially identifiable from sensor spectra:

    * all edges incident on Default mode network;
    * a DorsAttn-versus-SalVentAttn incident-edge balance.
    """

    names = sorted(str(name) for name in np.unique(networks))
    gains: dict[tuple[str, str], float] = {}
    for first_index, first in enumerate(names):
        for second in names[first_index:]:
            default_loading = float("Default" in {first, second})
            has_dorsal = "DorsAttn" in {first, second}
            has_salience = "SalVentAttn" in {first, second}
            balance_loading = (
                1.0 if has_dorsal and not has_salience
                else -1.0 if has_salience and not has_dorsal
                else 0.0
            )
            gain = (
                1.0
                + candidate.default_incident_weight_contrast * default_loading
            ) * (
                1.0
                + candidate.dorsattn_salventattn_weight_balance
                * balance_loading
            )
            gains[(first, second)] = gain
    return with_network_pair_gains(
        connectome,
        networks,
        gains,
        preserve_total_strength=True,
    )


def build_spectral_run_config(
    baseline: Any,
    networks: np.ndarray,
    candidate: SpectralCandidate,
    config: SpectralM5Config,
    replicate: int,
) -> Any:
    design = config.design
    noise_contrasts = {
        "Default": candidate.default_noise_contrast,
        "Vis": candidate.visual_noise_contrast,
    }
    for coefficient, mode in (
        (candidate.network_noise_mode_1, design.network_noise_mode_1),
        (candidate.network_noise_mode_2, design.network_noise_mode_2),
    ):
        for network, loading in mode:
            noise_contrasts[network] = (
                noise_contrasts.get(network, 0.0) + coefficient * loading
            )
    return replace(
        baseline,
        model=replace(
            baseline.model,
            mu=candidate.mu,
            a=baseline.model.a * candidate.a_scale,
            b=baseline.model.b * candidate.b_scale,
        ),
        coupling=replace(
            baseline.coupling, global_gain=candidate.global_coupling
        ),
        simulation=replace(
            baseline.simulation,
            duration_ms=design.duration_ms,
            transient_ms=design.transient_ms,
            dt_ms=design.dt_ms,
            noise_nsig=candidate.noise_nsig,
            noise_tau_ms=candidate.noise_tau_ms,
            seed=design.simulation_seed + replicate * 100003,
        ),
        heterogeneity=replace(
            baseline.heterogeneity,
            network_time_scale_multipliers=_network_contrast_map(
                networks,
                {
                    "DorsAttn": candidate.dorsattn_time_contrast,
                    "Vis": candidate.visual_time_contrast,
                },
            ),
            network_noise_multipliers=_network_contrast_map(
                networks,
                noise_contrasts,
            ),
        ),
        monitor=replace(
            baseline.monitor,
            surface_laplacian=config.empirical.apply_surface_laplacian,
        ),
    )


def simulate_spectral_candidate(
    baseline_config_path: str,
    candidate: SpectralCandidate,
    replicate: int,
    config: SpectralM5Config,
) -> dict[str, Any]:
    logging.raiseExceptions = False
    baseline, connectome, networks = _inputs(baseline_config_path)
    run_config = build_spectral_run_config(
        baseline, networks, candidate, config, replicate
    )
    candidate_connectome = connectome_for_spectral_candidate(
        connectome, networks, candidate
    )
    result = run_dual_jansen_rit(
        run_config,
        candidate_connectome,
        fast_ratio=candidate.fast_ratio,
        fast_fraction=candidate.fast_fraction,
        inhibitory_scale=1.0,
        speed_mm_per_ms=candidate.speed_mm_per_ms,
        cross_coupling=config.design.cross_coupling,
    )
    sfreq_hz = 1000.0 / run_config.simulation.monitor_period_ms
    frequency, csd, epoch_count = estimate_cross_spectrum(
        result.eeg, sfreq_hz, config.spectral
    )
    return {
        "candidate_index": candidate.candidate_index,
        "replicate": replicate,
        "frequency_hz": frequency,
        "channel_names": np.asarray(result.channel_names, dtype="U16"),
        "csd": csd,
        "epoch_count": epoch_count,
    }


def build_spectral_simulation_bank(
    config: SpectralM5Config,
    design_samples: int | None = None,
    candidates: list[SpectralCandidate] | None = None,
) -> SpectralSimulationBank:
    design_settings = (
        config.design
        if design_samples is None
        else replace(config.design, samples=design_samples)
    )
    working = replace(config, design=design_settings)
    candidates = candidates or make_spectral_design(design_settings)
    if len(candidates) != design_settings.samples:
        design_settings = replace(design_settings, samples=len(candidates))
        working = replace(config, design=design_settings)
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
                rows.append(
                    simulate_spectral_candidate(
                        str(config.paths.baseline_config), candidate, replicate, working
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "candidate_index": candidate.candidate_index,
                        "replicate": replicate,
                        "error": repr(error),
                    }
                )
            print(f"Spectral TVB bank: {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=design_settings.n_jobs) as pool:
            futures = {
                pool.submit(
                    simulate_spectral_candidate,
                    str(config.paths.baseline_config),
                    candidate,
                    replicate,
                    working,
                ): (candidate.candidate_index, replicate)
                for candidate, replicate in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                candidate_index, replicate = futures[future]
                try:
                    rows.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            "candidate_index": candidate_index,
                            "replicate": replicate,
                            "error": repr(error),
                        }
                    )
                if index % design_settings.n_jobs == 0 or index == len(futures):
                    print(f"Spectral TVB bank: {index}/{len(futures)}", flush=True)

    bank_dir = config.paths.output_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        failures, columns=["candidate_index", "replicate", "error"]
    ).to_csv(bank_dir / "simulation_failures.csv", index=False)
    if failures:
        raise RuntimeError(
            f"{len(failures)} spectral simulations failed; see simulation_failures.csv"
        )
    rows.sort(key=lambda row: (row["candidate_index"], row["replicate"]))
    frequency = rows[0]["frequency_hz"]
    channels = rows[0]["channel_names"]
    if any(
        not np.array_equal(row["frequency_hz"], frequency)
        or not np.array_equal(row["channel_names"], channels)
        for row in rows
    ):
        raise RuntimeError("Simulation bank spectral axes differ")
    csd = np.stack([row["csd"] for row in rows]).reshape(
        len(candidates), design_settings.replicates, *rows[0]["csd"].shape
    )
    bank = SpectralSimulationBank(
        parameters=np.stack([candidate.numeric_vector() for candidate in candidates]),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="U48"),
        frequency_hz=frequency,
        channel_names=channels,
        csd_replicates=csd,
    )
    np.savez_compressed(bank_dir / "spectral_simulation_bank.npz", **bank.__dict__)
    table = pd.DataFrame(bank.parameters, columns=bank.parameter_names)
    table.insert(0, "candidate_index", np.arange(len(candidates)))
    table.to_csv(bank_dir / "candidate_parameters.csv", index=False)
    pd.DataFrame(
        {
            "candidate_index": [row["candidate_index"] for row in rows],
            "replicate": [row["replicate"] for row in rows],
            "spectral_epoch_count": [row["epoch_count"] for row in rows],
        }
    ).to_csv(bank_dir / "replicates.csv", index=False)
    return bank


def load_spectral_simulation_bank(path: Path) -> SpectralSimulationBank:
    with np.load(path) as arrays:
        return SpectralSimulationBank(
            **{name: arrays[name] for name in SpectralSimulationBank.__dataclass_fields__}
        )
