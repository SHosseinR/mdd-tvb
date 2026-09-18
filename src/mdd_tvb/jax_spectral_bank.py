"""GPU-batched construction of the M5 cross-spectral simulation bank."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .connectome import load_connectome
from .eeg import build_eeg_monitor, regularize_analytic_eeg_gain
from .heterogeneity import build_regional_parameters, network_labels
from .jax_backend import JaxDualBatch, run_dual_jansen_rit_jax
from .spectral_bank import (
    SpectralSimulationBank,
    build_spectral_run_config,
    connectome_for_spectral_candidate,
)
from .spectral_config import SpectralM5Config
from .spectral_features import estimate_cross_spectrum
from .spectral_parameterization import PARAMETER_NAMES, make_spectral_design


def _observation_gain(baseline: Any, connectome: Any) -> np.ndarray:
    monitor, _ = build_eeg_monitor(
        baseline.monitor,
        baseline.connectivity.expected_regions,
        baseline.simulation.monitor_period_ms,
    )
    monitor.configure()
    regularize_analytic_eeg_gain(
        monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        baseline.monitor.minimum_source_sensor_distance_mm,
    )
    return monitor.gain.copy()


def _save_bank(
    config: SpectralM5Config,
    candidates: list[Any],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
    backend_metadata: dict[str, Any],
) -> SpectralSimulationBank:
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
        len(candidates), config.design.replicates, *rows[0]["csd"].shape
    )
    bank = SpectralSimulationBank(
        parameters=np.stack([candidate.numeric_vector() for candidate in candidates]),
        parameter_names=np.asarray(PARAMETER_NAMES, dtype="U48"),
        frequency_hz=frequency,
        channel_names=channels,
        csd_replicates=csd,
    )
    bank_dir = config.paths.output_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
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
    pd.Series(
        {
            "backend": "jax",
            "wall_seconds": elapsed_seconds,
            "simulations": len(rows),
            **backend_metadata,
        }
    ).to_json(bank_dir / "backend_metadata.json", indent=2)
    pd.DataFrame(columns=["candidate_index", "replicate", "error"]).to_csv(
        bank_dir / "simulation_failures.csv", index=False
    )
    return bank


def build_spectral_simulation_bank_jax(
    config: SpectralM5Config,
    design_samples: int | None = None,
    *,
    batch_size: int = 64,
    precision: str = "float32",
) -> SpectralSimulationBank:
    """Build the same bank as the TVB path, batching candidates on JAX.

    TVB supplies configuration, connectome validation, spatial heterogeneity,
    and the analytic EEG gain.  Only repeated time integration and projection
    are delegated to JAX.  Cross-spectral estimation remains shared with the
    reference pipeline.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    design = (
        config.design
        if design_samples is None
        else replace(config.design, samples=design_samples)
    )
    working = replace(config, design=design)
    baseline = load_config(config.paths.baseline_config)
    connectome = load_connectome(baseline.paths, baseline.connectivity)
    networks = network_labels(connectome.region_labels)
    candidates = make_spectral_design(design)
    tasks = [
        (candidate, replicate)
        for candidate in candidates
        for replicate in range(design.replicates)
    ]
    gain = _observation_gain(baseline, connectome)
    max_history = int(
        np.rint(
            np.max(connectome.tract_lengths)
            / min(design.speed_range)
            / design.dt_ms
        )
    ) + 1
    rows: list[dict[str, Any]] = []
    backend_metadata: dict[str, Any] = {}
    started = perf_counter()

    for first in range(0, len(tasks), batch_size):
        selected = tasks[first : first + batch_size]
        run_configs = [
            build_spectral_run_config(baseline, networks, candidate, working, replicate)
            for candidate, replicate in selected
        ]
        regional = [
            build_regional_parameters(
                run_config.model,
                run_config.simulation.noise_nsig,
                connectome.region_labels,
                run_config.heterogeneity,
            )
            for run_config in run_configs
        ]
        candidate_weights = {
            candidate.candidate_index: connectome_for_spectral_candidate(
                connectome, networks, candidate
            ).weights
            for candidate, _ in selected
        }
        jax_batch = JaxDualBatch(
            weights=np.stack(
                [candidate_weights[candidate.candidate_index] for candidate, _ in selected]
            ),
            tract_lengths_ms_at_unit_speed=connectome.tract_lengths,
            gain_matrix=gain,
            regional_a=np.stack([item.a for item in regional]),
            regional_b=np.stack([item.b for item in regional]),
            regional_mu=np.stack([item.mu for item in regional]),
            regional_noise_nsig=np.stack([item.noise_nsig for item in regional]),
            global_coupling=np.asarray(
                [candidate.global_coupling for candidate, _ in selected]
            ),
            speed_mm_per_ms=np.asarray(
                [candidate.speed_mm_per_ms for candidate, _ in selected]
            ),
            fast_ratio=np.asarray([candidate.fast_ratio for candidate, _ in selected]),
            fast_fraction=np.asarray(
                [candidate.fast_fraction for candidate, _ in selected]
            ),
            noise_tau_ms=np.asarray(
                [candidate.noise_tau_ms for candidate, _ in selected]
            ),
            seeds=np.asarray(
                [design.simulation_seed + replicate * 100003 for _, replicate in selected],
                dtype=np.int32,
            ),
        )
        result = run_dual_jansen_rit_jax(
            jax_batch,
            duration_ms=design.duration_ms,
            transient_ms=design.transient_ms,
            dt_ms=design.dt_ms,
            monitor_period_ms=baseline.simulation.monitor_period_ms,
            cross_coupling=design.cross_coupling,
            reference=baseline.monitor.reference,
            channel_names=baseline.monitor.channels,
            history_steps=max_history,
            precision=precision,
            model_A=baseline.model.A,
            model_B=baseline.model.B,
            model_J=baseline.model.J,
        )
        backend_metadata = result.metadata
        sfreq_hz = 1000.0 / baseline.simulation.monitor_period_ms
        for local_index, (candidate, replicate) in enumerate(selected):
            eeg = result.eeg[local_index]
            if config.empirical.apply_surface_laplacian:
                # This branch intentionally imports lazily; M5 currently keeps
                # the physical sensor covariance and therefore disables CSD.
                from .eeg import apply_surface_laplacian

                eeg = apply_surface_laplacian(
                    eeg, baseline.monitor.channels, sfreq_hz, baseline.monitor
                )
            frequency, csd, epoch_count = estimate_cross_spectrum(
                eeg, sfreq_hz, config.spectral
            )
            rows.append(
                {
                    "candidate_index": candidate.candidate_index,
                    "replicate": replicate,
                    "frequency_hz": frequency,
                    "channel_names": np.asarray(baseline.monitor.channels, dtype="U16"),
                    "csd": csd,
                    "epoch_count": epoch_count,
                }
            )
        print(
            f"Spectral JAX bank: {min(first + len(selected), len(tasks))}/{len(tasks)}",
            flush=True,
        )

    return _save_bank(
        working,
        candidates,
        rows,
        perf_counter() - started,
        backend_metadata,
    )
