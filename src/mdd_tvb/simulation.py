"""Configure and run the delayed whole-brain Jansen-Rit model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tvb.simulator import coupling, integrators, models, monitors, noise, simulator

from .config import RunConfig
from .connectome import ConnectomeBundle
from .eeg import (
    apply_surface_laplacian,
    build_eeg_monitor,
    regularize_analytic_eeg_gain,
)
from .heterogeneity import RegionalParameters, build_regional_parameters


@dataclass(frozen=True)
class SimulationResult:
    time_ms: np.ndarray
    region_psp: np.ndarray
    eeg: np.ndarray
    eeg_surface_laplacian: np.ndarray | None
    gain_matrix: np.ndarray
    channel_names: tuple[str, ...]
    region_labels: np.ndarray
    regional_parameters: RegionalParameters
    metadata: dict[str, Any]


def _scalar(value: float) -> np.ndarray:
    return np.array([float(value)], dtype=float)


def run_baseline(config: RunConfig, connectome: ConnectomeBundle) -> SimulationResult:
    """Run one reproducible stochastic resting-state baseline simulation."""

    sim_cfg = config.simulation
    np.random.seed(sim_cfg.seed)

    regional = build_regional_parameters(
        config.model,
        sim_cfg.noise_nsig,
        connectome.region_labels,
        config.heterogeneity,
    )
    model = models.JansenRit(
        A=_scalar(config.model.A),
        B=_scalar(config.model.B),
        a=regional.a,
        b=regional.b,
        J=_scalar(config.model.J),
        mu=regional.mu,
    )
    long_range_coupling = coupling.SigmoidalJansenRit(
        a=_scalar(config.coupling.global_gain)
    )
    # Apply stochastic forcing only to y4, the derivative state associated with
    # the excitatory input pathway. Adding equal noise directly to every JR
    # state would contaminate the PSP variables with broadband fluctuations.
    neural_noise = np.zeros(
        (len(model.state_variables), config.connectivity.expected_regions), dtype=float
    )
    neural_noise[4] = regional.noise_nsig
    stochastic_noise = noise.Additive(
        nsig=neural_noise,
        ntau=sim_cfg.noise_tau_ms,
        noise_seed=sim_cfg.seed,
    )
    integrator = integrators.HeunStochastic(
        dt=sim_cfg.dt_ms, noise=stochastic_noise
    )
    region_monitor = monitors.TemporalAverage(
        period=sim_cfg.monitor_period_ms,
        variables_of_interest=np.array([1, 2], dtype=np.int64),
    )
    eeg_monitor, _ = build_eeg_monitor(
        config.monitor,
        config.connectivity.expected_regions,
        sim_cfg.monitor_period_ms,
    )

    # TVB otherwise samples the full declared state-variable ranges to create
    # its delayed history. Those ranges are useful phase-plane bounds but can
    # create a numerically extreme transient for JR. A quiescent history lets
    # the constant drive and explicitly seeded noise start the dynamics cleanly.
    max_delay_ms = float(connectome.audit["max_delay_ms"])
    history_steps = int(np.ceil(max_delay_ms / sim_cfg.dt_ms)) + 2
    initial_history = np.zeros(
        (
            history_steps,
            len(model.state_variables),
            config.connectivity.expected_regions,
            model.number_of_modes,
        ),
        dtype=float,
    )

    engine = simulator.Simulator(
        model=model,
        connectivity=connectome.connectivity,
        coupling=long_range_coupling,
        integrator=integrator,
        monitors=(region_monitor, eeg_monitor),
        initial_conditions=initial_history,
    )
    engine.configure()
    gain_audit = regularize_analytic_eeg_gain(
        eeg_monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        config.monitor.minimum_source_sensor_distance_mm,
    )
    region_output, eeg_output = engine.run(simulation_length=sim_cfg.duration_ms)
    region_time, region_state = region_output
    eeg_time, eeg_state = eeg_output

    if not np.allclose(region_time, eeg_time, atol=sim_cfg.dt_ms):
        raise RuntimeError("Regional and EEG monitors returned different sample times")
    # TVB monitors y1 and y2 separately. Their difference is the pyramidal PSP.
    region_psp = region_state[:, 0, :, 0] - region_state[:, 1, :, 0]
    eeg = eeg_state[:, 0, :, 0] - eeg_state[:, 1, :, 0]
    # Supplying a complete delayed history advances TVB's internal clock.
    # Define burn-in relative to the start of this run, not that internal clock.
    relative_time = region_time - region_time[0] + sim_cfg.monitor_period_ms
    keep = relative_time > sim_cfg.transient_ms
    time_ms = relative_time[keep] - sim_cfg.transient_ms
    region_psp = region_psp[keep]
    eeg = eeg[keep]

    if not np.isfinite(region_psp).all() or not np.isfinite(eeg).all():
        raise FloatingPointError("Simulation produced non-finite values")
    if region_psp.shape[1] != config.connectivity.expected_regions:
        raise RuntimeError("Unexpected number of regional signals")
    if eeg.shape[1] != len(config.monitor.channels):
        raise RuntimeError("Unexpected number of EEG signals")

    sfreq_hz = 1000.0 / sim_cfg.monitor_period_ms
    eeg_csd = None
    if config.monitor.surface_laplacian:
        eeg_csd = apply_surface_laplacian(
            eeg, config.monitor.channels, sfreq_hz, config.monitor.montage
        )

    metadata: dict[str, Any] = {
        "model": "TVB JansenRit",
        "observation": "TVB EEG analytic single-sphere",
        "eeg_reprojected_from_saved_regional_psp": False,
        "observation_noise": "disabled",
        "regional_observable": "y1_minus_y2",
        "eeg_reference": config.monitor.reference,
        "surface_laplacian": config.monitor.surface_laplacian,
        "duration_ms": sim_cfg.duration_ms,
        "discarded_transient_ms": sim_cfg.transient_ms,
        "dt_ms": sim_cfg.dt_ms,
        "monitor_period_ms": sim_cfg.monitor_period_ms,
        "sample_frequency_hz": sfreq_hz,
        "seed": sim_cfg.seed,
        "noise_nsig": sim_cfg.noise_nsig,
        "noise_tau_ms": sim_cfg.noise_tau_ms,
        "noise_target_state": "y4_only",
        "global_coupling": config.coupling.global_gain,
        "samples_after_transient": int(time_ms.size),
        "average_reference_max_abs_mean": float(np.max(np.abs(eeg.mean(axis=1)))),
        "region_psp_standard_deviation": float(region_psp.std()),
        "eeg_standard_deviation": float(eeg.std()),
        "gain_shape": list(eeg_monitor.gain.shape),
        "analytic_gain_regularization": gain_audit,
        "initial_history": "quiescent_zero_history_covering_maximum_delay",
        "initial_history_steps": history_steps,
        "heterogeneity_enabled": config.heterogeneity.enabled,
        "regional_mu_range": [float(regional.mu.min()), float(regional.mu.max())],
        "regional_a_range": [float(regional.a.min()), float(regional.a.max())],
        "regional_b_range": [float(regional.b.min()), float(regional.b.max())],
        "regional_noise_nsig_range": [
            float(regional.noise_nsig.min()), float(regional.noise_nsig.max())
        ],
    }
    return SimulationResult(
        time_ms=time_ms,
        region_psp=region_psp,
        eeg=eeg,
        eeg_surface_laplacian=eeg_csd,
        gain_matrix=eeg_monitor.gain.copy(),
        channel_names=config.monitor.channels,
        region_labels=connectome.region_labels,
        regional_parameters=regional,
        metadata=metadata,
    )
