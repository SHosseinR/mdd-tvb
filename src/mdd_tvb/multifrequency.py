"""Two-generator Jansen--Rit whole-brain model for resting-state EEG.

The classical TVB Jansen--Rit node has one preferred alpha time scale.  This
module keeps the same equations but places an alpha and a faster cortical
column in every parcel.  The two columns exchange local excitatory input and
their firing rates jointly drive the delayed structural network.  The EEG
observable is their weighted pyramidal PSP.  This is deliberately called a
two-generator model rather than claiming a detailed laminar microcircuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import float64, guvectorize
from tvb.basic.neotraits.api import Final, List, NArray, Range
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator import integrators, monitors, noise, simulator
from tvb.simulator.coupling import Coupling
from tvb.simulator.models.base import ModelNumbaDfun

from .config import RunConfig
from .connectome import ConnectomeBundle
from .eeg import (
    apply_surface_laplacian,
    build_eeg_monitor,
    project_regional_psp_to_eeg,
    regularize_analytic_eeg_gain,
)
from .heterogeneity import RegionalParameters, build_regional_parameters


def _array(value: float) -> np.ndarray:
    return np.asarray([float(value)], dtype=float)


class MixedSigmoidalJansenRit(Coupling):
    """Long-range firing rate from a weighted alpha/fast PSP mixture."""

    cmin = NArray(default=_array(0.0), domain=Range(lo=-1.0, hi=1.0, step=0.01))
    cmax = NArray(default=_array(0.005), domain=Range(lo=0.0, hi=0.02, step=0.0001))
    midpoint = NArray(default=_array(5.52), domain=Range(lo=0.0, hi=10.0, step=0.01))
    r = NArray(default=_array(0.56), domain=Range(lo=0.01, hi=2.0, step=0.01))
    a = NArray(default=_array(1.0), domain=Range(lo=0.0, hi=100.0, step=0.1))
    fast_fraction = NArray(default=_array(0.25), domain=Range(lo=0.0, hi=1.0, step=0.01))

    def pre(self, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
        del x_i
        alpha_psp = x_j[:, 0] - x_j[:, 1]
        fast_psp = x_j[:, 2] - x_j[:, 3]
        alpha_rate = self.cmin + (self.cmax - self.cmin) / (
            1.0 + np.exp(self.r * (self.midpoint - alpha_psp))
        )
        fast_rate = self.cmin + (self.cmax - self.cmin) / (
            1.0 + np.exp(self.r * (self.midpoint - fast_psp))
        )
        mixed = (1.0 - self.fast_fraction) * alpha_rate + self.fast_fraction * fast_rate
        return mixed[:, np.newaxis]

    def post(self, gx: np.ndarray) -> np.ndarray:
        return self.a * gx


class DualJansenRit(ModelNumbaDfun):
    """Alpha and fast Jansen--Rit generators coupled within every parcel."""

    A = NArray(default=_array(3.25), domain=Range(lo=1.0, hi=10.0, step=0.05))
    B = NArray(default=_array(22.0), domain=Range(lo=5.0, hi=80.0, step=0.1))
    a = NArray(default=_array(0.13), domain=Range(lo=0.03, hi=0.3, step=0.001))
    b = NArray(default=_array(0.065), domain=Range(lo=0.01, hi=0.2, step=0.001))
    fast_ratio = NArray(default=_array(2.0), domain=Range(lo=1.1, hi=4.0, step=0.01))
    J = NArray(default=_array(135.0), domain=Range(lo=20.0, hi=1000.0, step=1.0))
    mu = NArray(default=_array(0.22), domain=Range(lo=0.0, hi=0.5, step=0.001))
    fast_drive_ratio = NArray(default=_array(0.85), domain=Range(lo=0.1, hi=2.0, step=0.01))
    inhibitory_scale = NArray(default=_array(1.0), domain=Range(lo=0.5, hi=2.0, step=0.01))
    cross_coupling = NArray(default=_array(0.12), domain=Range(lo=0.0, hi=1.0, step=0.01))
    v0 = NArray(default=_array(5.52), domain=Range(lo=3.0, hi=8.0, step=0.01))
    nu_max = NArray(default=_array(0.0025), domain=Range(lo=0.001, hi=0.01, step=0.0001))
    r = NArray(default=_array(0.56), domain=Range(lo=0.1, hi=1.5, step=0.01))
    a_1 = NArray(default=_array(1.0), domain=Range(lo=0.1, hi=2.0, step=0.01))
    a_2 = NArray(default=_array(0.8), domain=Range(lo=0.1, hi=2.0, step=0.01))
    a_3 = NArray(default=_array(0.25), domain=Range(lo=0.05, hi=1.0, step=0.01))
    a_4 = NArray(default=_array(0.25), domain=Range(lo=0.05, hi=1.0, step=0.01))

    state_variables = (
        "a_y0", "a_y1", "a_y2", "a_y3", "a_y4", "a_y5",
        "f_y0", "f_y1", "f_y2", "f_y3", "f_y4", "f_y5",
    )
    _nvar = 12
    # The custom coupling turns these four delayed PSP states into one mixed
    # firing-rate input for both local generators.
    cvar = np.asarray([1, 2, 7, 8], dtype=np.int32)
    variables_of_interest = List(
        of=str,
        choices=state_variables,
        default=state_variables,
    )
    state_variable_range = Final(
        default={
            name: np.asarray([-500.0, 500.0])
            for name in state_variables
        }
    )

    def _sigmoid(self, value: np.ndarray) -> np.ndarray:
        argument = np.clip(self.r * (self.v0 - value), -60.0, 60.0)
        return 2.0 * self.nu_max / (1.0 + np.exp(argument))

    def _generator_derivative(
        self,
        states: tuple[np.ndarray, ...],
        a: np.ndarray,
        b: np.ndarray,
        drive: np.ndarray,
        long_range: np.ndarray,
        other_pyramidal_rate: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        y0, y1, y2, y3, y4, y5 = states
        output_rate = self._sigmoid(y1 - y2)
        excitatory_rate = self._sigmoid(self.a_1 * self.J * y0)
        inhibitory_rate = self._sigmoid(self.a_3 * self.J * y0)
        local_cross = self.cross_coupling * self.J * other_pyramidal_rate
        return (
            y3,
            y4,
            y5,
            self.A * a * output_rate - 2.0 * a * y3 - a**2 * y0,
            self.A * a * (
                drive + self.a_2 * self.J * excitatory_rate + long_range + local_cross
            ) - 2.0 * a * y4 - a**2 * y1,
            self.B * self.inhibitory_scale * b * (
                self.a_4 * self.J * inhibitory_rate
            ) - 2.0 * b * y5 - b**2 * y2,
        )

    def dfun(
        self,
        state_variables: np.ndarray,
        coupling: np.ndarray,
        local_coupling: float = 0.0,
    ) -> np.ndarray:
        source = local_coupling * (
            state_variables[1] - state_variables[2]
        )[:, 0]
        states = state_variables.reshape(state_variables.shape[:-1]).T
        delayed = coupling.reshape(coupling.shape[:-1]).T
        derivative = _numba_dual_jansen_rit(
            states,
            delayed,
            source,
            self.nu_max,
            self.r,
            self.v0,
            self.a,
            self.a_1,
            self.a_2,
            self.a_3,
            self.a_4,
            self.A,
            self.b,
            self.B,
            self.J,
            self.mu,
            self.fast_ratio,
            self.fast_drive_ratio,
            self.inhibitory_scale,
            self.cross_coupling,
        )
        return derivative.T[..., np.newaxis]


@guvectorize(
    [(float64[:],) * 21],
    "(n),(m)" + ",()" * 18 + "->(n)",
    nopython=True,
)
def _numba_dual_jansen_rit(
    y,
    c,
    source,
    nu_max,
    r,
    v0,
    a,
    a_1,
    a_2,
    a_3,
    a_4,
    A,
    b,
    B,
    J,
    mu,
    fast_ratio,
    fast_drive_ratio,
    inhibitory_scale,
    cross_coupling,
    dx,
):
    # Numba's scalar math here is substantially faster than dispatching a
    # Python/numpy derivative for every 0.5-ms integration step.
    def sigmoid(value):
        argument = r[0] * (v0[0] - value)
        if argument > 60.0:
            argument = 60.0
        elif argument < -60.0:
            argument = -60.0
        return 2.0 * nu_max[0] / (1.0 + np.exp(argument))

    alpha_output = sigmoid(y[1] - y[2])
    alpha_excitatory = sigmoid(a_1[0] * J[0] * y[0])
    alpha_inhibitory = sigmoid(a_3[0] * J[0] * y[0])
    fast_output = sigmoid(y[7] - y[8])
    fast_excitatory = sigmoid(a_1[0] * J[0] * y[6])
    fast_inhibitory = sigmoid(a_3[0] * J[0] * y[6])
    long_range = c[0] + source[0]
    alpha_cross = cross_coupling[0] * J[0] * fast_excitatory
    fast_cross = cross_coupling[0] * J[0] * alpha_excitatory
    alpha_a = a[0]
    alpha_b = b[0]
    fast_a = alpha_a * fast_ratio[0]
    fast_b = alpha_b * fast_ratio[0]
    scaled_B = B[0] * inhibitory_scale[0]

    dx[0] = y[3]
    dx[1] = y[4]
    dx[2] = y[5]
    dx[3] = A[0] * alpha_a * alpha_output - 2.0 * alpha_a * y[3] - alpha_a**2 * y[0]
    dx[4] = A[0] * alpha_a * (
        mu[0] + a_2[0] * J[0] * alpha_excitatory + long_range + alpha_cross
    ) - 2.0 * alpha_a * y[4] - alpha_a**2 * y[1]
    dx[5] = scaled_B * alpha_b * (
        a_4[0] * J[0] * alpha_inhibitory
    ) - 2.0 * alpha_b * y[5] - alpha_b**2 * y[2]

    dx[6] = y[9]
    dx[7] = y[10]
    dx[8] = y[11]
    dx[9] = A[0] * fast_a * fast_output - 2.0 * fast_a * y[9] - fast_a**2 * y[6]
    dx[10] = A[0] * fast_a * (
        mu[0] * fast_drive_ratio[0]
        + a_2[0] * J[0] * fast_excitatory
        + long_range
        + fast_cross
    ) - 2.0 * fast_a * y[10] - fast_a**2 * y[7]
    dx[11] = scaled_B * fast_b * (
        a_4[0] * J[0] * fast_inhibitory
    ) - 2.0 * fast_b * y[11] - fast_b**2 * y[8]


@dataclass(frozen=True)
class DualSimulationResult:
    time_ms: np.ndarray
    region_psp: np.ndarray
    eeg: np.ndarray
    gain_matrix: np.ndarray
    channel_names: tuple[str, ...]
    regional_parameters: RegionalParameters
    metadata: dict[str, Any]


def _connectivity_with_speed(
    bundle: ConnectomeBundle, speed_mm_per_ms: float
) -> Connectivity:
    conn = Connectivity(
        weights=bundle.weights.copy(),
        tract_lengths=bundle.tract_lengths.copy(),
        centres=bundle.centres.copy(),
        region_labels=bundle.region_labels.copy(),
        orientations=bundle.connectivity.orientations.copy(),
        cortical=bundle.connectivity.cortical.copy(),
        hemispheres=bundle.connectivity.hemispheres.copy(),
        speed=_array(speed_mm_per_ms),
    )
    conn.configure()
    return conn


def run_dual_jansen_rit(
    config: RunConfig,
    connectome: ConnectomeBundle,
    *,
    fast_ratio: float,
    fast_fraction: float,
    inhibitory_scale: float,
    speed_mm_per_ms: float,
    cross_coupling: float = 0.12,
    observation_gain: tuple[np.ndarray, dict[str, Any]] | None = None,
) -> DualSimulationResult:
    """Run the two-generator model with a matched TVB EEG observation model."""

    sim_cfg = config.simulation
    np.random.seed(sim_cfg.seed)
    regional = build_regional_parameters(
        config.model,
        sim_cfg.noise_nsig,
        connectome.region_labels,
        config.heterogeneity,
    )
    model = DualJansenRit(
        A=_array(config.model.A),
        B=_array(config.model.B),
        a=regional.a,
        b=regional.b,
        J=_array(config.model.J),
        mu=regional.mu,
        fast_ratio=_array(fast_ratio),
        inhibitory_scale=_array(inhibitory_scale),
        cross_coupling=_array(cross_coupling),
    )
    long_range = MixedSigmoidalJansenRit(
        a=_array(config.coupling.global_gain),
        fast_fraction=_array(fast_fraction),
        midpoint=_array(5.52),
        r=_array(0.56),
        cmax=_array(0.005),
    )
    neural_noise = np.zeros(
        (model.nvar, config.connectivity.expected_regions), dtype=float
    )
    neural_noise[4] = regional.noise_nsig
    neural_noise[10] = regional.noise_nsig
    stochastic_noise = noise.Additive(
        nsig=neural_noise,
        ntau=sim_cfg.noise_tau_ms,
        noise_seed=sim_cfg.seed,
    )
    integrator = integrators.HeunStochastic(dt=sim_cfg.dt_ms, noise=stochastic_noise)
    observed_variables = np.asarray([1, 2, 7, 8], dtype=np.int64)
    region_monitor = monitors.TemporalAverage(
        period=sim_cfg.monitor_period_ms,
        variables_of_interest=observed_variables,
    )
    eeg_monitor, _ = build_eeg_monitor(
        config.monitor,
        config.connectivity.expected_regions,
        sim_cfg.monitor_period_ms,
    )
    eeg_monitor.variables_of_interest = observed_variables

    dynamic_connectivity = _connectivity_with_speed(connectome, speed_mm_per_ms)
    max_delay_ms = float(np.max(connectome.tract_lengths) / speed_mm_per_ms)
    history_steps = int(np.ceil(max_delay_ms / sim_cfg.dt_ms)) + 2
    initial_history = np.zeros(
        (history_steps, model.nvar, config.connectivity.expected_regions, 1),
        dtype=float,
    )
    engine = simulator.Simulator(
        model=model,
        connectivity=dynamic_connectivity,
        coupling=long_range,
        integrator=integrator,
        monitors=(region_monitor, eeg_monitor),
        initial_conditions=initial_history,
    )
    engine.configure()
    gain_audit = regularize_analytic_eeg_gain(
        eeg_monitor,
        connectome.centres,
        dynamic_connectivity.orientations,
        config.monitor.minimum_source_sensor_distance_mm,
    )
    region_output, eeg_output = engine.run(simulation_length=sim_cfg.duration_ms)
    region_time, region_state = region_output
    eeg_time, eeg_state = eeg_output
    if not np.allclose(region_time, eeg_time, atol=sim_cfg.dt_ms):
        raise RuntimeError("Regional and EEG monitors returned different sample times")
    alpha_weight = 1.0 - float(fast_fraction)
    region_psp = alpha_weight * (
        region_state[:, 0, :, 0] - region_state[:, 1, :, 0]
    ) + fast_fraction * (
        region_state[:, 2, :, 0] - region_state[:, 3, :, 0]
    )
    if observation_gain is None:
        eeg = alpha_weight * (
            eeg_state[:, 0, :, 0] - eeg_state[:, 1, :, 0]
        ) + fast_fraction * (
            eeg_state[:, 2, :, 0] - eeg_state[:, 3, :, 0]
        )
        active_gain = eeg_monitor.gain.copy()
        observation_audit: dict[str, Any] = {
            "observation_gain": "analytic_single_sphere",
            **gain_audit,
        }
    else:
        active_gain, observation_audit = observation_gain
        if active_gain.shape != (
            len(config.monitor.channels),
            config.connectivity.expected_regions,
        ):
            raise ValueError("Custom observation gain has incompatible shape")
        eeg = project_regional_psp_to_eeg(
            region_psp,
            active_gain,
            config.monitor.reference,
            config.monitor.channels,
        )
    relative_time = region_time - region_time[0] + sim_cfg.monitor_period_ms
    keep = relative_time > sim_cfg.transient_ms
    time_ms = relative_time[keep] - sim_cfg.transient_ms
    region_psp = region_psp[keep]
    eeg = eeg[keep]
    if not np.isfinite(region_psp).all() or not np.isfinite(eeg).all():
        raise FloatingPointError("Dual Jansen--Rit simulation produced non-finite values")
    sfreq_hz = 1000.0 / sim_cfg.monitor_period_ms
    if config.monitor.surface_laplacian:
        eeg = apply_surface_laplacian(
            eeg, config.monitor.channels, sfreq_hz, config.monitor
        )
    return DualSimulationResult(
        time_ms=time_ms,
        region_psp=region_psp,
        eeg=eeg,
        gain_matrix=active_gain,
        channel_names=config.monitor.channels,
        regional_parameters=regional,
        metadata={
            "model": "dual-generator Jansen-Rit",
            "generators_per_region": 2,
            "fast_ratio": float(fast_ratio),
            "fast_fraction": float(fast_fraction),
            "inhibitory_scale": float(inhibitory_scale),
            "cross_coupling": float(cross_coupling),
            "speed_mm_per_ms": float(speed_mm_per_ms),
            "global_coupling": float(config.coupling.global_gain),
            "duration_ms": float(sim_cfg.duration_ms),
            "discarded_transient_ms": float(sim_cfg.transient_ms),
            "sample_frequency_hz": float(sfreq_hz),
            "seed": int(sim_cfg.seed),
            "noise_nsig": float(sim_cfg.noise_nsig),
            "noise_tau_ms": float(sim_cfg.noise_tau_ms),
            "gain_regularization": observation_audit,
        },
    )
