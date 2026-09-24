"""Analytic cross spectra of the linearised delayed dual Jansen--Rit network.

Resting EEG is usually modelled as noise-driven fluctuations around a stable
operating point (steady-state DCM, Robinson's corticothalamic field model,
spectral graph models).  In that regime the expected sensor cross-spectral
density of the stochastic simulator is available in closed form:

``S_V(w) = L M(w) diag(S_noise(w)) M(w)^H L^T``

where ``M`` is the transfer function from the regional noise inputs to the
regional EEG observable through the delayed structural network.  This module
computes that expectation for exactly the equations integrated by
``multifrequency.DualJansenRit`` and ``jax_backend.run_dual_jansen_rit_jax``:
the same local generators, mixed sigmoidal long-range coupling, integer-step
conduction delays, Ornstein--Uhlenbeck input noise on ``y4``/``y10`` and a
linear average-referenced lead field.

The result replaces a finite stochastic bank by a deterministic, smooth and
cheap forward model.  It is exact only for small fluctuations about a stable
fixed point; ``validate`` style comparisons against the stochastic simulator
are therefore required before scientific use (see
``scripts/audit_linear_spectral.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Constants shared with DualJansenRit / the JAX backend.
A_CONST = 3.25
B_CONST = 22.0
J_CONST = 135.0
V0 = 5.52
NU_MAX = 0.0025
SLOPE = 0.56
A1, A2, A3, A4 = 1.0, 0.8, 0.25, 0.25
FAST_DRIVE_RATIO = 0.85
CMAX = 0.005


def sigmoid(value: np.ndarray) -> np.ndarray:
    argument = np.clip(SLOPE * (V0 - value), -60.0, 60.0)
    return 2.0 * NU_MAX / (1.0 + np.exp(argument))


def sigmoid_slope(value: np.ndarray) -> np.ndarray:
    rate = sigmoid(value)
    return SLOPE * rate * (1.0 - rate / (2.0 * NU_MAX))


@dataclass(frozen=True)
class LinearNetworkParameters:
    """Numerical inputs of one linearised network state."""

    weights: np.ndarray  # (region, region), target x source
    delays_ms: np.ndarray  # (region, region)
    a: np.ndarray  # alpha excitatory inverse time constants (region,)
    b: np.ndarray  # alpha inhibitory inverse time constants (region,)
    mu: np.ndarray  # mean drive (region,)
    noise_nsig: np.ndarray  # TVB diffusion intensity (region,)
    global_coupling: float
    fast_ratio: float
    fast_fraction: float
    noise_tau_ms: float
    cross_coupling: float = 0.12
    inhibitory_scale: float = 1.0
    # Optional short-range (intracortical / U-fibre) coupling with its own
    # slow conduction delays.  Zero reproduces the tract-only model.
    local_weights: np.ndarray | None = None
    local_delays_ms: np.ndarray | None = None
    local_coupling: float = 0.0

    def effective_static_weights(self) -> np.ndarray:
        """Coupling seen by the equilibrium (delays do not matter at rest)."""
        total = self.global_coupling * self.weights
        if self.local_weights is not None and self.local_coupling:
            total = total + self.local_coupling * self.local_weights
        return total

    def kernel(self, omega: np.ndarray) -> np.ndarray:
        kernel = (
            self.global_coupling
            * self.weights[None]
            * np.exp(-1j * omega[:, None, None] * self.delays_ms[None])
        )
        if self.local_weights is not None and self.local_coupling:
            kernel = kernel + self.local_coupling * self.local_weights[None] * np.exp(
                -1j * omega[:, None, None] * self.local_delays_ms[None]
            )
        return kernel


@dataclass(frozen=True)
class FixedPoint:
    psi_alpha: np.ndarray
    psi_fast: np.ndarray
    y0: np.ndarray
    y6: np.ndarray
    long_range: np.ndarray
    residual: float
    iterations: int


def _psi_map(
    psi: np.ndarray, params: LinearNetworkParameters
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return Phi(psi) whose fixed point is the network equilibrium."""

    n = params.a.size
    psi_alpha, psi_fast = psi[:n], psi[n:]
    a = params.a
    b = params.b
    af = a * params.fast_ratio
    bf = b * params.fast_ratio
    f = params.fast_fraction
    s_alpha = sigmoid(psi_alpha)
    s_fast = sigmoid(psi_fast)
    y0 = A_CONST / a * s_alpha
    y6 = A_CONST / af * s_fast
    long_range = params.effective_static_weights() @ (
        (1.0 - f) * s_alpha + f * s_fast
    )
    exc_alpha = sigmoid(A1 * J_CONST * y0)
    exc_fast = sigmoid(A1 * J_CONST * y6)
    inh_alpha = sigmoid(A3 * J_CONST * y0)
    inh_fast = sigmoid(A3 * J_CONST * y6)
    cross = params.cross_coupling * J_CONST
    b_scaled = B_CONST * params.inhibitory_scale
    new_alpha = A_CONST / a * (
        params.mu + A2 * J_CONST * exc_alpha + long_range + cross * exc_fast
    ) - b_scaled / b * A4 * J_CONST * inh_alpha
    new_fast = A_CONST / af * (
        params.mu * FAST_DRIVE_RATIO
        + A2 * J_CONST * exc_fast
        + long_range
        + cross * exc_alpha
    ) - b_scaled / bf * A4 * J_CONST * inh_fast
    return np.concatenate((new_alpha, new_fast)), {
        "y0": y0,
        "y6": y6,
        "long_range": long_range,
    }


def _psi_map_jacobian(psi: np.ndarray, params: LinearNetworkParameters) -> np.ndarray:
    n = params.a.size
    psi_alpha, psi_fast = psi[:n], psi[n:]
    a = params.a
    b = params.b
    af = a * params.fast_ratio
    bf = b * params.fast_ratio
    f = params.fast_fraction
    ds_alpha = sigmoid_slope(psi_alpha)
    ds_fast = sigmoid_slope(psi_fast)
    y0 = A_CONST / a * sigmoid(psi_alpha)
    y6 = A_CONST / af * sigmoid(psi_fast)
    dy0 = A_CONST / a * ds_alpha
    dy6 = A_CONST / af * ds_fast
    d_exc_alpha = sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST * dy0
    d_exc_fast = sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST * dy6
    d_inh_alpha = sigmoid_slope(A3 * J_CONST * y0) * A3 * J_CONST * dy0
    d_inh_fast = sigmoid_slope(A3 * J_CONST * y6) * A3 * J_CONST * dy6
    cross = params.cross_coupling * J_CONST
    b_scaled = B_CONST * params.inhibitory_scale
    gw = params.effective_static_weights()
    # d long_range / d psi_alpha and psi_fast.
    dlr_alpha = gw * ((1.0 - f) * ds_alpha)[np.newaxis, :]
    dlr_fast = gw * (f * ds_fast)[np.newaxis, :]
    jac = np.zeros((2 * n, 2 * n))
    ka = (A_CONST / a)[:, np.newaxis]
    kf = (A_CONST / af)[:, np.newaxis]
    jac[:n, :n] = ka * dlr_alpha
    jac[:n, n:] = ka * dlr_fast
    jac[n:, :n] = kf * dlr_alpha
    jac[n:, n:] = kf * dlr_fast
    idx = np.arange(n)
    jac[idx, idx] += (
        A_CONST / a * A2 * J_CONST * d_exc_alpha
        - b_scaled / b * A4 * J_CONST * d_inh_alpha
    )
    jac[idx, n + idx] += A_CONST / a * cross * d_exc_fast
    jac[n + idx, n + idx] += (
        A_CONST / af * A2 * J_CONST * d_exc_fast
        - b_scaled / bf * A4 * J_CONST * d_inh_fast
    )
    jac[n + idx, idx] += A_CONST / af * cross * d_exc_alpha
    return jac


def solve_fixed_point(
    params: LinearNetworkParameters,
    initial: np.ndarray | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> FixedPoint:
    """Damped Newton solve of the equilibrium PSPs of both generators."""

    n = params.a.size
    # Default start on the upper (alpha-generating) branch.  Where the network
    # is multistable this is the equilibrium the simulator reaches from rest;
    # a zero start can converge to the lower branch instead.
    psi = (
        np.concatenate((np.full(n, 7.0), np.full(n, 1.5)))
        if initial is None
        else np.asarray(initial, float).copy()
    )
    residual_norm = np.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        mapped, _ = _psi_map(psi, params)
        residual = psi - mapped
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm < tolerance:
            break
        jac = np.eye(2 * n) - _psi_map_jacobian(psi, params)
        step = np.linalg.solve(jac, residual)
        scale = 1.0
        for _ in range(30):
            trial = psi - scale * step
            trial_residual = trial - _psi_map(trial, params)[0]
            if np.max(np.abs(trial_residual)) < residual_norm:
                break
            scale *= 0.5
        psi = trial
    mapped, extra = _psi_map(psi, params)
    return FixedPoint(
        psi_alpha=psi[:n],
        psi_fast=psi[n:],
        y0=extra["y0"],
        y6=extra["y6"],
        long_range=extra["long_range"],
        residual=float(np.max(np.abs(psi - mapped))),
        iterations=iteration,
    )


def local_jacobians(
    params: LinearNetworkParameters, point: FixedPoint
) -> np.ndarray:
    """Return per-region 12x12 Jacobians excluding delayed network input."""

    n = params.a.size
    a = params.a
    b = params.b
    af = a * params.fast_ratio
    bf = b * params.fast_ratio
    cross = params.cross_coupling * J_CONST
    b_scaled = B_CONST * params.inhibitory_scale
    ds_out_alpha = sigmoid_slope(point.psi_alpha)
    ds_out_fast = sigmoid_slope(point.psi_fast)
    ds_exc_alpha = sigmoid_slope(A1 * J_CONST * point.y0) * A1 * J_CONST
    ds_exc_fast = sigmoid_slope(A1 * J_CONST * point.y6) * A1 * J_CONST
    ds_inh_alpha = sigmoid_slope(A3 * J_CONST * point.y0) * A3 * J_CONST
    ds_inh_fast = sigmoid_slope(A3 * J_CONST * point.y6) * A3 * J_CONST
    jac = np.zeros((n, 12, 12))
    for offset, rate, inhibitory, out_slope, exc_slope, inh_slope, other_exc, other_offset in (
        (0, a, b, ds_out_alpha, ds_exc_alpha, ds_inh_alpha, ds_exc_fast, 6),
        (6, af, bf, ds_out_fast, ds_exc_fast, ds_inh_fast, ds_exc_alpha, 0),
    ):
        y0, y1, y2, y3, y4, y5 = (offset + k for k in range(6))
        jac[:, y0, y3] = 1.0
        jac[:, y1, y4] = 1.0
        jac[:, y2, y5] = 1.0
        jac[:, y3, y1] = A_CONST * rate * out_slope
        jac[:, y3, y2] = -A_CONST * rate * out_slope
        jac[:, y3, y3] = -2.0 * rate
        jac[:, y3, y0] = -rate**2
        jac[:, y4, y0] = A_CONST * rate * A2 * J_CONST * exc_slope
        jac[:, y4, other_offset] = A_CONST * rate * cross * other_exc
        jac[:, y4, y4] = -2.0 * rate
        jac[:, y4, y1] = -rate**2
        jac[:, y5, y0] = b_scaled * inhibitory * A4 * J_CONST * inh_slope
        jac[:, y5, y5] = -2.0 * inhibitory
        jac[:, y5, y2] = -inhibitory**2
    return jac


def _local_responses(
    params: LinearNetworkParameters,
    jac: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """Return responses of (psi_alpha, psi_fast) to inputs on y4 and y10.

    Output shape is (frequency, region, 2 outputs, 2 inputs).
    """

    n = jac.shape[0]
    eye = np.eye(12)
    system = 1j * omega[:, None, None, None] * eye - jac[None]
    rhs = np.zeros((12, 2))
    rhs[4, 0] = 1.0
    rhs[10, 1] = 1.0
    solution = np.linalg.solve(system, np.broadcast_to(rhs, system.shape[:-1] + (2,)))
    out = np.empty((omega.size, n, 2, 2), dtype=complex)
    out[:, :, 0, :] = solution[:, :, 1, :] - solution[:, :, 2, :]
    out[:, :, 1, :] = solution[:, :, 7, :] - solution[:, :, 8, :]
    return out


def linear_regional_transfer(
    params: LinearNetworkParameters,
    frequency_hz: np.ndarray,
    point: FixedPoint | None = None,
) -> tuple[np.ndarray, np.ndarray, FixedPoint]:
    """Return regional transfer and noise spectra.

    The regional observable is ``(1-f) psi_alpha + f psi_fast``.  The returned
    transfer has shape ``(frequency, region, 2*region)`` and maps the alpha and
    fast noise input of every region (interleaved) to that observable.
    """

    point = point or solve_fixed_point(params)
    frequency = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * frequency / 1000.0  # rad/ms
    jac = local_jacobians(params, point)
    response = _local_responses(params, jac, omega)
    n = params.a.size
    f = params.fast_fraction
    af = params.a * params.fast_ratio
    input_gain = np.stack((A_CONST * params.a, A_CONST * af), axis=1)  # (n, 2)
    rate_slope = np.stack(
        ((1.0 - f) * sigmoid_slope(point.psi_alpha), f * sigmoid_slope(point.psi_fast)),
        axis=1,
    )
    observable = np.asarray([1.0 - f, f])
    # Long-range input enters y4 and y10 with the local A*a factors.
    long_range_response = np.einsum("fnoi,ni->fno", response, input_gain)
    h = np.einsum("no,fno->fn", rate_slope, long_range_response)
    q = np.einsum("o,fno->fn", observable, long_range_response)
    g = np.einsum("no,fnoi->fni", rate_slope, response)
    p = np.einsum("o,fnoi->fni", observable, response)
    kernel = params.kernel(omega)
    # z = (I - diag(h) K)^-1 G_n N ; s = diag(q) K z + P_n N
    system = np.eye(n)[None] - h[:, :, None] * kernel
    noise_to_z = np.zeros((frequency.size, n, 2 * n), dtype=complex)
    idx = np.arange(n)
    noise_to_z[:, idx, 2 * idx] = g[:, :, 0]
    noise_to_z[:, idx, 2 * idx + 1] = g[:, :, 1]
    z_transfer = np.linalg.solve(system, noise_to_z)
    transfer = q[:, :, None] * (kernel @ z_transfer)
    transfer[:, idx, 2 * idx] += p[:, :, 0]
    transfer[:, idx, 2 * idx + 1] += p[:, :, 1]
    omega_tau = omega * params.noise_tau_ms
    noise_psd = (
        4.0 * params.noise_nsig[None, :] / (1.0 + omega_tau[:, None] ** 2)
    )
    noise_psd = np.repeat(noise_psd, 2, axis=1)
    return transfer, noise_psd, point


def sensor_csd_from_transfer(
    gain: np.ndarray,
    transfer: np.ndarray,
    noise_psd: np.ndarray,
    reference: str = "average",
) -> np.ndarray:
    """Return E[X_c^* X_d] on the transfer frequency grid."""

    lead = np.asarray(gain, dtype=float)
    if reference and reference.lower() == "average":
        lead = lead - lead.mean(axis=0, keepdims=True)
    sensor_transfer = np.einsum("cr,frk->fck", lead, transfer)
    weighted = sensor_transfer * np.sqrt(noise_psd)[:, None, :]
    return np.einsum("fck,fdk->fcd", np.conjugate(weighted), weighted)


def welch_expectation_grid(
    frequency_min_hz: float,
    frequency_max_hz: float,
    bin_hz: float,
    epoch_seconds: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return FFT frequencies used by ``estimate_cross_spectrum`` per bin."""

    resolution = 1.0 / epoch_seconds
    first = np.ceil(frequency_min_hz / bin_hz) * bin_hz
    centers = np.arange(first, frequency_max_hz + bin_hz * 0.25, bin_hz)
    fft = np.arange(0.0, frequency_max_hz + bin_hz, resolution)
    groups = [
        np.flatnonzero((fft >= center - bin_hz / 2.0) & (fft < center + bin_hz / 2.0))
        for center in centers
    ]
    used = np.unique(np.concatenate(groups))
    remap = {old: new for new, old in enumerate(used)}
    return fft[used], [np.asarray([remap[i] for i in group]) for group in groups]


def bin_csd(csd_fine: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    binned = np.stack([csd_fine[group].mean(axis=0) for group in groups])
    return 0.5 * (binned + np.conjugate(np.swapaxes(binned, -1, -2)))


def linear_sensor_csd(
    params: LinearNetworkParameters,
    gain: np.ndarray,
    *,
    frequency_min_hz: float = 2.0,
    frequency_max_hz: float = 40.0,
    bin_hz: float = 1.0,
    epoch_seconds: float = 4.0,
    reference: str = "average",
    monitor_period_ms: float = 2.0,
) -> tuple[np.ndarray, FixedPoint]:
    """Expected binned sensor CSD matching ``estimate_cross_spectrum``."""

    fine, groups = welch_expectation_grid(
        frequency_min_hz, frequency_max_hz, bin_hz, epoch_seconds
    )
    transfer, noise_psd, point = linear_regional_transfer(params, fine)
    csd = sensor_csd_from_transfer(gain, transfer, noise_psd, reference)
    # TVB/JAX temporal-average monitor: boxcar of one sampling period.
    boxcar = np.sinc(fine * monitor_period_ms / 1000.0) ** 2
    csd = csd * boxcar[:, None, None]
    return bin_csd(csd, groups), point


class CandidateLinearizer:
    """Map M5 spectral candidates to linear-network parameters.

    Uses the same run-configuration, heterogeneity and connectome functions as
    the TVB and JAX banks so that the analytic spectrum describes precisely the
    simulated candidate.
    """

    def __init__(self, config) -> None:  # SpectralM5Config
        from .config import load_config
        from .connectome import load_connectome
        from .heterogeneity import network_labels
        from .spectral_bank import spectral_observation_gain

        self.config = config
        self.baseline = load_config(config.paths.baseline_config)
        self.connectome = load_connectome(
            self.baseline.paths, self.baseline.connectivity
        )
        self.networks = network_labels(self.connectome.region_labels)
        gain, _ = spectral_observation_gain(config, self.baseline, self.connectome)
        self.gain = np.asarray(gain, dtype=float)

    def parameters(self, vector: np.ndarray) -> LinearNetworkParameters:
        from .heterogeneity import build_regional_parameters
        from .spectral_bank import (
            build_spectral_run_config,
            connectome_for_spectral_candidate,
        )
        from .spectral_parameterization import PARAMETER_NAMES, SpectralCandidate

        values = np.asarray(vector, dtype=float)
        candidate = SpectralCandidate(
            candidate_index=0,
            **{name: float(values[i]) for i, name in enumerate(PARAMETER_NAMES)},
        )
        run = build_spectral_run_config(
            self.baseline, self.networks, candidate, self.config, 0
        )
        regional = build_regional_parameters(
            run.model,
            run.simulation.noise_nsig,
            self.connectome.region_labels,
            run.heterogeneity,
        )
        weights = connectome_for_spectral_candidate(
            self.connectome, self.networks, candidate
        ).weights
        dt = self.config.design.dt_ms
        delays = np.rint(
            self.connectome.tract_lengths / candidate.speed_mm_per_ms / dt
        ) * dt
        return LinearNetworkParameters(
            weights=np.asarray(weights, dtype=float),
            delays_ms=delays,
            a=np.asarray(regional.a, dtype=float),
            b=np.asarray(regional.b, dtype=float),
            mu=np.asarray(regional.mu, dtype=float),
            noise_nsig=np.asarray(regional.noise_nsig, dtype=float),
            global_coupling=candidate.global_coupling,
            fast_ratio=candidate.fast_ratio,
            fast_fraction=candidate.fast_fraction,
            noise_tau_ms=candidate.noise_tau_ms,
            cross_coupling=self.config.design.cross_coupling,
        )

    def csd(self, vector: np.ndarray) -> tuple[np.ndarray, FixedPoint]:
        spectral = self.config.spectral
        return linear_sensor_csd(
            self.parameters(vector),
            self.gain,
            frequency_min_hz=spectral.frequency_min_hz,
            frequency_max_hz=spectral.frequency_max_hz,
            bin_hz=spectral.frequency_bin_hz,
            epoch_seconds=spectral.epoch_seconds,
            reference=self.baseline.monitor.reference,
            monitor_period_ms=self.baseline.simulation.monitor_period_ms,
        )


def stability_report(
    params: LinearNetworkParameters,
    point: FixedPoint | None = None,
    frequency_max_hz: float = 150.0,
    frequency_step_hz: float = 0.02,
) -> dict[str, float]:
    """Check open-loop node stability and closed-loop delayed-network stability.

    Nodes: largest real part of the 12x12 local Jacobian eigenvalues.
    Network: argument principle for ``det(I - diag(h(iw)) K(iw))`` along the
    positive imaginary axis (the negative half is its conjugate).  With stable
    nodes the number of right-half-plane characteristic roots equals minus the
    winding number, so ``network_winding`` must be zero.
    """

    point = point or solve_fixed_point(params)
    jac = local_jacobians(params, point)
    if not np.isfinite(jac).all():
        return {"node_max_real_eigenvalue_per_s": float("inf"), "network_winding": -1,
                "det_phase_at_zero": float("nan"), "min_log_abs_det": float("nan"),
                "fixed_point_residual": point.residual, "stable": False}
    eigen = np.linalg.eigvals(jac)
    node_max = float(np.max(eigen.real)) * 1000.0  # 1/s
    frequency = np.arange(0.0, frequency_max_hz + frequency_step_hz, frequency_step_hz)
    phases: list[float] = []
    min_abs_log_det = np.inf
    n = params.a.size
    f = params.fast_fraction
    rate_slope = np.stack(
        ((1.0 - f) * sigmoid_slope(point.psi_alpha), f * sigmoid_slope(point.psi_fast)),
        axis=1,
    )
    input_gain = np.stack((A_CONST * params.a, A_CONST * params.a * params.fast_ratio), 1)
    for start in range(0, frequency.size, 256):
        chunk = frequency[start : start + 256]
        omega = 2.0 * np.pi * chunk / 1000.0
        response = _local_responses(params, jac, omega)
        long_range_response = np.einsum("fnoi,ni->fno", response, input_gain)
        h = np.einsum("no,fno->fn", rate_slope, long_range_response)
        kernel = params.kernel(omega)
        system = np.eye(n)[None] - h[:, :, None] * kernel
        sign, logabs = np.linalg.slogdet(system)
        phases.extend(np.angle(sign).tolist())
        min_abs_log_det = min(min_abs_log_det, float(np.min(logabs)))
    unwrapped = np.unwrap(np.asarray(phases))
    delta = float(unwrapped[-1] - unwrapped[0])
    # det(0) must be real and positive; the conjugate half doubles the winding.
    winding = int(np.rint(2.0 * delta / (2.0 * np.pi)))
    return {
        "node_max_real_eigenvalue_per_s": node_max,
        "network_winding": winding,
        "det_phase_at_zero": float(phases[0]),
        "min_log_abs_det": min_abs_log_det,
        "fixed_point_residual": point.residual,
        "stable": bool(node_max < 0.0 and winding == 0 and abs(phases[0]) < 1e-6),
    }
