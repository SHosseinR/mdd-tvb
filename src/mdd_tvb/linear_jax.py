"""Differentiable JAX version of the linearised dual Jansen--Rit spectrum.

``linear_spectral`` is the readable NumPy reference.  This module computes the
same expected sensor cross spectra with

* closed-form 2x2 node transfer functions (instead of 12x12 solves),
* an equilibrium found by relaxation from the zero state followed by Newton
  polishing, differentiated implicitly with ``jax.lax.custom_root``, and
* per-network source contributions, so that network-level input-noise gains
  (the spatial parameters) enter linearly and are cheap to optimise.

All times are in milliseconds and all inverse time constants in 1/ms, exactly
as in the TVB/JAX simulators.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import os as _os

# Float64 by default (reference accuracy on CPU); set MDD_TVB_JAX_X64=0 for
# float32/complex64 on GPUs with weak double-precision throughput.
jax.config.update("jax_enable_x64", _os.environ.get("MDD_TVB_JAX_X64", "1") != "0")

A_CONST = 3.25
B_CONST = 22.0
J_CONST = 135.0
V0 = 5.52
NU_MAX = 0.0025
SLOPE = 0.56
A1, A2, A3, A4 = 1.0, 0.8, 0.25, 0.25
FAST_DRIVE_RATIO = 0.85
# One-sided per-Hz PSD of a process whose two-sided per-(rad/ms) PSD is S:
# S_Hz = 2 * S * 1e-3 (time in ms).  Only needed for absolute-level checks.
UNITS = 2.0e-3


def _sigmoid(v):
    return 2.0 * NU_MAX / (1.0 + jnp.exp(jnp.clip(SLOPE * (V0 - v), -60.0, 60.0)))


def _sigmoid_slope(v):
    rate = _sigmoid(v)
    return SLOPE * rate * (1.0 - rate / (2.0 * NU_MAX))


@dataclass(frozen=True)
class StaticNetwork:
    """Fixed anatomy/observation arrays shared by every parameter state."""

    weights: np.ndarray  # (n, n) target x source
    lengths_mm: np.ndarray  # (n, n)
    gain: np.ndarray  # (channels, n), NOT yet referenced
    network_index: np.ndarray  # (n,) int in [0, n_networks)
    network_names: tuple[str, ...]
    drive_multiplier: np.ndarray  # seeded heterogeneity (n,)
    time_multiplier: np.ndarray  # (n,)
    noise_multiplier: np.ndarray  # (n,)
    fine_frequency_hz: np.ndarray  # FFT frequencies used by the Welch estimator
    bin_groups: tuple[np.ndarray, ...]
    dt_ms: float = 0.5
    monitor_period_ms: float = 2.0
    cross_coupling: float = 0.12
    # Source groups for separately scalable input-noise contributions
    # (default: network x hemisphere).  ``None`` falls back to networks.
    group_index: np.ndarray | None = None
    group_names: tuple[str, ...] | None = None
    centres_mm: np.ndarray | None = None

    @property
    def contribution_index(self) -> np.ndarray:
        return self.network_index if self.group_index is None else self.group_index

    @property
    def n_groups(self) -> int:
        return self.n_networks if self.group_names is None else len(self.group_names)

    @property
    def n_regions(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_networks(self) -> int:
        return len(self.network_names)


def _psi_map(psi, p):
    n = p["a"].shape[0]
    pa, pf = psi[:n], psi[n:]
    a, b = p["a"], p["b"]
    af, bf = a * p["fast_ratio"], b * p["fast_ratio"]
    f = p["fast_fraction"]
    sa, sf = _sigmoid(pa), _sigmoid(pf)
    y0 = A_CONST / a * sa
    y6 = A_CONST / af * sf
    lr = p["G"] * (p["W"] @ ((1.0 - f) * sa + f * sf))
    cross = p["cross"] * J_CONST
    new_a = A_CONST / a * (
        p["mu"] + A2 * J_CONST * _sigmoid(A1 * J_CONST * y0) + lr
        + cross * _sigmoid(A1 * J_CONST * y6)
    ) - B_CONST / b * A4 * J_CONST * _sigmoid(A3 * J_CONST * y0)
    new_f = A_CONST / af * (
        p["mu"] * FAST_DRIVE_RATIO + A2 * J_CONST * _sigmoid(A1 * J_CONST * y6) + lr
        + cross * _sigmoid(A1 * J_CONST * y0)
    ) - B_CONST / bf * A4 * J_CONST * _sigmoid(A3 * J_CONST * y6)
    return jnp.concatenate((new_a, new_f))


def _residual(psi, p):
    return psi - _psi_map(psi, p)


def _solve_equilibrium(p, relax_steps: int = 400, newton_steps: int = 12):
    n = p["a"].shape[0]

    def relax(_, psi):
        return psi + 0.15 * (_psi_map(psi, p) - psi)

    # Start on the upper (alpha-generating) branch; in the monostable regime
    # enforced by ``fold_margin`` the start point is irrelevant.
    start = jnp.concatenate((jnp.full(n, 7.0), jnp.full(n, 1.5)))
    psi = jax.lax.fori_loop(0, relax_steps, relax, start)

    def newton(_, psi):
        jac = jax.jacfwd(_residual)(psi, p)
        step = jnp.linalg.solve(jac, _residual(psi, p))
        return psi - jnp.clip(step, -2.0, 2.0)

    return jax.lax.fori_loop(0, newton_steps, newton, psi)


def equilibrium(p):
    """Equilibrium PSPs with implicit differentiation."""

    def solve(f, guess):
        del f, guess
        return jax.lax.stop_gradient(_solve_equilibrium(jax.lax.stop_gradient(p)))

    def tangent_solve(g, y):
        jac = jax.jacfwd(g)(y)
        return jnp.linalg.solve(jac, y)

    n = p["a"].shape[0]
    return jax.lax.custom_root(
        lambda psi: _residual(psi, p), jnp.zeros(2 * n), solve, tangent_solve
    )


def node_responses(p, psi, s):
    """Closed-form linear responses of both generators.

    Returns (R, bu, bn): R has shape (freq, n, 2, 2) mapping the per-generator
    right-hand side to (psi_alpha, psi_fast).
    """

    n = p["a"].shape[0]
    pa, pf = psi[:n], psi[n:]
    a, b = p["a"], p["b"]
    af, bf = a * p["fast_ratio"], b * p["fast_ratio"]
    y0 = A_CONST / a * _sigmoid(pa)
    y6 = A_CONST / af * _sigmoid(pf)
    so, sof = _sigmoid_slope(pa), _sigmoid_slope(pf)
    ce = A2 * J_CONST * _sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST
    cef = A2 * J_CONST * _sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST
    ci = A4 * J_CONST * _sigmoid_slope(A3 * J_CONST * y0) * A3 * J_CONST
    cif = A4 * J_CONST * _sigmoid_slope(A3 * J_CONST * y6) * A3 * J_CONST
    cross = p["cross"] * J_CONST
    cxf = cross * _sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST
    cxa = cross * _sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST
    s = s[:, None]
    he = A_CONST * a / (s + a) ** 2
    hi = B_CONST * b / (s + b) ** 2
    hef = A_CONST * af / (s + af) ** 2
    hif = B_CONST * bf / (s + bf) ** 2
    m11 = 1.0 - he * he * ce * so + hi * ci * he * so
    m12 = -he * cxf * hef * sof
    m21 = -hef * cxa * he * so
    m22 = 1.0 - hef * hef * cef * sof + hif * cif * hef * sof
    det = m11 * m22 - m12 * m21
    r = jnp.stack(
        (jnp.stack((m22, -m12), -1), jnp.stack((-m21, m11), -1)), -2
    ) / det[..., None, None]
    bu = jnp.stack((he, hef), -1)  # long-range input (enters with A*a factors)
    bn = jnp.stack((he / (A_CONST * a), hef / (A_CONST * af)), -1)  # noise input
    return r, bu, bn


def network_sensor_transfer(p, lead, delays_ms, fine_hz):
    """Sensor transfer from every (region, generator) noise input.

    Returns T of shape (freq, channels, n, 2) and the equilibrium.
    """

    psi = equilibrium(p)
    n = p["a"].shape[0]
    omega = 2.0 * jnp.pi * fine_hz / 1000.0
    s = 1j * omega
    r, bu, bn = node_responses(p, psi, s)
    f = p["fast_fraction"]
    rate_slope = jnp.stack(
        ((1.0 - f) * _sigmoid_slope(psi[:n]), f * _sigmoid_slope(psi[n:])), -1
    )  # (n, 2)
    observable = jnp.stack((1.0 - f, f))
    resp_u = jnp.einsum("fnij,fnj->fni", r, bu)  # (F, n, 2)
    resp_n = r * bn[:, :, None, :]  # (F, n, 2 out, 2 in)
    h = jnp.einsum("ni,fni->fn", rate_slope, resp_u)
    q = jnp.einsum("i,fni->fn", observable, resp_u)
    g = jnp.einsum("ni,fnij->fnj", rate_slope, resp_n)  # (F, n, 2)
    pn = jnp.einsum("i,fnij->fnj", observable, resp_n)
    kernel = p["G"] * p["W"][None] * jnp.exp(-1j * omega[:, None, None] * delays_ms[None])
    system = jnp.eye(n)[None] - h[:, :, None] * kernel
    y = lead[None] * q[:, None, :]  # (F, C, n): L diag(q)
    y = jnp.einsum("fcn,fnm->fcm", y, kernel)  # L diag(q) K
    # X = Y (I - diag(h) K)^-1  <=>  (I - diag(h)K)^T X^T = Y^T
    x = jnp.swapaxes(
        jnp.linalg.solve(jnp.swapaxes(system, -1, -2), jnp.swapaxes(y, -1, -2)), -1, -2
    )
    transfer = x[..., None] * g[:, None, :, :] + lead[None, :, :, None] * pn[:, None, :, :]
    return transfer, psi


def _noise_weight(p, static: StaticNetwork, fine):
    omega_tau = 2.0 * jnp.pi * fine / 1000.0 * p["noise_tau"]
    shape = 4.0 * UNITS / (1.0 + omega_tau**2)
    return shape * jnp.sinc(fine * static.monitor_period_ms / 1000.0) ** 2


def _bin(csd, static: StaticNetwork, axis: int = 0):
    binned = jnp.stack(
        [jnp.take(csd, jnp.asarray(g_), axis=axis).mean(axis=axis) for g_ in static.bin_groups], axis
    )
    return 0.5 * (binned + jnp.conjugate(jnp.swapaxes(binned, -1, -2)))


def _group_contributions(transfer, p, static: StaticNetwork, fine):
    weight = p["noise_base"][None, :] * _noise_weight(p, static, fine)[:, None]  # (F, n)
    weighted = transfer * jnp.sqrt(weight)[:, None, :, None]
    contrib = jnp.stack(
        [
            jnp.einsum(
                "fcnk,fdnk->fcd",
                jnp.conjugate(weighted[:, :, members]),
                weighted[:, :, members],
            )
            for members in (
                np.flatnonzero(static.contribution_index == k)
                for k in range(static.n_groups)
            )
        ]
    )
    return _bin(contrib, static, axis=1)


def _common_drive(transfer, p, static: StaticNetwork, fine, origin_mm, speed_mm_per_ms):
    omega = 2.0 * jnp.pi * fine / 1000.0
    delay = jnp.linalg.norm(
        jnp.asarray(static.centres_mm) - jnp.asarray(origin_mm), axis=1
    ) / speed_mm_per_ms
    vector = jnp.einsum("fcn,fn->fc", transfer[..., 0], jnp.exp(-1j * omega[:, None] * delay[None]))
    csd = jnp.einsum(
        "fc,fd,f->fcd", jnp.conjugate(vector), vector, _noise_weight(p, static, fine)
    )
    return _bin(csd, static, axis=0)


def network_contributions(p, static: StaticNetwork, lead, delays_ms):
    """Per-group binned CSD contributions with unit group noise gain."""

    fine = jnp.asarray(static.fine_frequency_hz)
    transfer, psi = network_sensor_transfer(p, lead, delays_ms, fine)
    return _group_contributions(transfer, p, static, fine), psi


def contributions_with_common_drive(p, static: StaticNetwork, lead, delays_ms, origin_mm, speed_mm_per_ms):
    """Group contributions plus the rank-one shared delayed-drive CSD.

    The shared drive is one noise source reaching each region's alpha
    excitatory input with delay ``|centre - origin| / speed`` (a
    travelling-wave-like common input), unit variance, same colour.
    """

    fine = jnp.asarray(static.fine_frequency_hz)
    transfer, psi = network_sensor_transfer(p, lead, delays_ms, fine)
    return (
        _group_contributions(transfer, p, static, fine),
        _common_drive(transfer, p, static, fine, origin_mm, speed_mm_per_ms),
        psi,
    )


# ---------------------------------------------------------------------------
# Parameterisation
# ---------------------------------------------------------------------------

GLOBAL_NAMES = (
    "global_coupling",
    "speed_mm_per_ms",
    "mu",
    "a_scale",
    "b_scale",
    "fast_ratio",
    "fast_fraction",
    "noise_tau_ms",
    "vis_time_contrast",
    "dorsattn_time_contrast",
)
GLOBAL_BOUNDS = np.asarray(
    [
        [0.5, 40.0],
        [2.0, 12.0],
        [0.10, 0.40],
        [0.60, 1.60],
        [0.60, 1.80],
        [1.3, 3.5],
        [0.02, 0.70],
        [0.25, 8.0],
        [-0.30, 0.30],
        [-0.30, 0.30],
    ]
)
LOG_SCALED = {0, 7}


def unit_to_physical(u):
    """Map an unconstrained vector to bounded physical global parameters."""

    z = jax.nn.sigmoid(u)
    lo = jnp.asarray(GLOBAL_BOUNDS[:, 0])
    hi = jnp.asarray(GLOBAL_BOUNDS[:, 1])
    mask = jnp.asarray([i in LOG_SCALED for i in range(len(GLOBAL_NAMES))])
    lin = lo + z * (hi - lo)
    # Evaluate the log branch only on positive bounds: an unselected NaN branch
    # would still poison the gradient through jnp.where.
    lo_safe = jnp.where(mask, lo, 1.0)
    hi_safe = jnp.where(mask, hi, 2.0)
    log = jnp.exp(jnp.log(lo_safe) + z * (jnp.log(hi_safe) - jnp.log(lo_safe)))
    return jnp.where(mask, log, lin)


def physical_to_unit(x):
    x = np.asarray(x, dtype=float)
    lo, hi = GLOBAL_BOUNDS[:, 0], GLOBAL_BOUNDS[:, 1]
    z = np.empty_like(x)
    for i in range(len(GLOBAL_NAMES)):
        if i in LOG_SCALED:
            z[..., i] = (np.log(x[..., i]) - np.log(lo[i])) / (np.log(hi[i]) - np.log(lo[i]))
        else:
            z[..., i] = (x[..., i] - lo[i]) / (hi[i] - lo[i])
    z = np.clip(z, 1e-6, 1 - 1e-6)
    return np.log(z / (1 - z))


def build_state(physical, static: StaticNetwork):
    """Regional arrays for one global physical parameter vector."""

    (G, speed, mu, a_scale, b_scale, fast_ratio, fast_fraction, noise_tau,
     vis_tc, dan_tc) = [physical[i] for i in range(len(GLOBAL_NAMES))]
    names = static.network_names
    net = jnp.asarray(static.network_index)
    contrast = jnp.zeros(static.n_networks)
    contrast = contrast.at[names.index("Vis")].set(vis_tc)
    contrast = contrast.at[names.index("DorsAttn")].set(dan_tc)
    mult = 1.0 + contrast
    region_mult = mult[net]
    region_mult = region_mult / jnp.mean(region_mult)
    time = jnp.asarray(static.time_multiplier) * region_mult
    return {
        "W": jnp.asarray(static.weights),
        "a": 0.13 * a_scale * time,
        "b": 0.065 * b_scale * time,
        "mu": mu * jnp.asarray(static.drive_multiplier),
        "noise_base": jnp.asarray(static.noise_multiplier),
        "G": G,
        "fast_ratio": fast_ratio,
        "fast_fraction": fast_fraction,
        "noise_tau": noise_tau,
        "cross": static.cross_coupling,
    }, speed


def delays_for_speed(static: StaticNetwork, speed):
    # Continuous delays keep the spectrum differentiable in conduction speed.
    return jnp.asarray(static.lengths_mm) / speed


def average_reference(gain):
    gain = jnp.asarray(gain)
    return gain - gain.mean(axis=0, keepdims=True)


def combine(contributions, log_gains, obs_fraction, obs_exponent, frequency_hz):
    """Sensor CSD = sum_k exp(log_gain_k) C_k + diagonal coloured sensor noise."""

    gains = jnp.exp(log_gains)
    csd = jnp.einsum("k,kfcd->fcd", gains, contributions)
    diag = jnp.real(jnp.diagonal(csd, axis1=1, axis2=2))
    mean_signal = jnp.mean(diag)
    shape = frequency_hz ** (-obs_exponent)
    shape = shape / jnp.mean(shape)
    ratio = obs_fraction / (1.0 - obs_fraction)
    noise = mean_signal * ratio * shape
    return csd + noise[:, None, None] * jnp.eye(csd.shape[-1])[None]


def static_from_linearizer(linearizer, config) -> StaticNetwork:
    """Build StaticNetwork using the repository's own seeded heterogeneity."""

    from dataclasses import replace
    from .heterogeneity import build_regional_parameters
    from .linear_spectral import welch_expectation_grid

    baseline = linearizer.baseline
    regional = build_regional_parameters(
        replace(baseline.model, mu=1.0, a=1.0, b=1.0),
        1.0,
        linearizer.connectome.region_labels,
        baseline.heterogeneity,
    )
    names = tuple(sorted(np.unique(linearizer.networks).tolist()))
    index = np.asarray([names.index(x) for x in linearizer.networks])
    labels = np.asarray(linearizer.connectome.region_labels).astype(str)
    right = np.asarray(["_RH_" in label for label in labels], dtype=int)
    group_index = index * 2 + right
    group_names = tuple(f"{name}_{hemi}" for name in names for hemi in ("LH", "RH"))
    spectral = config.spectral
    fine, groups = welch_expectation_grid(
        spectral.frequency_min_hz,
        spectral.frequency_max_hz,
        spectral.frequency_bin_hz,
        spectral.epoch_seconds,
    )
    return StaticNetwork(
        weights=np.asarray(linearizer.connectome.weights, float),
        lengths_mm=np.asarray(linearizer.connectome.tract_lengths, float),
        gain=np.asarray(linearizer.gain, float),
        network_index=index,
        network_names=names,
        drive_multiplier=np.asarray(regional.drive_multiplier, float),
        time_multiplier=np.asarray(regional.time_scale_multiplier, float),
        noise_multiplier=np.asarray(regional.noise_multiplier, float),
        fine_frequency_hz=np.asarray(fine, float),
        bin_groups=tuple(np.asarray(g) for g in groups),
        dt_ms=config.design.dt_ms,
        monitor_period_ms=baseline.simulation.monitor_period_ms,
        cross_coupling=config.design.cross_coupling,
        group_index=group_index,
        group_names=group_names,
        centres_mm=np.asarray(linearizer.connectome.centres, float),
    )


def local_jacobians(p, psi):
    """Per-region 12x12 Jacobians (JAX mirror of linear_spectral.local_jacobians)."""

    n = p["a"].shape[0]
    pa, pf = psi[:n], psi[n:]
    a, b = p["a"], p["b"]
    af, bf = a * p["fast_ratio"], b * p["fast_ratio"]
    y0 = A_CONST / a * _sigmoid(pa)
    y6 = A_CONST / af * _sigmoid(pf)
    cross = p["cross"] * J_CONST
    zeros = jnp.zeros((n, 12, 12))
    jac = zeros
    blocks = (
        (0, a, b, _sigmoid_slope(pa), _sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST,
         _sigmoid_slope(A3 * J_CONST * y0) * A3 * J_CONST,
         _sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST, 6),
        (6, af, bf, _sigmoid_slope(pf), _sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST,
         _sigmoid_slope(A3 * J_CONST * y6) * A3 * J_CONST,
         _sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST, 0),
    )
    for off, rate, inh, out_s, exc_s, inh_s, other_exc, other in blocks:
        y0i, y1i, y2i, y3i, y4i, y5i = (off + k for k in range(6))
        jac = jac.at[:, y0i, y3i].set(1.0)
        jac = jac.at[:, y1i, y4i].set(1.0)
        jac = jac.at[:, y2i, y5i].set(1.0)
        jac = jac.at[:, y3i, y1i].set(A_CONST * rate * out_s)
        jac = jac.at[:, y3i, y2i].set(-A_CONST * rate * out_s)
        jac = jac.at[:, y3i, y3i].set(-2.0 * rate)
        jac = jac.at[:, y3i, y0i].set(-rate**2)
        jac = jac.at[:, y4i, y0i].set(A_CONST * rate * A2 * J_CONST * exc_s)
        jac = jac.at[:, y4i, other].set(A_CONST * rate * cross * other_exc)
        jac = jac.at[:, y4i, y4i].set(-2.0 * rate)
        jac = jac.at[:, y4i, y1i].set(-rate**2)
        jac = jac.at[:, y5i, y0i].set(B_CONST * inh * A4 * J_CONST * inh_s)
        jac = jac.at[:, y5i, y5i].set(-2.0 * inh)
        jac = jac.at[:, y5i, y2i].set(-inh**2)
    return jac


def node_abscissa_per_s(p, psi):
    """Largest real part of each region's local eigenvalues, in 1/s."""

    jac = local_jacobians(p, psi)
    finite = jnp.all(jnp.isfinite(jac), axis=(-2, -1))
    # LAPACK eig can hang on NaN/inf input (diverged equilibria): sanitise and
    # report such regions as unstable (+inf abscissa) instead.
    jac = jnp.where(finite[:, None, None], jac, 0.0)
    # Eigenvalue derivatives are undefined for repeated roots, which the
    # second-order synaptic kernels produce.  Freeze the eigenvectors and use
    # the first-order (Rayleigh-type) form: exact value, finite gradient.
    _, vectors = jnp.linalg.eig(jax.lax.stop_gradient(jac))
    vectors = jax.lax.stop_gradient(vectors)
    left = jax.lax.stop_gradient(jnp.linalg.inv(vectors))
    eig = jnp.einsum("nij,njk,nki->ni", left, jac.astype(vectors.dtype), vectors)
    return jnp.where(finite, jnp.max(jnp.real(eig), axis=-1) * 1000.0, jnp.inf)


def small_gain(p, psi, fine_hz):
    """max_w max_i |h_i(iw)| * G * ||W||_2 on the fitted grid (sufficient if <1)."""

    n = p["a"].shape[0]
    s = 2j * jnp.pi * fine_hz / 1000.0
    r, bu, _ = node_responses(p, psi, s)
    f = p["fast_fraction"]
    rate_slope = jnp.stack(
        ((1.0 - f) * _sigmoid_slope(psi[:n]), f * _sigmoid_slope(psi[n:])), -1
    )
    h = jnp.einsum("ni,fnij,fnj->fn", rate_slope, r, bu)
    norm_w = jnp.linalg.norm(p["W"], ord=2)
    return jnp.max(jnp.abs(h)) * p["G"] * norm_w


def fold_margin(p, psi):
    """Per-region det(I - dPhi_local/dpsi) of the uncoupled 2x2 equilibrium map.

    Values near zero mean the node is close to a saddle-node fold (the
    Jansen--Rit bistable window): the equilibrium becomes ill-conditioned and
    small noise can switch branches, so the linear description fails.
    """

    n = p["a"].shape[0]
    pa, pf = psi[:n], psi[n:]
    a, b = p["a"], p["b"]
    af, bf = a * p["fast_ratio"], b * p["fast_ratio"]
    dy0 = A_CONST / a * _sigmoid_slope(pa)
    dy6 = A_CONST / af * _sigmoid_slope(pf)
    y0 = A_CONST / a * _sigmoid(pa)
    y6 = A_CONST / af * _sigmoid(pf)
    d_exc_a = _sigmoid_slope(A1 * J_CONST * y0) * A1 * J_CONST * dy0
    d_exc_f = _sigmoid_slope(A1 * J_CONST * y6) * A1 * J_CONST * dy6
    d_inh_a = _sigmoid_slope(A3 * J_CONST * y0) * A3 * J_CONST * dy0
    d_inh_f = _sigmoid_slope(A3 * J_CONST * y6) * A3 * J_CONST * dy6
    cross = p["cross"] * J_CONST
    j11 = A_CONST / a * A2 * J_CONST * d_exc_a - B_CONST / b * A4 * J_CONST * d_inh_a
    j12 = A_CONST / a * cross * d_exc_f
    j21 = A_CONST / af * cross * d_exc_a
    j22 = A_CONST / af * A2 * J_CONST * d_exc_f - B_CONST / bf * A4 * J_CONST * d_inh_f
    return (1.0 - j11) * (1.0 - j22) - j12 * j21
