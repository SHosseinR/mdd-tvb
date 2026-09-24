"""Tests for the analytic linear-regime spectrum (NumPy reference and JAX)."""

from __future__ import annotations

import numpy as np
import pytest

from mdd_tvb.linear_spectral import (
    LinearNetworkParameters,
    linear_regional_transfer,
    sensor_csd_from_transfer,
    solve_fixed_point,
    stability_report,
)

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mdd_tvb import linear_jax as LJ  # noqa: E402


def _small_network(n: int = 6, channels: int = 5, seed: int = 3):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0, 1.0, (n, n))
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)
    weights /= max(weights.sum(1).max(), 1e-12)
    lengths = rng.uniform(20.0, 120.0, (n, n))
    lengths = 0.5 * (lengths + lengths.T)
    np.fill_diagonal(lengths, 0.0)
    gain = rng.normal(size=(channels, n))
    params = LinearNetworkParameters(
        weights=weights,
        delays_ms=lengths / 6.0,
        a=np.full(n, 0.13 * 1.1) * rng.uniform(0.95, 1.05, n),
        b=np.full(n, 0.065 * 1.1) * rng.uniform(0.95, 1.05, n),
        mu=np.full(n, 0.28) * rng.uniform(0.95, 1.05, n),
        noise_nsig=rng.uniform(0.5, 1.5, n) * 1e-3,
        global_coupling=3.0,
        fast_ratio=2.2,
        fast_fraction=0.25,
        noise_tau_ms=1.0,
    )
    return params, gain, lengths


def _static(params, gain, lengths, fine):
    n = params.a.size
    return LJ.StaticNetwork(
        weights=params.weights,
        lengths_mm=lengths,
        gain=gain,
        network_index=np.arange(n) % 2,
        network_names=("A", "B"),
        drive_multiplier=np.ones(n),
        time_multiplier=np.ones(n),
        noise_multiplier=params.noise_nsig,
        fine_frequency_hz=fine,
        bin_groups=tuple(np.asarray([i]) for i in range(fine.size)),
        monitor_period_ms=0.0,
    )


def _jax_state(params):
    return {
        "W": jnp.asarray(params.weights), "a": jnp.asarray(params.a), "b": jnp.asarray(params.b),
        "mu": jnp.asarray(params.mu), "noise_base": jnp.asarray(params.noise_nsig),
        "G": params.global_coupling, "fast_ratio": params.fast_ratio,
        "fast_fraction": params.fast_fraction, "noise_tau": params.noise_tau_ms,
        "cross": params.cross_coupling,
    }


def test_fixed_point_is_accurate_and_stable():
    params, _, _ = _small_network()
    point = solve_fixed_point(params, initial=np.full(2 * params.a.size, 7.0))
    assert point.residual < 1e-10
    report = stability_report(params, point, frequency_max_hz=80.0, frequency_step_hz=0.05)
    assert report["node_max_real_eigenvalue_per_s"] < 0.0
    assert report["network_winding"] == 0


def test_jax_matches_numpy_reference():
    params, gain, lengths = _small_network()
    fine = np.arange(2.0, 30.0, 0.5)
    transfer, noise, _ = linear_regional_transfer(params, fine)
    reference = sensor_csd_from_transfer(gain, transfer, noise, reference="average")
    static = _static(params, gain, lengths, fine)
    contributions, psi = LJ.network_contributions(
        _jax_state(params), static, LJ.average_reference(gain), jnp.asarray(params.delays_ms)
    )
    got = np.asarray(contributions.sum(0))
    np.testing.assert_allclose(got, reference * LJ.UNITS, rtol=1e-8, atol=1e-14 * np.abs(reference).max())
    assert float(jnp.max(jnp.abs(LJ._residual(psi, _jax_state(params))))) < 1e-10


def test_jax_gradient_matches_finite_difference():
    params, gain, lengths = _small_network()
    fine = np.arange(4.0, 20.0, 1.0)
    static = _static(params, gain, lengths, fine)
    lead = LJ.average_reference(gain)
    delays = jnp.asarray(params.delays_ms)

    def loss(g, f):
        state = _jax_state(params)
        state["G"] = g
        state["fast_fraction"] = f
        c, _ = LJ.network_contributions(state, static, lead, delays)
        return jnp.log(jnp.real(jnp.einsum("kfii->", c)))

    grad = jax.grad(loss, argnums=(0, 1))(3.0, 0.25)
    eps = 1e-6
    numeric_g = (loss(3.0 + eps, 0.25) - loss(3.0 - eps, 0.25)) / (2 * eps)
    numeric_f = (loss(3.0, 0.25 + eps) - loss(3.0, 0.25 - eps)) / (2 * eps)
    assert float(grad[0]) == pytest.approx(float(numeric_g), rel=1e-5)
    assert float(grad[1]) == pytest.approx(float(numeric_f), rel=1e-5)


def test_fold_margin_shrinks_towards_the_jansen_rit_fold():
    """Tracking the upper branch towards lower drive approaches a saddle-node."""

    params, _, _ = _small_network(n=1, channels=2)
    margins = []
    for mu in (0.30, 0.22, 0.18, 0.17):
        state = dict(
            _jax_state(params), W=jnp.zeros((1, 1)), mu=jnp.asarray([mu]),
            a=jnp.asarray([0.13 * 1.175]), b=jnp.asarray([0.065 * 0.996]),
        )
        psi = LJ._solve_equilibrium(state)
        assert float(jnp.max(jnp.abs(LJ._residual(psi, state)))) < 1e-8
        margins.append(float(LJ.fold_margin(state, psi)[0]))
    assert margins[0] > margins[1] > margins[2] > margins[3]
    assert margins[3] < 0.3 * margins[0]


def test_analytic_spectrum_matches_small_noise_simulation():
    """The closed-form CSD is the small-noise limit of the stochastic simulator."""

    from mdd_tvb.jax_backend import JaxDualBatch, run_dual_jansen_rit_jax
    from mdd_tvb.spectral_config import CrossSpectralConfig
    from mdd_tvb.spectral_features import estimate_cross_spectrum
    from mdd_tvb.linear_spectral import linear_sensor_csd

    params, gain, lengths = _small_network(n=4, channels=4)
    delays = np.rint(params.delays_ms / 0.5) * 0.5
    params = LinearNetworkParameters(**{**params.__dict__, "delays_ms": delays})
    batch = JaxDualBatch(
        weights=params.weights, tract_lengths_ms_at_unit_speed=lengths,
        gain_matrix=gain, regional_a=params.a[None], regional_b=params.b[None],
        regional_mu=params.mu[None], regional_noise_nsig=params.noise_nsig[None] * 1e-4,
        global_coupling=np.asarray([params.global_coupling]), speed_mm_per_ms=np.asarray([6.0]),
        fast_ratio=np.asarray([params.fast_ratio]), fast_fraction=np.asarray([params.fast_fraction]),
        noise_tau_ms=np.asarray([params.noise_tau_ms]), seeds=np.asarray([1], dtype=np.int32),
    )
    result = run_dual_jansen_rit_jax(
        batch, duration_ms=42000.0, transient_ms=2000.0, dt_ms=0.5, monitor_period_ms=2.0,
        precision="float64",
    )
    settings = CrossSpectralConfig(frequency_min_hz=3.0, frequency_max_hz=30.0)
    _, simulated, _ = estimate_cross_spectrum(result.eeg[0], 500.0, settings)
    analytic, _ = linear_sensor_csd(params, gain, frequency_min_hz=3.0, frequency_max_hz=30.0)
    sim_power = np.log(np.real(np.einsum("fii->f", simulated)))
    lin_power = np.log(np.real(np.einsum("fii->f", analytic)))
    r = np.corrcoef(sim_power - sim_power.mean(), lin_power - lin_power.mean())[0, 1]
    assert r > 0.98
