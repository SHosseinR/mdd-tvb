"""Corticothalamic loop in the linear model (linear_jax.thalamic_loop)."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mdd_tvb import linear_jax as LJ  # noqa: E402


def test_loop_formula_matches_explicit_relay_reticular_solve():
    """theta(w) equals the cortical input obtained by solving the relay/reticular populations directly."""
    rng = np.random.default_rng(3)
    G_es, G_se, G_sr, G_re, G_rs = rng.uniform(0.5, 3.0, 5)
    t0 = 85.0
    omega = 2 * np.pi * np.array([0.5, 4.0, 10.0, 21.0, 37.0]) / 1000.0
    p = {"thal_gain": G_es * G_se, "thal_gamma": G_sr * G_re / G_se, "thal_t0": t0, "thal_kappa": G_sr * G_rs}
    theta = np.asarray(LJ.thalamic_loop(p, jnp.asarray(omega), 1))[:, 0]
    for w, th in zip(omega, theta):
        L = 1.0 / ((1 + 1j * w / LJ.THAL_ALPHA) * (1 + 1j * w / LJ.THAL_BETA))
        e = np.exp(-1j * w * t0 / 2)  # one-way corticothalamic delay; cortical rate = 1
        # unknowns (phi_s, phi_r):  phi_s = L (G_se e - G_sr phi_r),  phi_r = L (G_re e + G_rs phi_s)
        A = np.array([[1.0, L * G_sr], [-L * G_rs, 1.0]])
        phi_s, _ = np.linalg.solve(A, np.array([L * G_se * e, L * G_re * e]))
        assert np.isclose(G_es * e * phi_s, th, rtol=1e-10)


def test_unit_parameter_map_roundtrip():
    x = np.array([35.0, 0.7, 92.0, 0.4])
    back = np.asarray(LJ.thal_unit_to_physical(jnp.asarray(LJ.thal_physical_to_unit(x))))
    assert np.allclose(back, x, rtol=1e-6)


def _toy_state(n=6, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, (n, n))
    np.fill_diagonal(W, 0)
    W /= W.sum(1, keepdims=True) * 2
    return {"W": jnp.asarray(W), "a": jnp.full(n, 0.13), "b": jnp.full(n, 0.06), "mu": jnp.full(n, 0.22),
            "noise_base": jnp.ones(n), "G": 5.0, "fast_ratio": 2.6, "fast_fraction": 0.1, "noise_tau": 4.0,
            "cross": 0.12}, rng.uniform(5, 60, (n, n))


def test_negligible_loop_leaves_the_transfer_unchanged_and_loop_changes_it():
    p, delays = _toy_state()
    lead = jnp.asarray(np.random.default_rng(1).normal(size=(4, 6)))
    f = jnp.linspace(2.0, 40.0, 20)
    base, _ = LJ.network_sensor_transfer(p, lead, jnp.asarray(delays), f)
    weak, _ = LJ.network_sensor_transfer(LJ.with_thalamus(p, jnp.asarray([1e-6, 0.5, 80.0, 0.3])), lead,
                                         jnp.asarray(delays), f)
    strong, _ = LJ.network_sensor_transfer(LJ.with_thalamus(p, jnp.asarray([40.0, 0.5, 80.0, 0.3])), lead,
                                           jnp.asarray(delays), f)
    assert np.max(np.abs(np.asarray(weak - base))) < 1e-6 * np.max(np.abs(np.asarray(base)))
    assert np.max(np.abs(np.asarray(strong - base))) > 1e-2 * np.max(np.abs(np.asarray(base)))


def test_certificate_flags_an_unstable_own_loop():
    p, _ = _toy_state()
    psi = LJ.equilibrium(p)
    ok_weak, _ = LJ.thalamic_certificate(LJ.with_thalamus(p, jnp.asarray([5.0, 0.5, 80.0, 0.3])), psi)
    ok_huge, _ = LJ.thalamic_certificate(LJ.with_thalamus(p, jnp.asarray([1e5, 0.0, 80.0, 0.0])), psi)
    ok_off, dist_off = LJ.thalamic_certificate(p, psi)
    assert bool(ok_weak) and not bool(ok_huge) and bool(ok_off) and float(dist_off) == 1.0
