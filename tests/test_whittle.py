"""Tests for the complex-Wishart (Whittle) likelihood and the v2 epoch selection."""
import os

os.environ.setdefault("MDD_TVB_JAX_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mdd_tvb import whittle as W
from mdd_tvb.preprocess_v2 import clean_epoch_starts, repair_matrix, split_halves, EEG_LABELS

LABELS = list(EEG_LABELS)
FREQ = np.arange(2.0, 41.0)


def _random_csd(rng, F=len(FREQ), n=26):
    X = rng.normal(size=(F, n, 40)) + 1j * rng.normal(size=(F, n, 40))
    S = np.einsum("fik,fjk->fij", X, X.conj()) / 40
    R = np.eye(n) - 1.0 / n
    return np.einsum("ij,fjk,lk->fil", R, S, R)


def test_deviance_zero_at_truth_and_positive_elsewhere():
    rng = np.random.default_rng(0)
    S = _random_csd(rng)
    D, A, pad = W.observation_operators(LABELS, FREQ)
    Cc, scale = W.project_data(S, D, pad)
    assert W.deviance(S, -np.log(scale), A, pad, Cc, 50.0) == pytest.approx(0.0, abs=1e-6)
    assert W.deviance(_random_csd(rng), -np.log(scale), A, pad, Cc, 50.0) > 1.0


def test_profiled_scale_recovers_the_scale():
    rng = np.random.default_rng(1)
    S = _random_csd(rng)
    D, A, pad = W.observation_operators(LABELS, FREQ)
    Cc, scale = W.project_data(S * 7.3, D, pad)
    nll, logs = W.profiled_nll(jnp.asarray(S), jnp.asarray(A), jnp.asarray(pad), jnp.asarray(Cc), 50.0)
    # data were divided by ``scale``: the model needs 7.3 / scale
    assert float(logs) == pytest.approx(np.log(7.3 / scale), rel=1e-8)
    direct = W.nll_at_scale(jnp.asarray(S), logs, jnp.asarray(A), jnp.asarray(pad), jnp.asarray(Cc), 50.0)
    npad = np.sum(pad)
    assert float(nll) == pytest.approx(float(direct) - 50.0 * npad, rel=1e-8)


def test_marginalising_channels_equals_submatrix_likelihood():
    rng = np.random.default_rng(2)
    S = _random_csd(rng)
    C = _random_csd(rng)
    emg = ("Fp1", "Fp2", "F7", "F8", "T7", "T8")
    D, A, pad = W.observation_operators(LABELS, FREQ, fixed_emg=emg)
    hi = FREQ >= W.EMG_SPLIT_HZ
    keep = [LABELS.index(c) for c in LABELS if c not in emg]
    f = int(np.flatnonzero(hi)[0])
    Sc = A[f] @ S[f] @ A[f].T
    Cc = D[f] @ C[f] @ D[f].T
    r = int(np.sum(~pad[f]))
    assert r == len(keep)  # 20 channels of an average-referenced set are linearly independent
    Ssub, Csub = S[f][np.ix_(keep, keep)], C[f][np.ix_(keep, keep)]
    ll_obs = np.linalg.slogdet(Sc[:r, :r])[1] + np.trace(np.linalg.solve(Sc[:r, :r], Cc[:r, :r])).real
    ll_sub = np.linalg.slogdet(Ssub)[1] + np.trace(np.linalg.solve(Ssub, Csub)).real
    # orthonormal change of basis leaves the trace term unchanged; log det differs by a constant (0 here)
    assert ll_obs == pytest.approx(ll_sub, rel=1e-8)


def test_repaired_channels_reduce_the_observed_rank():
    P = repair_matrix(LABELS, ["T7", "Cz"])
    D, A, pad = W.observation_operators(LABELS, FREQ, repair=P, fixed_emg=())
    assert int(np.sum(~pad[0])) == 23


def test_welch_degrees_of_freedom_calibration():
    from mdd_tvb.spectral_config import CrossSpectralConfig
    from mdd_tvb.spectral_features import estimate_cross_spectrum

    settings = CrossSpectralConfig()
    rng = np.random.default_rng(3)
    values = []
    for _ in range(120):
        _, C, K = estimate_cross_spectrum(rng.normal(size=(30000, 4)), 500.0, settings)
        values.append(np.real(C[:, 0, 0]))
    v = np.asarray(values)
    nu = np.mean(v.mean(0) ** 2 / v.var(0))
    assert nu / K == pytest.approx(W.DOF_PER_EPOCH, rel=0.12)


def test_clean_epochs_stay_inside_clean_spans_and_halves_do_not_overlap():
    clean = np.ones(30000, bool)
    clean[5000:5100] = False
    clean[17000:17500] = False
    starts = clean_epoch_starts(clean, 2000, 1000)
    for s in starts:
        assert clean[s:s + 2000].all()
    first, second = split_halves(starts, 2000)
    assert first.max() + 2000 <= second.min()
