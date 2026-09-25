"""Complex-Wishart (Whittle) likelihood for binned EEG cross-spectra.

A 1-Hz-binned Welch CSD of an average-referenced recording is approximately
complex-Wishart distributed around the true CSD with ``nu = DOF_PER_EPOCH x
epochs`` degrees of freedom (calibrated for 4-s Hann epochs, 50 % overlap,
four 0.25-Hz FFT bins per 1-Hz bin; see tests).  The negative log likelihood,
up to a constant, is

    NLL(S) = sum_f nu [ log det S_f + tr(S_f^{-1} C_f) ],

evaluated in *observed coordinates*: an orthonormal basis of what the channel
set actually measures after channel repair and average referencing.  Above
``EMG_SPLIT_HZ`` the muscle-prone channels are marginalised out (the Gaussian
marginal is the sub-matrix), so tonic scalp EMG is not scored.

Every per-subject operator is padded to 25 rows.  Padded coordinates carry an
identity in both model and data, contributing only a constant, so batches of
subjects with different channel sets vectorise.
"""
from __future__ import annotations

import numpy as np

DOF_PER_EPOCH = 2.2
FIXED_EMG_CHANNELS = ("Fp1", "Fp2", "F7", "F8", "T7", "T8")
EMG_SPLIT_HZ = 20.0
N_OBS = 25


def _operators(labels, keep, repair):
    """Data-side D and model-side A (N_OBS x 26) plus padding mask for one channel set."""
    n = len(labels)
    R = np.eye(n) - 1.0 / n
    P = np.eye(n) if repair is None else np.asarray(repair, float)
    rows = [labels.index(c) for c in keep]
    E = np.eye(n)[rows]
    M = E @ R @ P  # measured channels as functionals of the pre-reference potentials
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > 1e-8 * s[0]))
    r = min(r, N_OBS)
    D = np.zeros((N_OBS, n))
    A = np.zeros((N_OBS, n))
    D[:r] = U[:, :r].T @ E
    A[:r] = U[:, :r].T @ M
    pad = np.zeros(N_OBS, bool)
    pad[r:] = True
    return D, A, pad


def observation_operators(labels, frequency_hz, emg_channels=(), repair=None,
                          fixed_emg=FIXED_EMG_CHANNELS, split_hz=EMG_SPLIT_HZ):
    """Per-frequency (F, 25, 26) data/model operators and (F, 25) padding masks."""
    labels = list(labels)
    low = _operators(labels, labels, repair)
    excluded = set(fixed_emg) | set(emg_channels)
    high = _operators(labels, [c for c in labels if c not in excluded], repair)
    hi = np.asarray(frequency_hz) >= split_hz
    D = np.where(hi[:, None, None], high[0][None], low[0][None])
    A = np.where(hi[:, None, None], high[1][None], low[1][None])
    pad = np.where(hi[:, None], high[2][None], low[2][None])
    return D, A, pad


def project_data(csd, D, pad):
    """Observed-coordinate data CSD (F, 25, 25), normalised to unit mean power."""
    C = np.einsum("fij,fjk,flk->fil", D, csd, D)
    scale = np.mean(np.real(np.einsum("fii->fi", C))[~pad])
    C = C / scale
    C = 0.5 * (C + np.conj(np.swapaxes(C, -1, -2)))
    C[:, np.arange(N_OBS), np.arange(N_OBS)] += pad
    return C, scale


# ----------------------------------------------------------------------------
# JAX parts
# ----------------------------------------------------------------------------
def _jnp():
    import jax.numpy as jnp
    return jnp


def project_model(S, A, pad):
    jnp = _jnp()
    Sc = jnp.einsum("fij,fjk,flk->fil", A, S, A)
    Sc = 0.5 * (Sc + jnp.conj(jnp.swapaxes(Sc, -1, -2)))
    return Sc + pad[..., None] * jnp.eye(N_OBS)


def whittle_terms(Sc, Cc):
    """Per-frequency log det S and tr(S^{-1} C) for Hermitian positive-definite S."""
    jnp = _jnp()
    L = jnp.linalg.cholesky(Sc)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.real(jnp.diagonal(L, axis1=-2, axis2=-1))), axis=-1)
    X = jnp.linalg.solve(Sc, Cc)
    tr = jnp.real(jnp.trace(X, axis1=-2, axis2=-1))
    return logdet, tr


def profiled_nll(S, A, pad, Cc, nu):
    """NLL with the overall subject scale profiled out; returns (nll, log_scale).

    Padded coordinates are excluded from the scale estimate; their constant
    contribution is dropped.
    """
    jnp = _jnp()
    # normalise first so that float32 Cholesky factors stay well scaled
    level = jnp.mean(jnp.real(jnp.einsum("fii->fi", S)))
    Sn = S / level
    Sc = project_model(Sn, A, pad)
    # identity padding must stay unscaled: remove and re-add it around the scale
    logdet, tr = whittle_terms(Sc, Cc)
    r = jnp.sum(~pad, axis=-1)
    npad = N_OBS - r
    tr0 = tr - npad  # identity blocks contribute npad to the trace
    s = jnp.sum(nu * tr0) / jnp.sum(nu * r)
    nll = jnp.sum(nu * (r * jnp.log(s) + logdet)) + jnp.sum(nu * r)
    return nll, jnp.log(s) - jnp.log(level)


def nll_at_scale(S, log_scale, A, pad, Cc, nu):
    jnp = _jnp()
    Sc = project_model(S * jnp.exp(log_scale), A, pad)
    logdet, tr = whittle_terms(Sc, Cc)
    return jnp.sum(nu * (logdet + tr))


def deviance(S, log_scale, A, pad, Cc, nu):
    """Wishart deviance sum_f nu [tr(S^-1 C) - log det(S^-1 C) - r] (0 when S = C); numpy."""
    import numpy.linalg as la
    Sc = np.einsum("fij,fjk,flk->fil", A, S * np.exp(log_scale), A)
    Sc = 0.5 * (Sc + np.conj(np.swapaxes(Sc, -1, -2)))
    Sc[:, np.arange(N_OBS), np.arange(N_OBS)] += pad
    X = la.solve(Sc, Cc)
    sign, logdet = la.slogdet(X)
    r = np.sum(~pad, axis=-1)
    return float(np.sum(nu * (np.real(np.trace(X, axis1=-2, axis2=-1)) - (N_OBS - r) - logdet - r)))
