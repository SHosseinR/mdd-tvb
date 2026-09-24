"""Continuous subject fitting with the analytic (linear-regime) spectrum.

The M5.1 workflow selected subjects from a 2,048-state stochastic bank.  Here
the expected cross spectrum is analytic (``linear_jax``), so

* global dynamics are searched on a dense *analytic* bank (no seed noise), and
* network-level input-noise gains and the observation nuisances are fitted
  continuously for every subject (they enter the CSD linearly / cheaply).

The scoring features are exactly the M5.1 ``SpectralFeatureTransformer``
features re-implemented in JAX, so results are directly comparable with the
M5.1/M5.2 nested evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

import os as _os

# Float64 by default (reference accuracy on CPU); set MDD_TVB_JAX_X64=0 for
# float32/complex64 on GPUs with weak double-precision throughput.
jax.config.update("jax_enable_x64", _os.environ.get("MDD_TVB_JAX_X64", "1") != "0")


def load_transformer_arrays(path: Path) -> dict:
    from .spectral_features import load_spectral_transformer

    t = load_spectral_transformer(Path(path))
    arrays = {
        "basis": jnp.asarray(t.sensor_basis),
        "freq": jnp.asarray(t.frequency_hz),
        "auto_mean": jnp.asarray(t.auto_mean),
        "auto_scale": jnp.asarray(t.auto_scale),
        "auto_comp": jnp.asarray(t.auto_components),
        "cross_mean": jnp.asarray(t.cross_mean),
        "cross_scale": jnp.asarray(t.cross_scale),
        "cross_comp": jnp.asarray(t.cross_components),
        "topo_mean": jnp.asarray(t.topography_mean),
        "topo_scale": jnp.asarray(t.topography_scale),
        "topo_comp": jnp.asarray(t.topography_components),
    }
    if t.cross_metric != "lagged_coherency":
        raise ValueError("Only the production lagged-coherency transformer is ported")
    meta = {
        "shrink": float(t.diagonal_shrinkage),
        "w_auto": float(t.auto_weight),
        "w_cross": float(t.cross_weight),
        "w_topo": float(t.topography_weight),
    }
    return {**arrays, **meta}


def feature_blocks(csd, T):
    """JAX mirror of ``SpectralFeatureTransformer.transform_blocks`` (one CSD)."""

    freq = T["freq"]
    diag = jnp.maximum(jnp.real(jnp.diagonal(csd, axis1=-2, axis2=-1)), 1e-300)
    alpha = (freq >= 8.0) & (freq < 13.0)
    topo = jnp.log(jnp.sum(jnp.where(alpha[:, None], diag, 0.0), 0) / jnp.sum(alpha))
    topo = topo - topo.mean()

    basis = T["basis"]
    projected = jnp.einsum("ci,fcd,dj->fij", basis, csd, basis)
    pdiag = jnp.maximum(jnp.real(jnp.diagonal(projected, axis1=-2, axis2=-1)), 1e-300)
    m = projected.shape[-1]
    eye = jnp.eye(m)
    projected = (1.0 - T["shrink"]) * projected + T["shrink"] * pdiag[:, :, None] * eye
    auto = jnp.log(pdiag)  # (F, m)
    background = (freq <= 7.0) | (freq >= 30.0)
    logf = jnp.log(freq)
    bmean = jnp.sum(jnp.where(background, logf, 0.0)) / jnp.sum(background)
    centered = jnp.where(background, logf - bmean, 0.0)
    intercept = jnp.sum(jnp.where(background[:, None], auto, 0.0), 0) / jnp.sum(background)
    slope = jnp.einsum("fm,f->m", auto - intercept, centered) / jnp.sum(centered**2)
    fitted = intercept[None, :] + slope[None, :] * (logf - bmean)[:, None]
    residual = auto - fitted
    auto_flat = jnp.concatenate((residual.reshape(-1), -slope))

    denom = jnp.sqrt(pdiag[:, :, None] * pdiag[:, None, :])
    coh = projected / denom
    iu = np.triu_indices(m, k=1)
    cross = coh[:, iu[0], iu[1]]
    real = jnp.real(cross)
    lagged = jnp.imag(cross) / jnp.sqrt(jnp.maximum(1.0 - real**2, 1e-8))
    cross_flat = lagged.reshape(-1)

    a = ((auto_flat - T["auto_mean"]) / T["auto_scale"]) @ T["auto_comp"].T
    c = ((cross_flat - T["cross_mean"]) / T["cross_scale"]) @ T["cross_comp"].T
    t = ((topo - T["topo_mean"]) / T["topo_scale"]) @ T["topo_comp"].T
    return a, c, t


def weighted_features(csd, T):
    a, c, t = feature_blocks(csd, T)
    return jnp.concatenate(
        (
            a * jnp.sqrt(T["w_auto"] / a.shape[0]),
            c * jnp.sqrt(T["w_cross"] / c.shape[0]),
            t * jnp.sqrt(T["w_topo"] / t.shape[0]),
        )
    )
