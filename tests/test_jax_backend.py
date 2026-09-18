from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from mdd_tvb.jax_backend import JaxDualBatch, run_dual_jansen_rit_jax


def test_jax_backend_is_reproducible_and_average_referenced() -> None:
    weights = np.asarray(
        [[0.0, 0.2, 0.0], [0.2, 0.0, 0.1], [0.0, 0.1, 0.0]]
    )
    lengths = np.where(weights > 0, 8.0, 0.0)
    gain = np.asarray([[1.0, -0.4, 0.2], [-0.3, 0.8, -0.1]])
    n_batch, n_regions = 2, 3
    batch = JaxDualBatch(
        weights=weights,
        tract_lengths_ms_at_unit_speed=lengths,
        gain_matrix=gain,
        regional_a=np.full((n_batch, n_regions), 0.104),
        regional_b=np.full((n_batch, n_regions), 0.06175),
        regional_mu=np.full((n_batch, n_regions), 0.22),
        regional_noise_nsig=np.full((n_batch, n_regions), 0.005),
        global_coupling=np.full(n_batch, 7.0),
        speed_mm_per_ms=np.full(n_batch, 4.0),
        fast_ratio=np.full(n_batch, 2.2),
        fast_fraction=np.full(n_batch, 0.25),
        noise_tau_ms=np.full(n_batch, 1.0),
        seeds=np.asarray([11, 12]),
    )
    kwargs = dict(
        duration_ms=20.0,
        transient_ms=4.0,
        dt_ms=0.5,
        monitor_period_ms=2.0,
        reference="average",
    )
    first = run_dual_jansen_rit_jax(batch, **kwargs)
    second = run_dual_jansen_rit_jax(batch, **kwargs)
    assert first.eeg.shape == (2, 8, 2)
    assert np.array_equal(first.eeg, second.eeg)
    assert np.allclose(first.eeg.sum(axis=2), 0.0, atol=5e-7)
    assert np.isfinite(first.eeg).all()


def test_jax_backend_rejects_too_short_history() -> None:
    batch = JaxDualBatch(
        weights=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        tract_lengths_ms_at_unit_speed=np.asarray([[0.0, 20.0], [20.0, 0.0]]),
        gain_matrix=np.eye(2),
        regional_a=np.full((1, 2), 0.13),
        regional_b=np.full((1, 2), 0.065),
        regional_mu=np.full((1, 2), 0.22),
        regional_noise_nsig=np.zeros((1, 2)),
        global_coupling=np.asarray([7.0]),
        speed_mm_per_ms=np.asarray([4.0]),
        fast_ratio=np.asarray([2.0]),
        fast_fraction=np.asarray([0.25]),
        noise_tau_ms=np.asarray([1.0]),
        seeds=np.asarray([1]),
    )
    with pytest.raises(ValueError, match="shorter than required"):
        run_dual_jansen_rit_jax(
            batch,
            duration_ms=10.0,
            transient_ms=2.0,
            dt_ms=0.5,
            monitor_period_ms=2.0,
            history_steps=2,
        )
