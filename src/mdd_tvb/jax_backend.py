"""Batched JAX implementation of the delayed dual Jansen--Rit simulator.

This module is an accelerated numerical backend, not a second scientific
model.  Its equations, sparse delayed coupling, stochastic Heun update,
coloured neural noise, temporal averaging, and linear EEG observation mirror
``run_dual_jansen_rit``.  TVB remains the reference implementation used for
equivalence tests.

JAX is optional so the ordinary TVB workflow remains usable without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JaxDualBatch:
    """Numerical inputs for a batch of simulations sharing graph support."""

    weights: np.ndarray
    tract_lengths_ms_at_unit_speed: np.ndarray
    gain_matrix: np.ndarray
    regional_a: np.ndarray
    regional_b: np.ndarray
    regional_mu: np.ndarray
    regional_noise_nsig: np.ndarray
    global_coupling: np.ndarray
    speed_mm_per_ms: np.ndarray
    fast_ratio: np.ndarray
    fast_fraction: np.ndarray
    noise_tau_ms: np.ndarray
    seeds: np.ndarray


@dataclass(frozen=True)
class JaxDualResult:
    """EEG output and execution metadata returned by the JAX backend."""

    time_ms: np.ndarray
    eeg: np.ndarray
    metadata: dict[str, Any]


def _jax_modules() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The JAX backend requires the optional 'accelerated' dependencies. "
            "Install with: pip install -e .[accelerated]"
        ) from error
    return jax, jnp


def _validate_batch(batch: JaxDualBatch) -> tuple[int, int]:
    weights = np.asarray(batch.weights, dtype=float)
    lengths = np.asarray(batch.tract_lengths_ms_at_unit_speed, dtype=float)
    if weights.ndim not in {2, 3} or weights.shape[-1] != weights.shape[-2]:
        raise ValueError("weights must have shape (region, region) or (batch, region, region)")
    if lengths.shape != weights.shape[-2:]:
        raise ValueError("tract lengths must match the final two weight dimensions")
    n_regions = weights.shape[-1]
    regional = (
        batch.regional_a,
        batch.regional_b,
        batch.regional_mu,
        batch.regional_noise_nsig,
    )
    shapes = [np.asarray(value).shape for value in regional]
    if len(set(shapes)) != 1 or len(shapes[0]) != 2 or shapes[0][1] != n_regions:
        raise ValueError("regional parameters must all have shape (batch, region)")
    n_batch = shapes[0][0]
    scalar_names = (
        "global_coupling",
        "speed_mm_per_ms",
        "fast_ratio",
        "fast_fraction",
        "noise_tau_ms",
        "seeds",
    )
    for name in scalar_names:
        if np.asarray(getattr(batch, name)).shape != (n_batch,):
            raise ValueError(f"{name} must have shape (batch,)")
    if weights.ndim == 3 and weights.shape[0] != n_batch:
        raise ValueError("batched weights must have the same batch dimension as parameters")
    if np.any(np.asarray(batch.speed_mm_per_ms) <= 0):
        raise ValueError("conduction speeds must be positive")
    if np.any(np.asarray(batch.noise_tau_ms) <= 0):
        raise ValueError("this backend currently requires positive coloured-noise time constants")
    if not np.isfinite(weights).all() or not np.isfinite(lengths).all():
        raise ValueError("connectivity inputs must be finite")
    gain = np.asarray(batch.gain_matrix)
    if gain.ndim != 2 or gain.shape[1] != n_regions:
        raise ValueError("gain_matrix must have shape (channel, region)")
    return n_batch, n_regions


def run_dual_jansen_rit_jax(
    batch: JaxDualBatch,
    *,
    duration_ms: float,
    transient_ms: float,
    dt_ms: float,
    monitor_period_ms: float,
    cross_coupling: float = 0.12,
    reference: str = "average",
    channel_names: tuple[str, ...] | None = None,
    history_steps: int | None = None,
    precision: str = "float32",
    model_A: float = 3.25,
    model_B: float = 22.0,
    model_J: float = 135.0,
    fast_drive_ratio: float = 0.85,
    inhibitory_scale: float = 1.0,
) -> JaxDualResult:
    """Run many delayed dual-JR simulations in one compiled JAX program.

    The batch dimension can represent candidates, random replicates, or both.
    Structural support is shared, while edge weights may optionally vary by
    batch member.  Smooth per-edge connectome perturbations can therefore be
    added later without changing the solver.
    """

    n_batch, n_regions = _validate_batch(batch)
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    if duration_ms <= transient_ms or min(dt_ms, monitor_period_ms) <= 0:
        raise ValueError("duration, transient, dt, and monitor period are invalid")
    monitor_steps = int(round(monitor_period_ms / dt_ms))
    if not np.isclose(monitor_steps * dt_ms, monitor_period_ms):
        raise ValueError("monitor_period_ms must be an integer multiple of dt_ms")
    n_monitor = int(round(duration_ms / monitor_period_ms))
    if not np.isclose(n_monitor * monitor_period_ms, duration_ms):
        raise ValueError("duration_ms must be an integer multiple of monitor_period_ms")
    transient_samples = int(round(transient_ms / monitor_period_ms))
    if not np.isclose(transient_samples * monitor_period_ms, transient_ms):
        raise ValueError("transient_ms must be an integer multiple of monitor_period_ms")

    jax, jnp = _jax_modules()
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if precision == "float64" else jnp.float32

    weights = np.asarray(batch.weights, dtype=float)
    support = np.any(weights != 0.0, axis=0) if weights.ndim == 3 else weights != 0.0
    targets, sources = np.nonzero(support)
    if targets.size == 0:
        raise ValueError("connectivity has no non-zero edges")
    edge_weights = (
        weights[:, targets, sources]
        if weights.ndim == 3
        else np.broadcast_to(weights[targets, sources], (n_batch, targets.size))
    )
    edge_lengths = np.asarray(batch.tract_lengths_ms_at_unit_speed)[targets, sources]
    delays = np.rint(
        edge_lengths[np.newaxis, :]
        / np.asarray(batch.speed_mm_per_ms)[:, np.newaxis]
        / dt_ms
    ).astype(np.int32)
    required_history = int(delays.max()) + 1
    if history_steps is None:
        history_steps = required_history
    if history_steps < required_history:
        raise ValueError(
            f"history_steps={history_steps} is shorter than required {required_history}"
        )

    arrays = {
        "edge_weights": jnp.asarray(edge_weights, dtype=dtype),
        "delays": jnp.asarray(delays, dtype=jnp.int32),
        "targets": jnp.asarray(targets, dtype=jnp.int32),
        "sources": jnp.asarray(sources, dtype=jnp.int32),
        "gain": jnp.asarray(batch.gain_matrix, dtype=dtype),
        "a": jnp.asarray(batch.regional_a, dtype=dtype),
        "b": jnp.asarray(batch.regional_b, dtype=dtype),
        "mu": jnp.asarray(batch.regional_mu, dtype=dtype),
        "nsig": jnp.asarray(batch.regional_noise_nsig, dtype=dtype),
        "global_coupling": jnp.asarray(batch.global_coupling, dtype=dtype),
        "fast_ratio": jnp.asarray(batch.fast_ratio, dtype=dtype),
        "fast_fraction": jnp.asarray(batch.fast_fraction, dtype=dtype),
        "noise_tau": jnp.asarray(batch.noise_tau_ms, dtype=dtype),
        "seeds": jnp.asarray(batch.seeds, dtype=jnp.int32),
    }

    if reference and reference.lower() != "average":
        if channel_names is None or reference not in channel_names:
            raise ValueError("a named EEG reference requires matching channel_names")
        reference_index = channel_names.index(reference)
    else:
        reference_index = -1

    # Constants are explicit here so the accelerated path remains an equation-
    # for-equation implementation of DualJansenRit rather than depending on an
    # evolving third-party model definition.
    A, B, J = map(dtype, (model_A, model_B, model_J))
    v0, nu_max, slope = map(dtype, (5.52, 0.0025, 0.56))
    a_1, a_2, a_3, a_4 = map(dtype, (1.0, 0.8, 0.25, 0.25))
    fast_drive_ratio_value = dtype(fast_drive_ratio)
    inhibitory_scale_value = dtype(inhibitory_scale)
    cross = dtype(cross_coupling)
    dt = dtype(dt_ms)
    cmax = dtype(0.005)
    batch_index = jnp.arange(n_batch, dtype=jnp.int32)[:, None]

    def sigmoid(value: Any) -> Any:
        argument = jnp.clip(slope * (v0 - value), -60.0, 60.0)
        return dtype(2.0) * nu_max / (dtype(1.0) + jnp.exp(argument))

    def derivative(state: Any, long_range: Any) -> Any:
        alpha_out = sigmoid(state[:, :, 1] - state[:, :, 2])
        alpha_exc = sigmoid(a_1 * J * state[:, :, 0])
        alpha_inh = sigmoid(a_3 * J * state[:, :, 0])
        fast_out = sigmoid(state[:, :, 7] - state[:, :, 8])
        fast_exc = sigmoid(a_1 * J * state[:, :, 6])
        fast_inh = sigmoid(a_3 * J * state[:, :, 6])
        alpha_a = arrays["a"]
        alpha_b = arrays["b"]
        fast_a = alpha_a * arrays["fast_ratio"][:, None]
        fast_b = alpha_b * arrays["fast_ratio"][:, None]

        dx = jnp.zeros_like(state)
        dx = dx.at[:, :, 0].set(state[:, :, 3])
        dx = dx.at[:, :, 1].set(state[:, :, 4])
        dx = dx.at[:, :, 2].set(state[:, :, 5])
        dx = dx.at[:, :, 3].set(
            A * alpha_a * alpha_out
            - dtype(2.0) * alpha_a * state[:, :, 3]
            - alpha_a**2 * state[:, :, 0]
        )
        dx = dx.at[:, :, 4].set(
            A
            * alpha_a
            * (
                arrays["mu"]
                + a_2 * J * alpha_exc
                + long_range
                + cross * J * fast_exc
            )
            - dtype(2.0) * alpha_a * state[:, :, 4]
            - alpha_a**2 * state[:, :, 1]
        )
        dx = dx.at[:, :, 5].set(
            B * inhibitory_scale_value * alpha_b * (a_4 * J * alpha_inh)
            - dtype(2.0) * alpha_b * state[:, :, 5]
            - alpha_b**2 * state[:, :, 2]
        )
        dx = dx.at[:, :, 6].set(state[:, :, 9])
        dx = dx.at[:, :, 7].set(state[:, :, 10])
        dx = dx.at[:, :, 8].set(state[:, :, 11])
        dx = dx.at[:, :, 9].set(
            A * fast_a * fast_out
            - dtype(2.0) * fast_a * state[:, :, 9]
            - fast_a**2 * state[:, :, 6]
        )
        dx = dx.at[:, :, 10].set(
            A
            * fast_a
            * (
                arrays["mu"] * fast_drive_ratio_value
                + a_2 * J * fast_exc
                + long_range
                + cross * J * alpha_exc
            )
            - dtype(2.0) * fast_a * state[:, :, 10]
            - fast_a**2 * state[:, :, 7]
        )
        dx = dx.at[:, :, 11].set(
            B * inhibitory_scale_value * fast_b * (a_4 * J * fast_inh)
            - dtype(2.0) * fast_b * state[:, :, 11]
            - fast_b**2 * state[:, :, 8]
        )
        return dx

    def delayed_coupling(history: Any, head: Any) -> Any:
        time_index = (head - arrays["delays"]) % history_steps
        delayed = history[
            time_index,
            batch_index,
            arrays["sources"][None, :],
            :,
        ]
        alpha_psp = delayed[:, :, 0] - delayed[:, :, 1]
        fast_psp = delayed[:, :, 2] - delayed[:, :, 3]
        alpha_rate = cmax / (dtype(1.0) + jnp.exp(slope * (v0 - alpha_psp)))
        fast_rate = cmax / (dtype(1.0) + jnp.exp(slope * (v0 - fast_psp)))
        fraction = arrays["fast_fraction"][:, None]
        mixed = (dtype(1.0) - fraction) * alpha_rate + fraction * fast_rate
        contributions = arrays["edge_weights"] * mixed
        summed = jnp.zeros((n_batch, n_regions), dtype=dtype)
        summed = summed.at[:, arrays["targets"]].add(contributions)
        return arrays["global_coupling"][:, None] * summed

    def integrate_step(carry: tuple[Any, ...], _: Any) -> tuple[tuple[Any, ...], Any]:
        state, history, head, eta, keys = carry
        long_range = delayed_coupling(history, head)
        pairs = jax.vmap(jax.random.split)(keys)
        next_keys, sample_keys = pairs[:, 0], pairs[:, 1]
        innovations = jax.vmap(
            lambda key: jax.random.normal(key, (n_regions, 2), dtype=dtype)
        )(sample_keys)
        decay = jnp.exp(-dt / arrays["noise_tau"])[:, None, None]
        eta = eta * decay + jnp.sqrt(dtype(1.0) - decay**2) * innovations
        noise_pair = (
            dt
            * jnp.sqrt(dtype(1.0) / arrays["noise_tau"])[:, None, None]
            * eta
            # TVB Additive.gfun() treats nsig as diffusion intensity D and
            # multiplies the generated process by sqrt(2D).
            * jnp.sqrt(dtype(2.0) * arrays["nsig"][:, :, None])
        )
        noise = jnp.zeros_like(state)
        noise = noise.at[:, :, 4].set(noise_pair[:, :, 0])
        noise = noise.at[:, :, 10].set(noise_pair[:, :, 1])
        first = derivative(state, long_range)
        predictor = state + dt * first + noise
        second = derivative(predictor, long_range)
        state = state + dt * (first + second) / dtype(2.0) + noise
        head = (head + 1) % history_steps
        cvars = state[:, :, jnp.asarray([1, 2, 7, 8])]
        history = history.at[head].set(cvars)
        fraction = arrays["fast_fraction"][:, None]
        regional_psp = (
            (dtype(1.0) - fraction) * (state[:, :, 1] - state[:, :, 2])
            + fraction * (state[:, :, 7] - state[:, :, 8])
        )
        return (state, history, head, eta, next_keys), regional_psp

    def monitor_step(carry: tuple[Any, ...], _: Any) -> tuple[tuple[Any, ...], Any]:
        carry, regional = jax.lax.scan(
            integrate_step, carry, xs=None, length=monitor_steps
        )
        averaged = jnp.mean(regional, axis=0)
        eeg = averaged @ arrays["gain"].T
        if reference:
            if reference.lower() == "average":
                eeg = eeg - jnp.mean(eeg, axis=1, keepdims=True)
            else:
                eeg = eeg - eeg[:, reference_index, None]
        return carry, eeg

    def simulate() -> Any:
        keys = jax.vmap(jax.random.PRNGKey)(arrays["seeds"])
        pairs = jax.vmap(jax.random.split)(keys)
        initial_keys, keys = pairs[:, 0], pairs[:, 1]
        eta = jax.vmap(
            lambda key: jax.random.normal(key, (n_regions, 2), dtype=dtype)
        )(initial_keys)
        state = jnp.zeros((n_batch, n_regions, 12), dtype=dtype)
        history = jnp.zeros((history_steps, n_batch, n_regions, 4), dtype=dtype)
        carry = (state, history, jnp.asarray(history_steps - 1), eta, keys)
        _, eeg = jax.lax.scan(monitor_step, carry, xs=None, length=n_monitor)
        return jnp.transpose(eeg[transient_samples:], (1, 0, 2))

    compiled = jax.jit(simulate)
    started = perf_counter()
    device_eeg = compiled()
    device_eeg.block_until_ready()
    elapsed = perf_counter() - started
    eeg = np.asarray(device_eeg)
    time_ms = np.arange(1, eeg.shape[1] + 1, dtype=float) * monitor_period_ms
    return JaxDualResult(
        time_ms=time_ms,
        eeg=eeg,
        metadata={
            "backend": "jax",
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "precision": precision,
            "batch_size": n_batch,
            "regions": n_regions,
            "directed_edges": int(targets.size),
            "history_steps": int(history_steps),
            "duration_ms": float(duration_ms),
            "transient_ms": float(transient_ms),
            "dt_ms": float(dt_ms),
            "monitor_period_ms": float(monitor_period_ms),
            "model_A": float(model_A),
            "model_B": float(model_B),
            "model_J": float(model_J),
            "fast_drive_ratio": float(fast_drive_ratio),
            "inhibitory_scale": float(inhibitory_scale),
            "compile_and_run_seconds": float(elapsed),
        },
    )
