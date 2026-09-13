"""Low-dimensional, fixed-connectome design for spectral M5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .spectral_config import SpectralDesignConfig


PARAMETER_NAMES: tuple[str, ...] = (
    "global_coupling",
    "speed_mm_per_ms",
    "mu",
    "a_scale",
    "b_scale",
    "fast_ratio",
    "fast_fraction",
    "noise_nsig",
    "noise_tau_ms",
    "dorsattn_time_contrast",
)


@dataclass(frozen=True)
class SpectralCandidate:
    candidate_index: int
    global_coupling: float
    speed_mm_per_ms: float
    mu: float
    a_scale: float
    b_scale: float
    fast_ratio: float
    fast_fraction: float
    noise_nsig: float
    noise_tau_ms: float
    dorsattn_time_contrast: float

    def numeric_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in PARAMETER_NAMES], dtype=float
        )


def _linear(value: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + value * (bounds[1] - bounds[0]))


def _log(value: float, bounds: tuple[float, float]) -> float:
    return float(np.exp(np.log(bounds[0]) + value * np.log(bounds[1] / bounds[0])))


def make_spectral_design(settings: SpectralDesignConfig) -> list[SpectralCandidate]:
    engine = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=settings.seed)
    exponent = int(np.ceil(np.log2(max(settings.samples - 1, 1))))
    unit = engine.random_base2(exponent)[: max(settings.samples - 1, 1)]
    ranges = (
        settings.global_coupling_range,
        settings.speed_range,
        settings.mu_range,
        settings.a_scale_range,
        settings.b_scale_range,
        settings.fast_ratio_range,
        settings.fast_fraction_range,
        settings.noise_nsig_range,
        settings.noise_tau_ms_range,
        settings.dorsattn_time_contrast_range,
    )
    logarithmic = {7, 8}
    candidates = [SpectralCandidate(
        candidate_index=0,
        global_coupling=settings.reference_global_coupling,
        speed_mm_per_ms=settings.reference_speed_mm_per_ms,
        mu=settings.reference_mu,
        a_scale=settings.reference_a_scale,
        b_scale=settings.reference_b_scale,
        fast_ratio=settings.reference_fast_ratio,
        fast_fraction=settings.reference_fast_fraction,
        noise_nsig=settings.reference_noise_nsig,
        noise_tau_ms=settings.reference_noise_tau_ms,
        dorsattn_time_contrast=settings.reference_dorsattn_time_contrast,
    )]
    candidates.extend(
        SpectralCandidate(
            candidate_index=index + 1,
            **{
                name: (
                    _log(row[column], ranges[column])
                    if column in logarithmic
                    else _linear(row[column], ranges[column])
                )
                for column, name in enumerate(PARAMETER_NAMES)
            },
        )
        for index, row in enumerate(unit)
    )
    return candidates[: settings.samples]


def normalized_spectral_parameters(
    values: np.ndarray, settings: SpectralDesignConfig
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float).copy()
    ranges = (
        settings.global_coupling_range,
        settings.speed_range,
        settings.mu_range,
        settings.a_scale_range,
        settings.b_scale_range,
        settings.fast_ratio_range,
        settings.fast_fraction_range,
        settings.noise_nsig_range,
        settings.noise_tau_ms_range,
        settings.dorsattn_time_contrast_range,
    )
    for column, bounds in enumerate(ranges):
        if column in {7, 8}:
            transformed = np.log(matrix[:, column])
            bound_values = np.log(bounds)
        else:
            transformed = matrix[:, column]
            bound_values = np.asarray(bounds)
        matrix[:, column] = 2.0 * (
            transformed - bound_values.mean()
        ) / (bound_values[1] - bound_values[0])
    return matrix
