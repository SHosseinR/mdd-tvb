"""Structured low-dimensional physiology/connectome design for spectral M5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    "visual_time_contrast",
    "default_noise_contrast",
    "visual_noise_contrast",
    "network_noise_mode_1",
    "network_noise_mode_2",
    "default_dorsattn_weight_contrast",
    "default_salventattn_weight_contrast",
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
    visual_time_contrast: float
    default_noise_contrast: float
    visual_noise_contrast: float
    network_noise_mode_1: float
    network_noise_mode_2: float
    default_dorsattn_weight_contrast: float
    default_salventattn_weight_contrast: float

    def numeric_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in PARAMETER_NAMES], dtype=float
        )


def _linear(value: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + value * (bounds[1] - bounds[0]))


def _log(value: float, bounds: tuple[float, float]) -> float:
    return float(np.exp(np.log(bounds[0]) + value * np.log(bounds[1] / bounds[0])))


def spectral_parameter_ranges(
    settings: SpectralDesignConfig,
) -> tuple[tuple[float, float], ...]:
    """Return ranges in the exact order declared by ``PARAMETER_NAMES``."""

    return (
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
        settings.visual_time_contrast_range,
        settings.default_noise_contrast_range,
        settings.visual_noise_contrast_range,
        settings.network_noise_mode_1_range,
        settings.network_noise_mode_2_range,
        settings.default_dorsattn_weight_contrast_range,
        settings.default_salventattn_weight_contrast_range,
    )


def candidates_from_parameter_matrix(
    values: np.ndarray, settings: SpectralDesignConfig
) -> list[SpectralCandidate]:
    """Validate and convert an explicit adaptive design into candidates."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"candidate matrix must have shape (samples, {len(PARAMETER_NAMES)})"
        )
    if matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("candidate matrix must contain at least two finite rows")
    for column, (name, bounds) in enumerate(
        zip(PARAMETER_NAMES, spectral_parameter_ranges(settings), strict=True)
    ):
        if np.any(matrix[:, column] < bounds[0]) or np.any(
            matrix[:, column] > bounds[1]
        ):
            raise ValueError(f"adaptive candidate {name} lies outside {bounds}")
    return [
        SpectralCandidate(
            candidate_index=index,
            **{
                name: float(matrix[index, column])
                for column, name in enumerate(PARAMETER_NAMES)
            },
        )
        for index in range(len(matrix))
    ]


def load_spectral_candidate_table(
    path: str | Path, settings: SpectralDesignConfig
) -> list[SpectralCandidate]:
    """Load an explicit candidate CSV while ignoring provenance columns."""

    table = pd.read_csv(path)
    missing = [name for name in PARAMETER_NAMES if name not in table]
    if missing:
        raise ValueError(f"candidate table is missing columns: {missing}")
    return candidates_from_parameter_matrix(
        table.loc[:, PARAMETER_NAMES].to_numpy(dtype=float), settings
    )


def make_spectral_design(settings: SpectralDesignConfig) -> list[SpectralCandidate]:
    ranges = spectral_parameter_ranges(settings)
    logarithmic = {7, 8}
    reference = np.asarray(
        [
            settings.reference_global_coupling,
            settings.reference_speed_mm_per_ms,
            settings.reference_mu,
            settings.reference_a_scale,
            settings.reference_b_scale,
            settings.reference_fast_ratio,
            settings.reference_fast_fraction,
            settings.reference_noise_nsig,
            settings.reference_noise_tau_ms,
            settings.reference_dorsattn_time_contrast,
            settings.reference_visual_time_contrast,
            settings.reference_default_noise_contrast,
            settings.reference_visual_noise_contrast,
            settings.reference_network_noise_mode_1,
            settings.reference_network_noise_mode_2,
            settings.reference_default_dorsattn_weight_contrast,
            settings.reference_default_salventattn_weight_contrast,
        ],
        dtype=float,
    )
    reference_unit = np.empty_like(reference)
    for column, bounds in enumerate(ranges):
        if column in logarithmic:
            reference_unit[column] = np.log(
                reference[column] / bounds[0]
            ) / np.log(bounds[1] / bounds[0])
        else:
            reference_unit[column] = (
                reference[column] - bounds[0]
            ) / (bounds[1] - bounds[0])

    def sobol(count: int, dimensions: int, seed: int) -> np.ndarray:
        engine = qmc.Sobol(d=dimensions, scramble=True, seed=seed)
        exponent = int(np.ceil(np.log2(max(count, 1))))
        return engine.random_base2(exponent)[:count]

    if settings.strategy == "factorized":
        global_count = settings.global_samples
        spatial_count = settings.spatial_samples
        global_unit = sobol(global_count, 9, settings.seed)
        spatial_dimensions = len(PARAMETER_NAMES) - 9
        spatial_unit = sobol(
            spatial_count, spatial_dimensions, settings.seed + 104729
        )
        global_unit[0] = reference_unit[:9]
        spatial_unit[0] = reference_unit[9:]
        unit = np.asarray(
            [
                np.concatenate((global_row, spatial_row))
                for global_row in global_unit
                for spatial_row in spatial_unit
            ]
        )
    else:
        unit = sobol(settings.samples, len(PARAMETER_NAMES), settings.seed)
        unit[0] = reference_unit

    values = np.empty_like(unit)
    for column, bounds in enumerate(ranges):
        values[:, column] = [
            _log(value, bounds)
            if column in logarithmic
            else _linear(value, bounds)
            for value in unit[:, column]
        ]
    # Keep one exact reference state in every design for regression tests and
    # for a stable biological baseline across calibration/production banks.
    return candidates_from_parameter_matrix(values, settings)


def normalized_spectral_parameters(
    values: np.ndarray, settings: SpectralDesignConfig
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float).copy()
    ranges = spectral_parameter_ranges(settings)
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


def denormalized_spectral_parameters(
    values: np.ndarray, settings: SpectralDesignConfig
) -> np.ndarray:
    """Invert ``normalized_spectral_parameters`` for adaptive proposals."""

    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("normalized parameter matrix has the wrong shape")
    for column, bounds in enumerate(spectral_parameter_ranges(settings)):
        if column in {7, 8}:
            bound_values = np.log(bounds)
            transformed = (
                matrix[:, column] * (bound_values[1] - bound_values[0]) / 2.0
                + bound_values.mean()
            )
            matrix[:, column] = np.exp(transformed)
        else:
            bound_values = np.asarray(bounds)
            matrix[:, column] = (
                matrix[:, column] * (bound_values[1] - bound_values[0]) / 2.0
                + bound_values.mean()
            )
    return matrix
