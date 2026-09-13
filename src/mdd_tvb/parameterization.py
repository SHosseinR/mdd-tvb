"""Identifiable low-dimensional parameterization for the M5 simulation bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .fit_config import DesignConfig


NETWORK_ORDER: tuple[str, ...] = (
    "Cont",
    "Default",
    "DorsAttn",
    "Limbic",
    "SalVentAttn",
    "SomMot",
    "Vis",
)


@dataclass(frozen=True)
class CandidateParameters:
    candidate_index: int
    global_coupling: float
    mu: float
    a_scale: float
    b_scale: float
    noise_nsig: float
    regional_time_log_sd: float
    network_gains: np.ndarray

    def numeric_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.global_coupling,
                self.mu,
                self.a_scale,
                self.b_scale,
                self.noise_nsig,
                self.regional_time_log_sd,
                *self.network_gains,
            ],
            dtype=float,
        )


PARAMETER_NAMES: tuple[str, ...] = (
    "global_coupling",
    "mu",
    "a_scale",
    "b_scale",
    "noise_nsig",
    "regional_time_log_sd",
    *(f"network_gain_{name}" for name in NETWORK_ORDER),
)


def _map_range(value: float, bounds: tuple[float, float]) -> float:
    return float(bounds[0] + value * (bounds[1] - bounds[0]))


def _map_log_range(value: float, bounds: tuple[float, float]) -> float:
    return float(np.exp(np.log(bounds[0]) + value * np.log(bounds[1] / bounds[0])))


def _network_gains(raw: np.ndarray, deviation: float) -> np.ndarray:
    contrast = np.asarray(raw, dtype=float) * 2.0 - 1.0
    contrast -= contrast.mean()
    maximum = float(np.max(np.abs(contrast)))
    if maximum > 1.0:
        contrast /= maximum
    gains = 1.0 + deviation * contrast
    # Arithmetic mean one separates relative topology from global coupling.
    gains -= gains.mean() - 1.0
    return gains


def make_design(
    settings: DesignConfig,
) -> list[CandidateParameters]:
    """Generate a scrambled Sobol design plus an exact reference candidate."""

    engine = qmc.Sobol(d=6 + len(NETWORK_ORDER), scramble=True, seed=settings.seed)
    exponent = int(np.ceil(np.log2(max(settings.samples - 1, 1))))
    unit = engine.random_base2(exponent)[: max(settings.samples - 1, 1)]
    candidates: list[CandidateParameters] = [CandidateParameters(
        candidate_index=0,
        global_coupling=settings.reference_global_coupling,
        mu=settings.reference_mu,
        a_scale=settings.reference_a_scale,
        b_scale=settings.reference_b_scale,
        noise_nsig=settings.reference_noise_nsig,
        regional_time_log_sd=settings.reference_regional_time_log_sd,
        network_gains=np.ones(len(NETWORK_ORDER)),
    )]
    for index, row in enumerate(unit, start=1):
        if index >= settings.samples:
            break
        candidates.append(CandidateParameters(
            candidate_index=index,
            global_coupling=_map_range(row[0], settings.global_coupling_range),
            mu=_map_range(row[1], settings.mu_range),
            a_scale=_map_range(row[2], settings.a_scale_range),
            b_scale=_map_range(row[3], settings.b_scale_range),
            noise_nsig=_map_log_range(row[4], settings.noise_nsig_range),
            regional_time_log_sd=_map_range(
                row[5], settings.regional_time_log_sd_range
            ),
            network_gains=(
                _network_gains(row[6:], settings.network_weight_deviation)
                if settings.fit_network_endpoint_gains
                else np.ones(len(NETWORK_ORDER))
            ),
        ))
    return candidates


def normalized_parameter_matrix(
    values: np.ndarray, settings: DesignConfig
) -> np.ndarray:
    """Normalize parameters to comparable prior-distance units."""

    matrix = np.asarray(values, dtype=float).copy()
    ranges = (
        settings.global_coupling_range,
        settings.mu_range,
        settings.a_scale_range,
        settings.b_scale_range,
        settings.noise_nsig_range,
        settings.regional_time_log_sd_range,
    )
    for column, bounds in enumerate(ranges):
        if column == 4:
            log_values = np.log(matrix[:, column])
            log_bounds = np.log(bounds)
            matrix[:, column] = (
                log_values - np.mean(log_bounds)
            ) / ((log_bounds[1] - log_bounds[0]) / 2.0)
        else:
            matrix[:, column] = (
                matrix[:, column] - np.mean(bounds)
            ) / ((bounds[1] - bounds[0]) / 2.0)
    matrix[:, 6:] = (matrix[:, 6:] - 1.0) / settings.network_weight_deviation
    return matrix
