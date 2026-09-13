"""Reproducible low-dimensional spatial heterogeneity for baseline dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HeterogeneityConfig, ModelConfig


@dataclass(frozen=True)
class RegionalParameters:
    network_labels: np.ndarray
    mu: np.ndarray
    a: np.ndarray
    b: np.ndarray
    noise_nsig: np.ndarray
    drive_multiplier: np.ndarray
    time_scale_multiplier: np.ndarray
    noise_multiplier: np.ndarray


def network_labels(region_labels: np.ndarray) -> np.ndarray:
    """Extract the seven-network token from official Schaefer labels."""

    parsed: list[str] = []
    for label in region_labels.astype(str):
        parts = label.split("_")
        if len(parts) < 4 or parts[0] != "7Networks" or parts[1] not in {"LH", "RH"}:
            raise ValueError(f"Unexpected Schaefer-7 label: {label}")
        parsed.append(parts[2])
    result = np.asarray(parsed, dtype="U32")
    if np.unique(result).size != 7:
        raise ValueError(f"Expected seven Schaefer networks; found {np.unique(result).tolist()}")
    return result


def _multipliers(
    rng: np.random.Generator,
    networks: np.ndarray,
    network_log_sd: float,
    regional_log_sd: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    unique = np.unique(networks)
    network_draw = {name: rng.normal() for name in unique}
    network_component = np.asarray([network_draw[name] for name in networks])
    regional_component = rng.normal(size=networks.size)
    log_values = network_log_sd * network_component + regional_log_sd * regional_component
    log_values -= log_values.mean()
    return np.clip(np.exp(log_values), lower, upper)


def build_regional_parameters(
    model: ModelConfig,
    base_noise_nsig: float,
    region_labels: np.ndarray,
    settings: HeterogeneityConfig,
) -> RegionalParameters:
    """Generate bounded parameter maps without changing the population means."""

    networks = network_labels(region_labels)
    n_regions = region_labels.size
    if not settings.enabled:
        ones = np.ones(n_regions, dtype=float)
        return RegionalParameters(
            network_labels=networks,
            mu=np.full(n_regions, model.mu),
            a=np.full(n_regions, model.a),
            b=np.full(n_regions, model.b),
            noise_nsig=np.full(n_regions, base_noise_nsig),
            drive_multiplier=ones,
            time_scale_multiplier=ones,
            noise_multiplier=ones,
        )

    rng = np.random.default_rng(settings.seed)
    bound = settings.max_parameter_deviation
    drive = _multipliers(
        rng, networks, settings.network_log_sd, settings.regional_drive_log_sd,
        1.0 - bound, 1.0 + bound,
    )
    time_scale = _multipliers(
        rng, networks, settings.network_log_sd, settings.regional_time_scale_log_sd,
        1.0 - bound, 1.0 + bound,
    )
    noise_multiplier = _multipliers(
        rng, networks, settings.network_log_sd, settings.regional_noise_log_sd,
        0.5, 1.5,
    )
    visual = networks == "Vis"
    drive[visual] *= settings.visual_drive_multiplier
    noise_multiplier[visual] *= settings.visual_noise_multiplier

    for name, multiplier in (settings.network_drive_multipliers or {}).items():
        drive[networks == name] *= float(multiplier)
    for name, multiplier in (settings.network_time_scale_multipliers or {}).items():
        time_scale[networks == name] *= float(multiplier)
    for name, multiplier in (settings.network_noise_multipliers or {}).items():
        noise_multiplier[networks == name] *= float(multiplier)

    return RegionalParameters(
        network_labels=networks,
        mu=model.mu * drive,
        a=model.a * time_scale,
        b=model.b * time_scale,
        noise_nsig=base_noise_nsig * noise_multiplier,
        drive_multiplier=drive,
        time_scale_multiplier=time_scale,
        noise_multiplier=noise_multiplier,
    )
