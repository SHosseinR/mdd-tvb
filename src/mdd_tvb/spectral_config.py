"""Configuration for the resting-state cross-spectral M5 redesign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SpectralPaths:
    baseline_config: Path
    dataset_root: Path
    output_dir: Path


@dataclass(frozen=True)
class SpectralEmpiricalConfig:
    groups: tuple[str, ...] = ("Healthy", "MDD")
    max_duration_s: float = 120.0
    n_jobs: int = 4
    apply_surface_laplacian: bool = True


@dataclass(frozen=True)
class CrossSpectralConfig:
    epoch_seconds: float = 4.0
    frequency_min_hz: float = 2.0
    frequency_max_hz: float = 40.0
    frequency_bin_hz: float = 1.0
    sensor_modes: int = 10
    auto_components: int = 16
    cross_components: int = 24
    reliability_threshold: float = 0.20
    reliability_max_auto_coordinates: int = 192
    reliability_max_cross_coordinates: int = 256
    reliability_min_coordinates: int = 24
    diagonal_shrinkage: float = 0.05
    auto_weight: float = 0.50
    cross_weight: float = 0.50
    holdout_fraction: float = 0.20
    split_seed: int = 20260913


@dataclass(frozen=True)
class SpectralDesignConfig:
    samples: int = 128
    replicates: int = 3
    duration_ms: float = 62000.0
    transient_ms: float = 2000.0
    dt_ms: float = 0.5
    n_jobs: int = 6
    seed: int = 161803
    simulation_seed: int = 29009
    global_coupling_range: tuple[float, float] = (4.0, 9.0)
    speed_range: tuple[float, float] = (3.0, 10.0)
    mu_range: tuple[float, float] = (0.19, 0.27)
    a_scale_range: tuple[float, float] = (0.70, 0.98)
    b_scale_range: tuple[float, float] = (0.85, 1.10)
    fast_ratio_range: tuple[float, float] = (1.6, 2.8)
    fast_fraction_range: tuple[float, float] = (0.10, 0.50)
    noise_nsig_range: tuple[float, float] = (0.001, 0.020)
    noise_tau_ms_range: tuple[float, float] = (0.5, 4.0)
    dorsattn_time_contrast_range: tuple[float, float] = (-0.10, 0.10)
    cross_coupling: float = 0.12
    reference_global_coupling: float = 7.0
    reference_speed_mm_per_ms: float = 4.0
    reference_mu: float = 0.22
    reference_a_scale: float = 0.80
    reference_b_scale: float = 0.95
    reference_fast_ratio: float = 2.2
    reference_fast_fraction: float = 0.25
    reference_noise_nsig: float = 0.010
    reference_noise_tau_ms: float = 1.0
    reference_dorsattn_time_contrast: float = 0.0


@dataclass(frozen=True)
class PosteriorConfig:
    observation_noise_fractions: tuple[float, ...] = (0.0, 0.15, 0.35, 0.55)
    observation_noise_exponents: tuple[float, ...] = (0.0, 0.75, 1.5)
    prior_strength: float = 0.02
    minimum_temperature: float = 0.05


@dataclass(frozen=True)
class SpectralM5Config:
    config_path: Path
    project_root: Path
    paths: SpectralPaths
    empirical: SpectralEmpiricalConfig
    spectral: CrossSpectralConfig
    design: SpectralDesignConfig
    posterior: PosteriorConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _pair(raw: object, name: str) -> tuple[float, float]:
    values = tuple(float(value) for value in raw)  # type: ignore[arg-type]
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must contain two increasing values")
    return values


def load_spectral_m5_config(path: str | Path) -> SpectralM5Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent.parent.resolve()
    paths_raw = raw["paths"]
    paths = SpectralPaths(
        baseline_config=_resolve(root, paths_raw["baseline_config"]),
        dataset_root=_resolve(root, paths_raw["dataset_root"]),
        output_dir=_resolve(root, paths_raw["output_dir"]),
    )
    empirical_raw = raw.get("empirical", {})
    empirical = SpectralEmpiricalConfig(
        groups=tuple(str(value) for value in empirical_raw.get("groups", ["Healthy", "MDD"])),
        max_duration_s=float(empirical_raw.get("max_duration_s", 120.0)),
        n_jobs=int(empirical_raw.get("n_jobs", 4)),
        apply_surface_laplacian=bool(empirical_raw.get("apply_surface_laplacian", True)),
    )
    spectral = CrossSpectralConfig(**raw.get("spectral", {}))
    design_raw = raw.get("design", {})
    design = SpectralDesignConfig(
        samples=int(design_raw.get("samples", 128)),
        replicates=int(design_raw.get("replicates", 3)),
        duration_ms=float(design_raw.get("duration_ms", 62000.0)),
        transient_ms=float(design_raw.get("transient_ms", 2000.0)),
        dt_ms=float(design_raw.get("dt_ms", 0.5)),
        n_jobs=int(design_raw.get("n_jobs", 6)),
        seed=int(design_raw.get("seed", 161803)),
        simulation_seed=int(design_raw.get("simulation_seed", 29009)),
        global_coupling_range=_pair(design_raw.get("global_coupling_range", [4.0, 9.0]), "global_coupling_range"),
        speed_range=_pair(design_raw.get("speed_range", [3.0, 10.0]), "speed_range"),
        mu_range=_pair(design_raw.get("mu_range", [0.19, 0.27]), "mu_range"),
        a_scale_range=_pair(design_raw.get("a_scale_range", [0.70, 0.98]), "a_scale_range"),
        b_scale_range=_pair(design_raw.get("b_scale_range", [0.85, 1.10]), "b_scale_range"),
        fast_ratio_range=_pair(design_raw.get("fast_ratio_range", [1.6, 2.8]), "fast_ratio_range"),
        fast_fraction_range=_pair(design_raw.get("fast_fraction_range", [0.10, 0.50]), "fast_fraction_range"),
        noise_nsig_range=_pair(design_raw.get("noise_nsig_range", [0.001, 0.020]), "noise_nsig_range"),
        noise_tau_ms_range=_pair(design_raw.get("noise_tau_ms_range", [0.5, 4.0]), "noise_tau_ms_range"),
        dorsattn_time_contrast_range=_pair(design_raw.get("dorsattn_time_contrast_range", [-0.10, 0.10]), "dorsattn_time_contrast_range"),
        cross_coupling=float(design_raw.get("cross_coupling", 0.12)),
        reference_global_coupling=float(design_raw.get("reference_global_coupling", 7.0)),
        reference_speed_mm_per_ms=float(design_raw.get("reference_speed_mm_per_ms", 4.0)),
        reference_mu=float(design_raw.get("reference_mu", 0.22)),
        reference_a_scale=float(design_raw.get("reference_a_scale", 0.80)),
        reference_b_scale=float(design_raw.get("reference_b_scale", 0.95)),
        reference_fast_ratio=float(design_raw.get("reference_fast_ratio", 2.2)),
        reference_fast_fraction=float(design_raw.get("reference_fast_fraction", 0.25)),
        reference_noise_nsig=float(design_raw.get("reference_noise_nsig", 0.010)),
        reference_noise_tau_ms=float(design_raw.get("reference_noise_tau_ms", 1.0)),
        reference_dorsattn_time_contrast=float(design_raw.get("reference_dorsattn_time_contrast", 0.0)),
    )
    posterior_raw = raw.get("posterior", {})
    posterior = PosteriorConfig(
        observation_noise_fractions=tuple(float(value) for value in posterior_raw.get("observation_noise_fractions", [0.0, 0.15, 0.35, 0.55])),
        observation_noise_exponents=tuple(float(value) for value in posterior_raw.get("observation_noise_exponents", [0.0, 0.75, 1.5])),
        prior_strength=float(posterior_raw.get("prior_strength", 0.02)),
        minimum_temperature=float(posterior_raw.get("minimum_temperature", 0.05)),
    )

    if not paths.baseline_config.is_file() or not paths.dataset_root.is_dir():
        raise FileNotFoundError("The baseline config or grouped EEG dataset path is missing")
    if design.samples < 2 or design.replicates < 1 or design.n_jobs < 1:
        raise ValueError("The design requires >=2 candidates, >=1 replicate, and >=1 worker")
    if design.duration_ms <= design.transient_ms or design.dt_ms <= 0:
        raise ValueError("Simulation duration/transient/dt are invalid")
    if not 0.0 <= spectral.diagonal_shrinkage < 1.0:
        raise ValueError("diagonal_shrinkage must be in [0, 1)")
    if spectral.sensor_modes < 2 or spectral.sensor_modes >= 26:
        raise ValueError("sensor_modes must be between 2 and 25 for average-referenced EEG")
    if not -1.0 <= spectral.reliability_threshold <= 1.0:
        raise ValueError("reliability_threshold must be in [-1, 1]")
    if min(
        spectral.reliability_max_auto_coordinates,
        spectral.reliability_max_cross_coordinates,
        spectral.reliability_min_coordinates,
    ) < 1:
        raise ValueError("reliability coordinate counts must be positive")
    if abs(spectral.auto_weight + spectral.cross_weight - 1.0) > 1e-9:
        raise ValueError("auto_weight and cross_weight must sum to one")
    if not 0.0 < spectral.holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if any(not 0.0 <= value < 1.0 for value in posterior.observation_noise_fractions):
        raise ValueError("observation noise fractions must be in [0, 1)")
    if any(value < 0.0 for value in posterior.observation_noise_exponents):
        raise ValueError("observation noise exponents must be non-negative")
    if posterior.prior_strength < 0.0 or posterior.minimum_temperature <= 0.0:
        raise ValueError("posterior regularization settings are invalid")
    references = (
        (design.reference_global_coupling, design.global_coupling_range),
        (design.reference_speed_mm_per_ms, design.speed_range),
        (design.reference_mu, design.mu_range),
        (design.reference_a_scale, design.a_scale_range),
        (design.reference_b_scale, design.b_scale_range),
        (design.reference_fast_ratio, design.fast_ratio_range),
        (design.reference_fast_fraction, design.fast_fraction_range),
        (design.reference_noise_nsig, design.noise_nsig_range),
        (design.reference_noise_tau_ms, design.noise_tau_ms_range),
        (design.reference_dorsattn_time_contrast, design.dorsattn_time_contrast_range),
    )
    if any(not low <= value <= high for value, (low, high) in references):
        raise ValueError("Every spectral reference parameter must lie inside its range")
    return SpectralM5Config(
        config_path=config_path,
        project_root=root,
        paths=paths,
        empirical=empirical,
        spectral=spectral,
        design=design,
        posterior=posterior,
    )
