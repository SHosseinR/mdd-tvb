"""Configuration for M5 hierarchical subject and group fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class FitPathsConfig:
    baseline_config: Path
    dataset_root: Path
    output_dir: Path


@dataclass(frozen=True)
class EmpiricalConfig:
    groups: tuple[str, ...] = ("Healthy", "MDD")
    max_duration_s: float = 120.0
    n_jobs: int = 4
    apply_surface_laplacian: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    epoch_seconds: float = 2.0
    frequency_min_hz: float = 2.0
    frequency_max_hz: float = 45.0
    coherence_pca_components: int = 12
    coherence_transform: str = "pca"
    coherence_reliability_threshold: float = 0.40
    coherence_max_features: int = 128
    coherence_fit_bands: tuple[str, ...] = ("theta", "alpha", "beta")
    use_mechanistic_spectrum: bool = False
    holdout_fraction: float = 0.2
    split_seed: int = 20260911
    psd_weight: float = 0.35
    spectral_summary_weight: float = 0.10
    alpha_topography_weight: float = 0.20
    coherence_weight: float = 0.35


@dataclass(frozen=True)
class DesignConfig:
    samples: int = 128
    replicates: int = 1
    duration_ms: float = 12000.0
    transient_ms: float = 2000.0
    dt_ms: float = 0.5
    n_jobs: int = 4
    seed: int = 271828
    simulation_seed: int = 9001
    global_coupling_range: tuple[float, float] = (4.0, 10.0)
    mu_range: tuple[float, float] = (0.18, 0.27)
    a_scale_range: tuple[float, float] = (0.65, 0.95)
    b_scale_range: tuple[float, float] = (0.85, 1.10)
    noise_nsig_range: tuple[float, float] = (0.003, 0.020)
    regional_time_log_sd_range: tuple[float, float] = (0.04, 0.16)
    noise_tau_ms: float = 1.0
    max_parameter_deviation: float = 0.40
    network_weight_deviation: float = 0.10
    fit_network_endpoint_gains: bool = True
    observation_noise_fraction: float = 0.0
    observation_noise_exponent: float = 0.5
    reference_global_coupling: float = 7.0
    reference_mu: float = 0.22
    reference_a_scale: float = 0.80
    reference_b_scale: float = 0.95
    reference_noise_nsig: float = 0.010
    reference_regional_time_log_sd: float = 0.08


@dataclass(frozen=True)
class HierarchicalFitConfig:
    group_prior_strength: float = 0.05
    subject_prior_strength: float = 0.01
    subject_prior_center: str = "pooled"


@dataclass(frozen=True)
class M5Config:
    config_path: Path
    project_root: Path
    paths: FitPathsConfig
    empirical: EmpiricalConfig
    features: FeatureConfig
    design: DesignConfig
    fit: HierarchicalFitConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _pair(raw: object, name: str) -> tuple[float, float]:
    values = tuple(float(value) for value in raw)  # type: ignore[arg-type]
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must contain two increasing values")
    return values


def load_m5_config(path: str | Path) -> M5Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent.parent.resolve()
    paths_raw = raw["paths"]
    paths = FitPathsConfig(
        baseline_config=_resolve(root, paths_raw["baseline_config"]),
        dataset_root=_resolve(root, paths_raw["dataset_root"]),
        output_dir=_resolve(root, paths_raw["output_dir"]),
    )
    empirical_raw = raw.get("empirical", {})
    empirical = EmpiricalConfig(
        groups=tuple(str(value) for value in empirical_raw.get("groups", ["Healthy", "MDD"])),
        max_duration_s=float(empirical_raw.get("max_duration_s", 120.0)),
        n_jobs=int(empirical_raw.get("n_jobs", 4)),
        apply_surface_laplacian=bool(
            empirical_raw.get("apply_surface_laplacian", True)
        ),
    )
    features = FeatureConfig(**raw.get("features", {}))
    design_raw = raw.get("design", {})
    design = DesignConfig(
        samples=int(design_raw.get("samples", 64)),
        replicates=int(design_raw.get("replicates", 1)),
        duration_ms=float(design_raw.get("duration_ms", 12000.0)),
        transient_ms=float(design_raw.get("transient_ms", 2000.0)),
        dt_ms=float(design_raw.get("dt_ms", 0.5)),
        n_jobs=int(design_raw.get("n_jobs", 4)),
        seed=int(design_raw.get("seed", 271828)),
        simulation_seed=int(design_raw.get("simulation_seed", 9001)),
        global_coupling_range=_pair(
            design_raw.get("global_coupling_range", [4.0, 10.0]),
            "global_coupling_range",
        ),
        mu_range=_pair(design_raw.get("mu_range", [0.18, 0.27]), "mu_range"),
        a_scale_range=_pair(
            design_raw.get("a_scale_range", [0.65, 0.95]),
            "a_scale_range",
        ),
        b_scale_range=_pair(
            design_raw.get("b_scale_range", [0.85, 1.10]),
            "b_scale_range",
        ),
        noise_nsig_range=_pair(
            design_raw.get("noise_nsig_range", [0.003, 0.020]),
            "noise_nsig_range",
        ),
        regional_time_log_sd_range=_pair(
            design_raw.get("regional_time_log_sd_range", [0.04, 0.16]),
            "regional_time_log_sd_range",
        ),
        noise_tau_ms=float(design_raw.get("noise_tau_ms", 1.0)),
        max_parameter_deviation=float(
            design_raw.get("max_parameter_deviation", 0.40)
        ),
        network_weight_deviation=float(
            design_raw.get("network_weight_deviation", 0.10)
        ),
        fit_network_endpoint_gains=bool(
            design_raw.get("fit_network_endpoint_gains", True)
        ),
        observation_noise_fraction=float(
            design_raw.get("observation_noise_fraction", 0.0)
        ),
        observation_noise_exponent=float(
            design_raw.get("observation_noise_exponent", 0.5)
        ),
        reference_global_coupling=float(
            design_raw.get("reference_global_coupling", 7.0)
        ),
        reference_mu=float(design_raw.get("reference_mu", 0.22)),
        reference_a_scale=float(design_raw.get("reference_a_scale", 0.80)),
        reference_b_scale=float(design_raw.get("reference_b_scale", 0.95)),
        reference_noise_nsig=float(
            design_raw.get("reference_noise_nsig", 0.010)
        ),
        reference_regional_time_log_sd=float(
            design_raw.get("reference_regional_time_log_sd", 0.08)
        ),
    )
    fit = HierarchicalFitConfig(**raw.get("fit", {}))

    if not paths.baseline_config.is_file():
        raise FileNotFoundError(f"Baseline config not found: {paths.baseline_config}")
    if not paths.dataset_root.is_dir():
        raise FileNotFoundError(f"TDBRAIN dataset root not found: {paths.dataset_root}")
    if len(empirical.groups) < 2 or len(set(empirical.groups)) != len(empirical.groups):
        raise ValueError("empirical groups must contain at least two unique labels")
    if empirical.max_duration_s <= 0 or empirical.n_jobs < 1 or design.n_jobs < 1:
        raise ValueError("durations and worker counts must be positive")
    if features.frequency_min_hz < 1.0 or features.frequency_max_hz > 60.0:
        raise ValueError("Feature frequencies must stay inside empirical 1-60 Hz preprocessing")
    if features.frequency_min_hz >= features.frequency_max_hz:
        raise ValueError("Feature frequency bounds are invalid")
    if not 0 < features.holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if abs(
        features.psd_weight
        + features.spectral_summary_weight
        + features.alpha_topography_weight
        + features.coherence_weight
        - 1.0
    ) > 1e-9:
        raise ValueError("Feature block weights must sum to one")
    if features.coherence_transform not in {"pca", "reliable_edges"}:
        raise ValueError("coherence_transform must be 'pca' or 'reliable_edges'")
    valid_bands = {"delta", "theta", "alpha", "beta"}
    if not features.coherence_fit_bands or not set(features.coherence_fit_bands).issubset(valid_bands):
        raise ValueError("coherence_fit_bands contains an invalid band")
    if not 0 <= features.coherence_reliability_threshold <= 1:
        raise ValueError("coherence_reliability_threshold must be in [0, 1]")
    if features.coherence_max_features < 1:
        raise ValueError("coherence_max_features must be positive")
    if design.samples < 2 or design.replicates < 1:
        raise ValueError("Design needs at least two samples and one replicate")
    if design.duration_ms <= design.transient_ms:
        raise ValueError("Design duration must exceed its transient")
    if not 0 < design.network_weight_deviation <= 0.10:
        raise ValueError("Network weight deviation must be in (0, 0.10]")
    if design.observation_noise_fraction < 0:
        raise ValueError("observation_noise_fraction must be non-negative")
    if not 0 <= design.observation_noise_exponent <= 2:
        raise ValueError("observation_noise_exponent must be in [0, 2]")
    if not 0 < design.max_parameter_deviation < 1:
        raise ValueError("max_parameter_deviation must be in (0, 1)")
    if design.noise_tau_ms <= 0:
        raise ValueError("noise_tau_ms must be positive")
    reference_checks = (
        (design.reference_global_coupling, design.global_coupling_range),
        (design.reference_mu, design.mu_range),
        (design.reference_a_scale, design.a_scale_range),
        (design.reference_b_scale, design.b_scale_range),
        (design.reference_noise_nsig, design.noise_nsig_range),
        (
            design.reference_regional_time_log_sd,
            design.regional_time_log_sd_range,
        ),
    )
    if any(not low <= value <= high for value, (low, high) in reference_checks):
        raise ValueError("Every reference parameter must lie inside its design range")
    if fit.group_prior_strength < 0 or fit.subject_prior_strength < 0:
        raise ValueError("Prior strengths must be non-negative")
    if fit.subject_prior_center not in {"pooled", "reference", "diagnosis"}:
        raise ValueError(
            "subject_prior_center must be 'pooled', 'reference', or 'diagnosis'"
        )

    return M5Config(
        config_path=config_path,
        project_root=root,
        paths=paths,
        empirical=empirical,
        features=features,
        design=design,
        fit=fit,
    )
