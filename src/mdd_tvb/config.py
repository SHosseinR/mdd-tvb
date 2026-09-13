"""Typed TOML configuration for the baseline simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    count_matrix: Path
    distance_matrix: Path
    centroids: Path
    output_dir: Path


@dataclass(frozen=True)
class ConnectivityConfig:
    expected_regions: int = 200
    weight_transform: str = "log1p"
    normalization: str = "region_sum"
    speed_mm_per_ms: float = 4.0
    symmetry_tolerance: float = 1e-10


@dataclass(frozen=True)
class ModelConfig:
    A: float = 3.25
    B: float = 22.0
    a: float = 0.13
    b: float = 0.065
    J: float = 135.0
    mu: float = 0.22


@dataclass(frozen=True)
class CouplingConfig:
    global_gain: float = 15.0


@dataclass(frozen=True)
class SimulationConfig:
    duration_ms: float = 30000.0
    transient_ms: float = 2000.0
    dt_ms: float = 0.25
    monitor_period_ms: float = 2.0
    noise_nsig: float = 1e-4
    noise_tau_ms: float = 5.0
    seed: int = 42


@dataclass(frozen=True)
class MonitorConfig:
    reference: str
    surface_laplacian: bool
    montage: str
    channels: tuple[str, ...]
    coordinate_file: Path | None = None
    minimum_source_sensor_distance_mm: float = 0.0
    visualization_highpass_hz: float = 1.0


@dataclass(frozen=True)
class HeterogeneityConfig:
    enabled: bool = True
    seed: int = 314159
    network_log_sd: float = 0.02
    regional_drive_log_sd: float = 0.06
    regional_time_scale_log_sd: float = 0.04
    regional_noise_log_sd: float = 0.20
    max_parameter_deviation: float = 0.15
    visual_drive_multiplier: float = 1.03
    visual_noise_multiplier: float = 1.40
    network_drive_multipliers: dict[str, float] | None = None
    network_time_scale_multipliers: dict[str, float] | None = None
    network_noise_multipliers: dict[str, float] | None = None


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    project_root: Path
    paths: PathsConfig
    connectivity: ConnectivityConfig
    model: ModelConfig
    coupling: CouplingConfig
    simulation: SimulationConfig
    heterogeneity: HeterogeneityConfig
    monitor: MonitorConfig


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a baseline TOML configuration."""

    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project_root = config_path.parent.parent.resolve()
    paths_raw = raw["paths"]
    paths = PathsConfig(
        count_matrix=_resolve(project_root, paths_raw["count_matrix"]),
        distance_matrix=_resolve(project_root, paths_raw["distance_matrix"]),
        centroids=_resolve(project_root, paths_raw["centroids"]),
        output_dir=_resolve(project_root, paths_raw["output_dir"]),
    )
    connectivity = ConnectivityConfig(**raw.get("connectivity", {}))
    model = ModelConfig(**raw.get("model", {}))
    coupling = CouplingConfig(**raw.get("coupling", {}))
    simulation = SimulationConfig(**raw.get("simulation", {}))
    heterogeneity = HeterogeneityConfig(**raw.get("heterogeneity", {}))
    monitor_raw = raw["monitor"]
    monitor = MonitorConfig(
        reference=str(monitor_raw.get("reference", "average")),
        surface_laplacian=bool(monitor_raw.get("surface_laplacian", False)),
        montage=str(monitor_raw.get("montage", "standard_1005")),
        channels=tuple(str(value) for value in monitor_raw["channels"]),
        coordinate_file=(
            _resolve(project_root, monitor_raw["coordinate_file"])
            if monitor_raw.get("coordinate_file")
            else None
        ),
        minimum_source_sensor_distance_mm=float(
            monitor_raw.get("minimum_source_sensor_distance_mm", 0.0)
        ),
        visualization_highpass_hz=float(
            monitor_raw.get("visualization_highpass_hz", 1.0)
        ),
    )

    if simulation.duration_ms <= simulation.transient_ms:
        raise ValueError("duration_ms must be greater than transient_ms")
    if simulation.dt_ms <= 0 or simulation.monitor_period_ms <= 0:
        raise ValueError("dt_ms and monitor_period_ms must be positive")
    steps_per_sample = simulation.monitor_period_ms / simulation.dt_ms
    if abs(steps_per_sample - round(steps_per_sample)) > 1e-9:
        raise ValueError("monitor_period_ms must be an integer multiple of dt_ms")
    if connectivity.speed_mm_per_ms <= 0:
        raise ValueError("speed_mm_per_ms must be positive")
    if coupling.global_gain < 0:
        raise ValueError("global_gain must be non-negative")
    if simulation.noise_nsig < 0:
        raise ValueError("noise_nsig must be non-negative")
    if simulation.noise_tau_ms < 0:
        raise ValueError("noise_tau_ms must be non-negative")
    if not 0 <= heterogeneity.max_parameter_deviation < 1:
        raise ValueError("max_parameter_deviation must be in [0, 1)")
    for name in (
        "network_log_sd",
        "regional_drive_log_sd",
        "regional_time_scale_log_sd",
        "regional_noise_log_sd",
    ):
        if getattr(heterogeneity, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if heterogeneity.visual_drive_multiplier <= 0 or heterogeneity.visual_noise_multiplier <= 0:
        raise ValueError("visual multipliers must be positive")
    for mapping_name in (
        "network_drive_multipliers",
        "network_time_scale_multipliers",
        "network_noise_multipliers",
    ):
        mapping = getattr(heterogeneity, mapping_name) or {}
        if any(float(value) <= 0 for value in mapping.values()):
            raise ValueError(f"{mapping_name} values must be positive")
    if len(monitor.channels) != len(set(monitor.channels)):
        raise ValueError("monitor channel labels must be unique")
    if monitor.coordinate_file is not None and not monitor.coordinate_file.is_file():
        raise FileNotFoundError(
            f"sensor coordinate file not found: {monitor.coordinate_file}"
        )
    if monitor.minimum_source_sensor_distance_mm < 0:
        raise ValueError("minimum_source_sensor_distance_mm must be non-negative")
    nyquist_hz = 500.0 / simulation.monitor_period_ms
    if not 0 < monitor.visualization_highpass_hz < nyquist_hz:
        raise ValueError("visualization_highpass_hz must lie between 0 and Nyquist")

    return RunConfig(
        config_path=config_path,
        project_root=project_root,
        paths=paths,
        connectivity=connectivity,
        model=model,
        coupling=coupling,
        simulation=simulation,
        heterogeneity=heterogeneity,
        monitor=monitor,
    )
