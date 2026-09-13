"""Template EEG sensors, TVB monitor, and optional surface-Laplacian transform."""

from __future__ import annotations

import csv
import hashlib

import numpy as np
import mne
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator import monitors

from .config import MonitorConfig


def _coordinate_values(settings: MonitorConfig) -> tuple[np.ndarray, str]:
    if settings.coordinate_file is None:
        montage = mne.channels.make_standard_montage(settings.montage)
        positions = montage.get_positions()["ch_pos"]
        missing = [name for name in settings.channels if name not in positions]
        if missing:
            raise ValueError(f"Channels missing from montage {settings.montage}: {missing}")
        return np.vstack([positions[name] for name in settings.channels]), "metres"

    with settings.coordinate_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"label", "x_mm", "y_mm", "z_mm"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Sensor coordinate file must contain {sorted(required)}: "
            f"{settings.coordinate_file}"
        )
    by_label = {row["label"]: row for row in rows}
    if len(by_label) != len(rows):
        raise ValueError("Sensor coordinate labels must be unique")
    missing = [name for name in settings.channels if name not in by_label]
    if missing:
        raise ValueError(f"Channels missing from sensor coordinate file: {missing}")
    xyz = np.asarray(
        [
            [float(by_label[name][axis]) for axis in ("x_mm", "y_mm", "z_mm")]
            for name in settings.channels
        ],
        dtype=float,
    )
    if not np.isfinite(xyz).all():
        raise ValueError("Sensor coordinate file contains non-finite positions")
    return xyz, "millimetres"


def sensor_locations(settings: MonitorConfig) -> np.ndarray:
    """Return unit-vector sensor locations from the configured coordinate source."""

    xyz, _ = _coordinate_values(settings)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("EEG montage contains an invalid zero-length position")
    return xyz / norms


def mne_sensor_montage(settings: MonitorConfig) -> mne.channels.DigMontage:
    """Return an MNE montage consistent with the forward-model coordinates."""

    if settings.coordinate_file is None:
        return mne.channels.make_standard_montage(settings.montage)
    xyz_mm, _ = _coordinate_values(settings)
    ch_pos = {
        name: xyz / 1000.0 for name, xyz in zip(settings.channels, xyz_mm, strict=True)
    }
    return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")


def sensor_coordinate_audit(settings: MonitorConfig) -> dict[str, object]:
    """Describe and compare the configured sensor-coordinate source."""

    xyz, units = _coordinate_values(settings)
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    audit: dict[str, object] = {
        "source": (
            str(settings.coordinate_file)
            if settings.coordinate_file is not None
            else f"MNE montage {settings.montage}"
        ),
        "coordinate_units_in_source": units,
        "normalization_for_tvb": "direction vectors normalized to unit length",
        "coordinate_norm_range": [
            float(np.linalg.norm(xyz, axis=1).min()),
            float(np.linalg.norm(xyz, axis=1).max()),
        ],
        "surface_laplacian_sphere": (
            {
                "origin_m": [0.0, 0.0, 0.0],
                "radius_m": float(np.median(np.linalg.norm(xyz, axis=1)) / 1000.0),
                "definition": "published-coordinate origin and median radius",
            }
            if settings.coordinate_file is not None
            else "MNE automatic fit"
        ),
    }
    if settings.coordinate_file is not None:
        digest = hashlib.sha256(settings.coordinate_file.read_bytes()).hexdigest()
        template = mne.channels.make_standard_montage(settings.montage)
        template_positions = template.get_positions()["ch_pos"]
        template_xyz = np.vstack([template_positions[name] for name in settings.channels])
        template_unit = template_xyz / np.linalg.norm(
            template_xyz, axis=1, keepdims=True
        )
        angles = np.degrees(
            np.arccos(np.clip(np.sum(unit * template_unit, axis=1), -1.0, 1.0))
        )
        audit.update({
            "sha256": digest,
            "comparison_mne_montage": settings.montage,
            "median_angular_difference_from_mne_degrees": float(np.median(angles)),
            "maximum_angular_difference_from_mne_degrees": float(np.max(angles)),
        })
    return audit


def build_eeg_monitor(
    settings: MonitorConfig,
    n_regions: int,
    period_ms: float,
) -> tuple[monitors.EEG, SensorsEEG]:
    """Build TVB's analytic single-sphere EEG monitor for a region simulation."""

    labels = np.asarray(settings.channels, dtype="U16")
    sensors = SensorsEEG(labels=labels, locations=sensor_locations(settings))
    sensors.configure()
    region_mapping = RegionMapping(array_data=np.arange(n_regions, dtype=np.int32))
    eeg_monitor = monitors.EEG(
        period=period_ms,
        sensors=sensors,
        region_mapping=region_mapping,
        reference=settings.reference,
        variables_of_interest=np.array([1, 2], dtype=np.int64),
        # None is required to disable observation noise. In this TVB monitor
        # path, Noise.generate() is called directly rather than through gfun(),
        # so an Additive object with nsig=0 would still generate samples.
        obsnoise=None,
    )
    return eeg_monitor, sensors


def regularize_analytic_eeg_gain(
    eeg_monitor: monitors.EEG,
    source_centres_mm: np.ndarray,
    source_orientations: np.ndarray,
    minimum_distance_mm: float,
) -> dict[str, float | int]:
    """Recompute TVB's one-sphere gain with a finite near-field distance.

    Parcel centroids are coarse point sources rather than cortical vertices. A
    centroid that happens to lie very close to the fitted sensor sphere can
    therefore dominate Equation 12's inverse-square distance term. The floor
    only regularizes those near-field pairs; zero reproduces TVB's formula.
    """

    center = source_centres_mm.mean(axis=0, keepdims=True)
    source_radii = np.linalg.norm(source_centres_mm - center, axis=1)
    sphere_radius = 1.05125 * float(source_radii.max())
    unit_sensors = eeg_monitor.sensors.locations.copy()
    unit_sensors /= np.linalg.norm(unit_sensors, axis=1, keepdims=True)
    sensor_positions = unit_sensors * sphere_radius + center

    displacement = sensor_positions[:, np.newaxis, :] - source_centres_mm[np.newaxis, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    effective_distances = np.maximum(distances, minimum_distance_mm)
    gain = np.sum(
        source_orientations[np.newaxis, :, :]
        * displacement
        / effective_distances[:, :, np.newaxis] ** 3,
        axis=2,
    ) / (4.0 * np.pi * float(eeg_monitor.sigma))
    eeg_monitor.gain = gain
    affected = distances < minimum_distance_mm
    return {
        "minimum_source_sensor_distance_mm": float(minimum_distance_mm),
        "minimum_unregularized_distance_mm": float(distances.min()),
        "regularized_source_sensor_pairs": int(affected.sum()),
        "total_source_sensor_pairs": int(distances.size),
        "maximum_absolute_gain": float(np.max(np.abs(gain))),
    }


def project_regional_psp_to_eeg(
    region_psp: np.ndarray,
    gain: np.ndarray,
    reference: str,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    """Apply a regional gain matrix and the same reference as TVB's EEG monitor."""

    eeg = region_psp @ gain.T
    if reference:
        if reference.lower() == "average":
            eeg -= eeg.mean(axis=1, keepdims=True)
        else:
            try:
                reference_index = channel_names.index(reference)
            except ValueError as error:
                raise ValueError(f"Unknown EEG reference channel: {reference}") from error
            eeg -= eeg[:, reference_index, np.newaxis]
    return eeg


def apply_surface_laplacian(
    eeg: np.ndarray,
    channel_names: tuple[str, ...],
    sfreq_hz: float,
    settings: MonitorConfig,
) -> np.ndarray:
    """Apply MNE's spherical-spline current-source-density transform."""

    info = mne.create_info(list(channel_names), sfreq=sfreq_hz, ch_types="eeg")
    info.set_montage(mne_sensor_montage(settings), on_missing="raise")
    raw = mne.io.RawArray(eeg.T, info, verbose="ERROR")
    if settings.coordinate_file is not None:
        xyz_mm, _ = _coordinate_values(settings)
        sphere: str | tuple[float, float, float, float] = (
            0.0,
            0.0,
            0.0,
            float(np.median(np.linalg.norm(xyz_mm, axis=1)) / 1000.0),
        )
    else:
        sphere = "auto"
    csd = mne.preprocessing.compute_current_source_density(
        raw,
        sphere=sphere,
        lambda2=1e-5,
        stiffness=4,
        n_legendre_terms=50,
        copy=True,
    )
    return csd.get_data().T


def add_colored_observation_noise(
    eeg: np.ndarray,
    sfreq_hz: float,
    fraction: float,
    exponent: float,
    seed: int,
) -> np.ndarray:
    """Add deterministic group-blind sensor background before CSD.

    ``fraction`` is the noise RMS relative to the median channel RMS of the
    simulated neural signal. ``exponent`` controls a 1/f**exponent spectrum.
    This term represents measurement and unmodelled neural background, not a
    fitted neurophysiological parameter.
    """

    data = np.asarray(eeg, dtype=float)
    if fraction == 0:
        return data.copy()
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("EEG must be a finite samples-by-channels array")
    if sfreq_hz <= 0 or fraction < 0 or not 0 <= exponent <= 2:
        raise ValueError("Invalid observation-noise settings")
    rng = np.random.default_rng(seed)
    background = rng.normal(size=data.shape)
    spectrum = np.fft.rfft(background, axis=0)
    frequency = np.fft.rfftfreq(data.shape[0], d=1.0 / sfreq_hz)
    spectrum[0] = 0.0
    if spectrum.shape[0] > 1:
        spectrum[1:] *= frequency[1:, np.newaxis] ** (-exponent / 2.0)
    background = np.fft.irfft(spectrum, n=data.shape[0], axis=0)
    background -= background.mean(axis=0, keepdims=True)
    scale = background.std(axis=0, keepdims=True)
    background /= np.where(scale > 0, scale, 1.0)
    neural_rms = float(np.median(data.std(axis=0, ddof=0)))
    mixed = data + fraction * neural_rms * background
    mixed -= mixed.mean(axis=1, keepdims=True)
    return mixed
