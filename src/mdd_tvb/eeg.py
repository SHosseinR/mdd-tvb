"""Template EEG sensors, TVB monitor, and optional surface-Laplacian transform."""

from __future__ import annotations

import numpy as np
import mne
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator import monitors

from .config import MonitorConfig


def sensor_locations(settings: MonitorConfig) -> np.ndarray:
    """Return unit-vector sensor locations from an MNE standard montage."""

    montage = mne.channels.make_standard_montage(settings.montage)
    positions = montage.get_positions()["ch_pos"]
    missing = [name for name in settings.channels if name not in positions]
    if missing:
        raise ValueError(f"Channels missing from montage {settings.montage}: {missing}")
    xyz = np.vstack([positions[name] for name in settings.channels]).astype(float)
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("EEG montage contains an invalid zero-length position")
    return xyz / norms


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
    montage_name: str,
) -> np.ndarray:
    """Apply MNE's spherical-spline current-source-density transform."""

    info = mne.create_info(list(channel_names), sfreq=sfreq_hz, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(montage_name), on_missing="raise")
    raw = mne.io.RawArray(eeg.T, info, verbose="ERROR")
    csd = mne.preprocessing.compute_current_source_density(
        raw,
        lambda2=1e-5,
        stiffness=4,
        n_legendre_terms=50,
        copy=True,
    )
    return csd.get_data().T
