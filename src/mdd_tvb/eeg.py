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
