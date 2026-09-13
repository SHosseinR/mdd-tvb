"""Matched empirical/simulated EEG features for M5 fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .fit_config import FeatureConfig


COHERENCE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 2.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)


@dataclass(frozen=True)
class EEGFeatureVector:
    psd_shape: np.ndarray
    alpha_topography: np.ndarray
    coherence: np.ndarray
    frequency_hz: np.ndarray
    alpha_peak_hz: float
    alpha_power_fraction: float
    spectral_entropy: float
    total_log_power: float


@dataclass(frozen=True)
class FeatureCollection:
    subject_ids: np.ndarray
    groups: np.ndarray
    source_files: np.ndarray
    durations_s: np.ndarray
    psd_shape: np.ndarray
    alpha_topography: np.ndarray
    coherence: np.ndarray
    frequency_hz: np.ndarray
    alpha_peak_hz: np.ndarray
    alpha_power_fraction: np.ndarray
    spectral_entropy: np.ndarray
    total_log_power: np.ndarray


@dataclass(frozen=True)
class FeatureTransformer:
    psd_mean: np.ndarray
    psd_scale: np.ndarray
    spectral_summary_mean: np.ndarray
    spectral_summary_scale: np.ndarray
    topography_mean: np.ndarray
    topography_scale: np.ndarray
    coherence_mean: np.ndarray
    coherence_scale: np.ndarray
    coherence_components: np.ndarray
    coherence_score_scale: np.ndarray
    frequency_hz: np.ndarray
    use_mechanistic_spectrum: bool
    coherence_transform: str
    coherence_selected_flat_indices: np.ndarray
    psd_weight: float
    spectral_summary_weight: float
    alpha_topography_weight: float
    coherence_weight: float

    def transform_blocks(
        self,
        psd_shape: np.ndarray,
        alpha_peak_hz: np.ndarray,
        alpha_power_fraction: np.ndarray,
        spectral_entropy: np.ndarray,
        alpha_topography: np.ndarray,
        coherence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        psd = np.atleast_2d(psd_shape)
        topography = np.atleast_2d(alpha_topography)
        coherence_flat = np.asarray(coherence).reshape(len(psd), -1)
        psd_z = (psd - self.psd_mean) / self.psd_scale
        spectral_summary = _spectral_summary_matrix(
            psd,
            np.asarray(alpha_peak_hz),
            np.asarray(alpha_power_fraction),
            np.asarray(spectral_entropy),
            self.frequency_hz,
            self.use_mechanistic_spectrum,
        )
        spectral_summary_z = (
            spectral_summary - self.spectral_summary_mean
        ) / self.spectral_summary_scale
        topography_z = (
            topography - self.topography_mean
        ) / self.topography_scale
        coherence_z = (
            coherence_flat - self.coherence_mean
        ) / self.coherence_scale
        coherence_scores = coherence_z @ self.coherence_components.T
        coherence_scores /= self.coherence_score_scale
        return psd_z, spectral_summary_z, topography_z, coherence_scores

    def transform(
        self,
        psd_shape: np.ndarray,
        alpha_peak_hz: np.ndarray,
        alpha_power_fraction: np.ndarray,
        spectral_entropy: np.ndarray,
        alpha_topography: np.ndarray,
        coherence: np.ndarray,
    ) -> np.ndarray:
        psd, spectral_summary, topography, coherence_scores = self.transform_blocks(
            psd_shape,
            alpha_peak_hz,
            alpha_power_fraction,
            spectral_entropy,
            alpha_topography,
            coherence,
        )
        return np.concatenate(
            (
                psd * np.sqrt(self.psd_weight / psd.shape[1]),
                spectral_summary
                * np.sqrt(
                    self.spectral_summary_weight / spectral_summary.shape[1]
                ),
                topography
                * np.sqrt(self.alpha_topography_weight / topography.shape[1]),
                coherence_scores
                * np.sqrt(self.coherence_weight / coherence_scores.shape[1]),
            ),
            axis=1,
        )


def _safe_scale(values: np.ndarray, axis: int = 0) -> np.ndarray:
    scale = np.std(values, axis=axis, ddof=1)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    floor = float(np.median(positive) * 1e-3) if positive.size else 1.0
    return np.where(np.isfinite(scale) & (scale > floor), scale, floor)


def _spectral_summary_matrix(
    psd_shape: np.ndarray,
    alpha_peak_hz: np.ndarray,
    alpha_power_fraction: np.ndarray,
    spectral_entropy: np.ndarray,
    frequency_hz: np.ndarray,
    use_mechanistic_spectrum: bool,
) -> np.ndarray:
    basic = np.column_stack(
        (alpha_peak_hz, alpha_power_fraction, spectral_entropy)
    )
    if not use_mechanistic_spectrum:
        return basic
    psd = np.atleast_2d(np.asarray(psd_shape, dtype=float))
    frequency = np.asarray(frequency_hz, dtype=float)
    if psd.shape[1] != frequency.size:
        raise ValueError("PSD and transformer frequency grids differ")

    background_mask = (
        ((frequency >= 2.0) & (frequency < 7.0))
        | ((frequency > 14.0) & (frequency <= 45.0))
    )
    log_frequency = np.log10(frequency[background_mask])
    centered_log_frequency = log_frequency - log_frequency.mean()
    exponent = -(
        psd[:, background_mask] @ centered_log_frequency
    ) / float(centered_log_frequency @ centered_log_frequency)

    relative_power = np.exp(psd)
    total = np.trapezoid(relative_power, frequency, axis=1)
    fractions: list[np.ndarray] = []
    for low, high in ((4.0, 8.0), (13.0, 30.0)):
        mask = (frequency >= low) & (frequency < high)
        fractions.append(
            np.trapezoid(relative_power[:, mask], frequency[mask], axis=1)
            / np.maximum(total, np.finfo(float).tiny)
        )

    alpha_mask = (frequency >= 8.0) & (frequency < 13.0)
    shoulder_mask = (
        ((frequency >= 6.0) & (frequency < 8.0))
        | ((frequency >= 13.0) & (frequency < 16.0))
    )
    alpha_values = psd[:, alpha_mask]
    shoulder = np.mean(psd[:, shoulder_mask], axis=1)
    peak_index = np.argmax(alpha_values, axis=1)
    peak_value = alpha_values[np.arange(len(psd)), peak_index]
    prominence = peak_value - shoulder
    spacing = float(np.median(np.diff(frequency)))
    width = np.empty(len(psd), dtype=float)
    for row in range(len(psd)):
        above_half = alpha_values[row] >= shoulder[row] + 0.5 * prominence[row]
        center = int(peak_index[row])
        left = center
        right = center
        while left > 0 and above_half[left - 1]:
            left -= 1
        while right + 1 < above_half.size and above_half[right + 1]:
            right += 1
        width[row] = (right - left + 1) * spacing
    return np.column_stack(
        (basic, exponent, fractions[0], fractions[1], prominence, width)
    )


def extract_eeg_features(
    eeg: np.ndarray,
    sfreq_hz: float,
    settings: FeatureConfig,
) -> EEGFeatureVector:
    """Extract amplitude-invariant spectral, topographic, and coherence features."""

    data = np.asarray(eeg, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("EEG must have shape (samples, channels)")
    if not np.isfinite(data).all():
        raise ValueError("EEG contains non-finite values")
    # Average reference is idempotent and ensures identical treatment of old
    # EEGLAB imports whose reference flag is not reliably retained.
    data = data - data.mean(axis=1, keepdims=True)
    data = data - data.mean(axis=0, keepdims=True)
    nperseg = int(round(settings.epoch_seconds * sfreq_hz))
    if data.shape[0] < 2 * nperseg:
        raise ValueError("EEG is too short for stable M5 feature extraction")

    frequencies, psd_channels = welch(
        data,
        fs=sfreq_hz,
        axis=0,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    frequency_mask = (
        (frequencies >= settings.frequency_min_hz)
        & (frequencies <= settings.frequency_max_hz)
    )
    selected_frequencies = frequencies[frequency_mask]
    mean_psd = np.mean(psd_channels[frequency_mask], axis=1)
    tiny = np.finfo(float).tiny
    psd_shape = np.log(mean_psd + tiny)
    psd_shape -= psd_shape.mean()

    alpha_mask = (frequencies >= 8.0) & (frequencies < 13.0)
    alpha_power = np.trapezoid(
        psd_channels[alpha_mask], frequencies[alpha_mask], axis=0
    )
    alpha_topography = np.log(alpha_power + tiny)
    alpha_topography -= alpha_topography.mean()
    alpha_peak_hz = float(
        frequencies[alpha_mask][np.argmax(np.mean(psd_channels[alpha_mask], axis=1))]
    )
    selected_alpha = (
        (selected_frequencies >= 8.0) & (selected_frequencies < 13.0)
    )
    alpha_power_fraction = float(
        np.trapezoid(
            mean_psd[selected_alpha], selected_frequencies[selected_alpha]
        )
        / max(np.trapezoid(mean_psd, selected_frequencies), tiny)
    )

    probability = mean_psd / mean_psd.sum()
    spectral_entropy = float(
        -(probability * np.log(probability + tiny)).sum()
        / np.log(probability.size)
    )
    total_log_power = float(
        np.log(np.trapezoid(mean_psd, selected_frequencies) + tiny)
    )

    n_epochs = data.shape[0] // nperseg
    segmented = data[: n_epochs * nperseg].reshape(
        n_epochs, nperseg, data.shape[1]
    )
    segmented -= segmented.mean(axis=1, keepdims=True)
    window = np.hanning(nperseg)
    spectra = np.fft.rfft(segmented * window[np.newaxis, :, np.newaxis], axis=1)
    fft_frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sfreq_hz)
    upper = np.triu_indices(data.shape[1], k=1)
    coherence_features: list[np.ndarray] = []
    for _, low, high in COHERENCE_BANDS:
        band = (fft_frequencies >= low) & (fft_frequencies < high)
        selected = spectra[:, band, :]
        cross = np.einsum(
            "efc,efd->cd", np.conjugate(selected), selected, optimize=True
        )
        auto = np.real(np.diag(cross))
        denominator = np.maximum(
            auto[:, np.newaxis] * auto[np.newaxis, :], tiny
        )
        coherence = np.clip(np.abs(cross) ** 2 / denominator, 0.0, 1.0)
        coherence_features.append(coherence[upper])

    return EEGFeatureVector(
        psd_shape=psd_shape,
        alpha_topography=alpha_topography,
        coherence=np.stack(coherence_features),
        frequency_hz=selected_frequencies,
        alpha_peak_hz=alpha_peak_hz,
        alpha_power_fraction=alpha_power_fraction,
        spectral_entropy=spectral_entropy,
        total_log_power=total_log_power,
    )


def fit_feature_transformer(
    collection: FeatureCollection,
    train_indices: np.ndarray,
    settings: FeatureConfig,
    validation_collection: FeatureCollection | None = None,
) -> FeatureTransformer:
    psd = collection.psd_shape[train_indices]
    topography = collection.alpha_topography[train_indices]
    coherence = collection.coherence[train_indices].reshape(len(train_indices), -1)
    psd_mean = psd.mean(axis=0)
    psd_scale = _safe_scale(psd)
    spectral_summary = _spectral_summary_matrix(
        psd,
        collection.alpha_peak_hz[train_indices],
        collection.alpha_power_fraction[train_indices],
        collection.spectral_entropy[train_indices],
        collection.frequency_hz,
        settings.use_mechanistic_spectrum,
    )
    spectral_summary_mean = spectral_summary.mean(axis=0)
    spectral_summary_scale = _safe_scale(spectral_summary)
    topography_mean = topography.mean(axis=0)
    topography_scale = _safe_scale(topography)
    coherence_mean = coherence.mean(axis=0)
    coherence_scale = _safe_scale(coherence)
    coherence_z = (coherence - coherence_mean) / coherence_scale
    selected_indices = np.empty(0, dtype=int)
    if settings.coherence_transform == "reliable_edges":
        if validation_collection is None:
            raise ValueError(
                "reliable_edges coherence transform requires validation halves"
            )
        validation_coherence = validation_collection.coherence[train_indices].reshape(
            len(train_indices), -1
        )
        reliability = np.asarray(
            [
                np.corrcoef(coherence[:, column], validation_coherence[:, column])[0, 1]
                if np.std(coherence[:, column]) > 0
                and np.std(validation_coherence[:, column]) > 0
                else -np.inf
                for column in range(coherence.shape[1])
            ]
        )
        allowed_band_indices = {
            index
            for index, (name, _, _) in enumerate(COHERENCE_BANDS)
            if name in settings.coherence_fit_bands
        }
        allowed = np.asarray(
            [
                column
                for column in range(coherence.shape[1])
                if column // collection.coherence.shape[2] in allowed_band_indices
            ],
            dtype=int,
        )
        order = allowed[np.argsort(reliability[allowed])[::-1]]
        selected_indices = order[
            reliability[order] >= settings.coherence_reliability_threshold
        ][: settings.coherence_max_features]
        if selected_indices.size == 0:
            raise ValueError("No coherence edge passes the reliability threshold")
        components = np.zeros((selected_indices.size, coherence.shape[1]), dtype=float)
        components[np.arange(selected_indices.size), selected_indices] = 1.0
        score_scale = np.ones(selected_indices.size, dtype=float)
    else:
        _, singular, right = np.linalg.svd(coherence_z, full_matrices=False)
        components_count = min(
            settings.coherence_pca_components,
            max(len(train_indices) - 1, 1),
            right.shape[0],
            right.shape[1],
        )
        components = right[:components_count]
        score_scale = singular[:components_count] / np.sqrt(
            max(len(train_indices) - 1, 1)
        )
        positive_score_scale = score_scale[score_scale > 0]
        score_floor = (
            float(np.median(positive_score_scale) * 1e-3)
            if positive_score_scale.size
            else 1.0
        )
        score_scale = np.maximum(score_scale, score_floor)
    return FeatureTransformer(
        psd_mean=psd_mean,
        psd_scale=psd_scale,
        spectral_summary_mean=spectral_summary_mean,
        spectral_summary_scale=spectral_summary_scale,
        topography_mean=topography_mean,
        topography_scale=topography_scale,
        coherence_mean=coherence_mean,
        coherence_scale=coherence_scale,
        coherence_components=components,
        coherence_score_scale=score_scale,
        frequency_hz=collection.frequency_hz.copy(),
        use_mechanistic_spectrum=settings.use_mechanistic_spectrum,
        coherence_transform=settings.coherence_transform,
        coherence_selected_flat_indices=selected_indices,
        psd_weight=settings.psd_weight,
        spectral_summary_weight=settings.spectral_summary_weight,
        alpha_topography_weight=settings.alpha_topography_weight,
        coherence_weight=settings.coherence_weight,
    )


def save_feature_collection(path: Path, collection: FeatureCollection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **collection.__dict__)


def load_feature_collection(path: Path) -> FeatureCollection:
    with np.load(path) as arrays:
        return FeatureCollection(**{name: arrays[name] for name in FeatureCollection.__dataclass_fields__})


def save_feature_transformer(path: Path, transformer: FeatureTransformer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **transformer.__dict__)


def load_feature_transformer(path: Path) -> FeatureTransformer:
    with np.load(path) as arrays:
        values = {
            name: arrays[name]
            for name in FeatureTransformer.__dataclass_fields__
            if name in arrays.files and name not in {
                "psd_weight",
                "spectral_summary_weight",
                "alpha_topography_weight",
                "coherence_weight",
            }
        }
        for name in (
            "psd_weight",
            "spectral_summary_weight",
            "alpha_topography_weight",
            "coherence_weight",
        ):
            values[name] = float(arrays[name])
        values.setdefault("frequency_hz", np.empty(0, dtype=float))
        values.setdefault("use_mechanistic_spectrum", False)
        values.setdefault("coherence_transform", "pca")
        values.setdefault("coherence_selected_flat_indices", np.empty(0, dtype=int))
        values["use_mechanistic_spectrum"] = bool(values["use_mechanistic_spectrum"])
        values["coherence_transform"] = str(values["coherence_transform"])
    return FeatureTransformer(**values)
