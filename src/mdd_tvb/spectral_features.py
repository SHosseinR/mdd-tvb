"""Frequency-resolved complex EEG cross spectra for spectral M5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .spectral_config import CrossSpectralConfig


@dataclass(frozen=True)
class CrossSpectralCollection:
    subject_ids: np.ndarray
    groups: np.ndarray
    source_files: np.ndarray
    durations_s: np.ndarray
    epoch_counts: np.ndarray
    frequency_hz: np.ndarray
    channel_names: np.ndarray
    csd: np.ndarray


def subset_cross_spectral_collection(
    collection: CrossSpectralCollection, indices: np.ndarray
) -> CrossSpectralCollection:
    """Select subjects while preserving all spectral and sensor axes."""

    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError("Subject indices must be a unique one-dimensional array")
    if len(selected) and (selected.min() < 0 or selected.max() >= len(collection.subject_ids)):
        raise IndexError("Subject index is outside the collection")
    return CrossSpectralCollection(
        subject_ids=collection.subject_ids[selected],
        groups=collection.groups[selected],
        source_files=collection.source_files[selected],
        durations_s=collection.durations_s[selected],
        epoch_counts=collection.epoch_counts[selected],
        frequency_hz=collection.frequency_hz.copy(),
        channel_names=collection.channel_names.copy(),
        csd=collection.csd[selected],
    )


@dataclass(frozen=True)
class SpectralFeatureTransformer:
    sensor_basis: np.ndarray
    frequency_hz: np.ndarray
    auto_mean: np.ndarray
    auto_scale: np.ndarray
    auto_components: np.ndarray
    auto_score_scale: np.ndarray
    cross_mean: np.ndarray
    cross_scale: np.ndarray
    cross_components: np.ndarray
    cross_score_scale: np.ndarray
    topography_mean: np.ndarray
    topography_scale: np.ndarray
    topography_components: np.ndarray
    topography_score_scale: np.ndarray
    cross_metric: str
    diagonal_shrinkage: float
    auto_weight: float
    cross_weight: float
    topography_weight: float

    def _raw_blocks(
        self, csd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrices = np.asarray(csd, dtype=np.complex128)
        if matrices.ndim == 3:
            matrices = matrices[np.newaxis]
        if matrices.ndim != 4:
            raise ValueError("CSD must have shape (samples, frequencies, channels, channels)")
        if matrices.shape[1] != self.frequency_hz.size:
            raise ValueError("CSD frequency grid differs from the fitted transformer")
        channel_diagonal = np.maximum(
            np.real(np.diagonal(matrices, axis1=2, axis2=3)),
            np.finfo(float).tiny,
        )
        alpha = (self.frequency_hz >= 8.0) & (self.frequency_hz < 13.0)
        alpha_topography = np.log(channel_diagonal[:, alpha].mean(axis=1))
        # Remove the unknown subject/electrode gain while retaining the scalp
        # pattern that is relevant to occipital alpha and stimulation targets.
        alpha_topography -= alpha_topography.mean(axis=1, keepdims=True)

        projected = np.einsum(
            "ci,nfcd,dj->nfij",
            self.sensor_basis,
            matrices,
            self.sensor_basis,
            optimize=True,
        )
        diagonal = np.real(np.diagonal(projected, axis1=2, axis2=3))
        diagonal = np.maximum(diagonal, np.finfo(float).tiny)
        diagonal_matrix = np.zeros_like(projected)
        indices = np.arange(projected.shape[-1])
        diagonal_matrix[:, :, indices, indices] = diagonal
        projected = (
            (1.0 - self.diagonal_shrinkage) * projected
            + self.diagonal_shrinkage * diagonal_matrix
        )

        auto = np.log(diagonal)
        # Separate the aperiodic background from oscillatory residuals. The
        # regression uses only 2-7 and 30-40 Hz, so alpha/beta activity does
        # not define its own baseline. Per-mode intercepts are nuisance gains;
        # the exponents remain explicit fitted summaries.
        background = (self.frequency_hz <= 7.0) | (self.frequency_hz >= 30.0)
        log_frequency = np.log(self.frequency_hz[background])
        centered_frequency = log_frequency - log_frequency.mean()
        selected = auto[:, background, :]
        intercept = selected.mean(axis=1, keepdims=True)
        slope = np.einsum(
            "nfm,f->nm", selected - intercept, centered_frequency, optimize=True
        ) / float(centered_frequency @ centered_frequency)
        fitted_background = intercept + slope[:, np.newaxis, :] * (
            np.log(self.frequency_hz) - log_frequency.mean()
        )[np.newaxis, :, np.newaxis]
        periodic_residual = auto - fitted_background
        exponent = -slope
        auto_flat = np.concatenate(
            (periodic_residual.reshape(len(auto), -1), exponent), axis=1
        )

        denominator = np.sqrt(
            diagonal[:, :, :, np.newaxis] * diagonal[:, :, np.newaxis, :]
        )
        coherency = projected / np.maximum(denominator, np.finfo(float).tiny)
        upper = np.triu_indices(projected.shape[-1], k=1)
        cross = coherency[:, :, upper[0], upper[1]]
        if self.cross_metric == "complex_coherency":
            cross_flat = np.concatenate(
                (
                    np.real(cross).reshape(len(cross), -1),
                    np.imag(cross).reshape(len(cross), -1),
                ),
                axis=1,
            )
        elif self.cross_metric == "imaginary_coherency":
            # Signed imaginary coherency rejects instantaneous mixing from a
            # common volume conductor while retaining phase-lag direction.
            cross_flat = np.imag(cross).reshape(len(cross), -1)
        elif self.cross_metric == "lagged_coherency":
            # Pascual-Marqui-style lagged normalization removes the portion
            # explainable by zero-lag real coherency.  It is not wPLI: wPLI
            # requires the epoch-wise cross-periodogram distribution, which is
            # intentionally not fabricated from an averaged CSD.
            real = np.real(cross)
            denominator = np.sqrt(np.maximum(1.0 - real**2, 1e-8))
            cross_flat = (np.imag(cross) / denominator).reshape(len(cross), -1)
        else:
            raise ValueError(f"Unknown cross metric: {self.cross_metric}")
        return auto_flat, cross_flat, alpha_topography

    def transform_blocks(
        self, csd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        auto, cross, topography = self._raw_blocks(csd)
        auto_standard = (auto - self.auto_mean) / self.auto_scale
        cross_standard = (cross - self.cross_mean) / self.cross_scale
        topography_standard = (
            topography - self.topography_mean
        ) / self.topography_scale
        auto_score = (
            auto_standard @ self.auto_components.T
            if self.auto_components.shape[0]
            else auto_standard
        )
        cross_score = (
            cross_standard @ self.cross_components.T
            if self.cross_components.shape[0]
            else cross_standard
        )
        topography_score = (
            topography_standard @ self.topography_components.T
            if self.topography_components.shape[0]
            else topography_standard
        )
        if self.auto_components.shape[0]:
            auto_score /= self.auto_score_scale
        if self.cross_components.shape[0]:
            cross_score /= self.cross_score_scale
        if self.topography_components.shape[0]:
            topography_score /= self.topography_score_scale
        return auto_score, cross_score, topography_score

    def transform(self, csd: np.ndarray) -> np.ndarray:
        auto, cross, topography = self.transform_blocks(csd)
        return np.concatenate(
            (
                auto * np.sqrt(self.auto_weight / auto.shape[1]),
                cross * np.sqrt(self.cross_weight / cross.shape[1]),
                topography
                * np.sqrt(self.topography_weight / topography.shape[1]),
            ),
            axis=1,
        )


def estimate_cross_spectrum(
    eeg: np.ndarray,
    sfreq_hz: float,
    settings: CrossSpectralConfig,
    starts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate a binned complex CSD using overlapped Hann epochs.

    ``starts`` optionally restricts the estimate to epochs beginning at these
    samples (for example only artefact-free epochs); by default every epoch on
    the half-overlapping grid is used.
    """

    data = np.asarray(eeg, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError("EEG must be a finite (samples, channels) array")
    data = data - data.mean(axis=1, keepdims=True)
    data = data - data.mean(axis=0, keepdims=True)
    nperseg = int(round(settings.epoch_seconds * sfreq_hz))
    step = nperseg // 2
    if starts is None:
        if data.shape[0] < nperseg * 2:
            raise ValueError("EEG segment is too short for stable cross spectra")
        starts = np.arange(0, data.shape[0] - nperseg + 1, step, dtype=int)
    else:
        starts = np.asarray(starts, dtype=int)
        if len(starts) < 2 or starts.min() < 0 or starts.max() + nperseg > data.shape[0]:
            raise ValueError("Epoch starts must give at least two epochs inside the data")
    window = np.hanning(nperseg)
    spectra = np.stack(
        [
            np.fft.rfft(
                (data[start : start + nperseg] - data[start : start + nperseg].mean(axis=0))
                * window[:, np.newaxis],
                axis=0,
            )
            for start in starts
        ]
    )
    fft_frequency = np.fft.rfftfreq(nperseg, d=1.0 / sfreq_hz)
    scale = 2.0 / (sfreq_hz * np.sum(window**2))
    raw_csd = (
        np.einsum("efc,efd->fcd", np.conjugate(spectra), spectra, optimize=True)
        * scale
        / len(starts)
    )
    bin_width = settings.frequency_bin_hz
    first = np.ceil(settings.frequency_min_hz / bin_width) * bin_width
    centers = np.arange(first, settings.frequency_max_hz + bin_width * 0.25, bin_width)
    binned: list[np.ndarray] = []
    for center in centers:
        mask = (fft_frequency >= center - bin_width / 2.0) & (
            fft_frequency < center + bin_width / 2.0
        )
        if not np.any(mask):
            raise ValueError(f"No FFT values fall in the {center:g}-Hz bin")
        value = raw_csd[mask].mean(axis=0)
        binned.append(0.5 * (value + np.conjugate(value.T)))
    return centers, np.stack(binned), int(len(starts))


def add_diagonal_observation_noise(
    csd: np.ndarray,
    frequency_hz: np.ndarray,
    fraction: float,
    exponent: float,
) -> np.ndarray:
    """Add a white/pink diagonal noise floor as an observation nuisance."""

    matrices = np.asarray(csd, dtype=np.complex128)
    if fraction == 0.0:
        return matrices.copy()
    diagonal = np.real(np.diagonal(matrices, axis1=-2, axis2=-1))
    mean_signal = np.maximum(np.mean(diagonal, axis=(-2, -1)), np.finfo(float).tiny)
    shape = np.asarray(frequency_hz, dtype=float) ** (-float(exponent))
    shape /= shape.mean()
    ratio = fraction / (1.0 - fraction)
    leading = (1,) * (matrices.ndim - 3)
    noise_power = (
        mean_signal[..., np.newaxis] * ratio * shape.reshape(leading + (shape.size,))
    )
    result = matrices.copy()
    indices = np.arange(matrices.shape[-1])
    result[..., indices, indices] += noise_power[..., np.newaxis]
    return result


def add_observation_backgrounds(
    csd: np.ndarray,
    frequency_hz: np.ndarray,
    diagonal_fraction: float,
    diagonal_exponent: float,
    source_fraction: float,
    source_exponent: float,
    source_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add sensor and lead-field-projected independent-source backgrounds.

    Both components are scaled against the original simulated signal power.
    The source covariance is fixed and group blind; only its fraction and 1/f
    exponent are nuisance states. Returned compact increments allow posterior
    CSD reconstruction without storing one dense CSD per nuisance state.
    """

    matrices = np.asarray(csd, dtype=np.complex128)
    covariance = np.asarray(source_covariance, dtype=float)
    if covariance.shape != matrices.shape[-2:]:
        raise ValueError("source covariance must match the CSD channel dimensions")
    covariance = 0.5 * (covariance + covariance.T)
    mean_diagonal = float(np.mean(np.diag(covariance)))
    if not np.isfinite(covariance).all() or mean_diagonal <= 0.0:
        raise ValueError("source covariance must be finite with positive diagonal power")
    covariance = covariance / mean_diagonal
    if any(
        not 0.0 <= fraction < 1.0
        for fraction in (diagonal_fraction, source_fraction)
    ) or min(diagonal_exponent, source_exponent) < 0.0:
        raise ValueError("invalid observation-background settings")

    diagonal = np.real(np.diagonal(matrices, axis1=-2, axis2=-1))
    mean_signal = np.maximum(
        np.mean(diagonal, axis=(-2, -1)), np.finfo(float).tiny
    )
    leading = (1,) * (matrices.ndim - 3)

    def spectral_power(fraction: float, exponent: float) -> np.ndarray:
        shape = np.asarray(frequency_hz, dtype=float) ** (-float(exponent))
        shape /= shape.mean()
        ratio = fraction / (1.0 - fraction) if fraction else 0.0
        return mean_signal[..., np.newaxis] * ratio * shape.reshape(
            leading + (shape.size,)
        )

    diagonal_power = spectral_power(diagonal_fraction, diagonal_exponent)
    source_power = spectral_power(source_fraction, source_exponent)
    result = matrices.copy()
    indices = np.arange(matrices.shape[-1])
    result[..., indices, indices] += diagonal_power[..., np.newaxis]
    result += source_power[..., np.newaxis, np.newaxis] * covariance
    diagonal_increment = np.broadcast_to(
        diagonal_power[..., np.newaxis], diagonal.shape
    ).copy()
    return result, diagonal_increment, source_power


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    valid = scale[np.isfinite(scale) & (scale > 0)]
    floor = float(np.percentile(valid, 10) * 0.1) if valid.size else 1.0
    return np.where(np.isfinite(scale) & (scale > floor), scale, floor)


def _pca(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        # Empty components are an explicit instruction to retain every
        # standardized CSD coordinate. This also avoids materializing a large
        # identity matrix for the complex cross-spectral block.
        return np.empty((0, values.shape[1])), np.empty(0)
    _, _, right = np.linalg.svd(values, full_matrices=False)
    count = min(count, max(len(values) - 1, 1), right.shape[0])
    components = right[:count]
    scores = values @ components.T
    score_scale = _safe_scale(scores)
    return components, score_scale


def _reliability(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x = first - first.mean(axis=0, keepdims=True)
    y = second - second.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(x**2, axis=0) * np.sum(y**2, axis=0))
    return np.divide(
        np.sum(x * y, axis=0),
        denominator,
        out=np.full(x.shape[1], -np.inf),
        where=denominator > 0,
    )


def _reliable_coordinate_components(
    first: np.ndarray,
    second: np.ndarray,
    threshold: float,
    maximum: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = _reliability(first, second)
    order = np.argsort(values)[::-1]
    selected = order[values[order] >= threshold][:maximum]
    if selected.size < min(minimum, values.size):
        selected = order[: min(minimum, maximum, values.size)]
    components = np.zeros((selected.size, values.size), dtype=float)
    components[np.arange(selected.size), selected] = 1.0
    return components, values[selected]


def _balanced_complex_components(
    first: np.ndarray,
    second: np.ndarray,
    threshold: float,
    maximum: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    if first.shape[1] % 2:
        raise ValueError("Complex feature block must contain equal real/imaginary halves")
    half = first.shape[1] // 2
    per_half_maximum = max(maximum // 2, 1)
    per_half_minimum = max(minimum // 2, 1)
    selected_parts: list[np.ndarray] = []
    reliability_parts: list[np.ndarray] = []
    for offset, stop in ((0, half), (half, first.shape[1])):
        local = _reliability(first[:, offset:stop], second[:, offset:stop])
        order = np.argsort(local)[::-1]
        selected = order[local[order] >= threshold][:per_half_maximum]
        if selected.size < min(per_half_minimum, local.size):
            selected = order[: min(per_half_minimum, per_half_maximum, local.size)]
        selected_parts.append(selected + offset)
        reliability_parts.append(local[selected])
    selected = np.concatenate(selected_parts)
    components = np.zeros((selected.size, first.shape[1]), dtype=float)
    components[np.arange(selected.size), selected] = 1.0
    return components, np.concatenate(reliability_parts)


def fit_spectral_transformer(
    collection: CrossSpectralCollection,
    train_indices: np.ndarray,
    settings: CrossSpectralConfig,
    reliability_first: CrossSpectralCollection | None = None,
    reliability_second: CrossSpectralCollection | None = None,
    sensor_basis: np.ndarray | None = None,
) -> SpectralFeatureTransformer:
    """Fit all dimensionality reduction without using diagnosis labels."""

    train_csd = collection.csd[train_indices]
    if sensor_basis is None:
        trace = np.real(np.trace(train_csd, axis1=2, axis2=3)).mean(axis=1)
        normalized = train_csd / np.maximum(trace[:, np.newaxis, np.newaxis, np.newaxis], np.finfo(float).tiny)
        pooled = np.real(normalized.mean(axis=(0, 1)))
        pooled = 0.5 * (pooled + pooled.T)
        eigenvalues, eigenvectors = np.linalg.eigh(pooled)
        order = np.argsort(eigenvalues)[::-1]
        basis = eigenvectors[:, order[: settings.sensor_modes]]
    else:
        basis = np.asarray(sensor_basis, dtype=float).copy()
        expected = (collection.channel_names.size, settings.sensor_modes)
        if basis.shape != expected or not np.isfinite(basis).all():
            raise ValueError(f"sensor_basis must be finite with shape {expected}")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(settings.sensor_modes), atol=1e-6):
            raise ValueError("sensor_basis columns must be orthonormal")
    # Eigenvector signs are otherwise platform-dependent and would make saved
    # complex coherency features difficult to compare across machines.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0

    shell = SpectralFeatureTransformer(
        sensor_basis=basis,
        frequency_hz=collection.frequency_hz.copy(),
        auto_mean=np.zeros(settings.sensor_modes * (collection.frequency_hz.size + 1)),
        auto_scale=np.ones(settings.sensor_modes * (collection.frequency_hz.size + 1)),
        auto_components=np.empty((0, settings.sensor_modes * (collection.frequency_hz.size + 1))),
        auto_score_scale=np.empty(0),
        cross_mean=np.zeros(
            (2 if settings.cross_metric == "complex_coherency" else 1)
            * collection.frequency_hz.size
            * settings.sensor_modes
            * (settings.sensor_modes - 1)
            // 2
        ),
        cross_scale=np.ones(
            (2 if settings.cross_metric == "complex_coherency" else 1)
            * collection.frequency_hz.size
            * settings.sensor_modes
            * (settings.sensor_modes - 1)
            // 2
        ),
        cross_components=np.empty(
            (
                0,
                (2 if settings.cross_metric == "complex_coherency" else 1)
                * collection.frequency_hz.size
                * settings.sensor_modes
                * (settings.sensor_modes - 1)
                // 2,
            )
        ),
        cross_score_scale=np.empty(0),
        topography_mean=np.zeros(collection.channel_names.size),
        topography_scale=np.ones(collection.channel_names.size),
        topography_components=np.empty((0, collection.channel_names.size)),
        topography_score_scale=np.empty(0),
        cross_metric=settings.cross_metric,
        diagonal_shrinkage=settings.diagonal_shrinkage,
        auto_weight=settings.auto_weight,
        cross_weight=settings.cross_weight,
        topography_weight=settings.topography_weight,
    )
    auto, cross, topography = shell._raw_blocks(train_csd)
    auto_mean = auto.mean(axis=0)
    auto_scale = _safe_scale(auto)
    cross_mean = cross.mean(axis=0)
    cross_scale = _safe_scale(cross)
    topography_mean = topography.mean(axis=0)
    topography_scale = _safe_scale(topography)
    auto_standard = (auto - auto_mean) / auto_scale
    cross_standard = (cross - cross_mean) / cross_scale
    topography_standard = (topography - topography_mean) / topography_scale
    if (reliability_first is None) != (reliability_second is None):
        raise ValueError("Both reliability calibration collections are required together")
    if reliability_first is not None and reliability_second is not None:
        if not np.array_equal(collection.subject_ids, reliability_first.subject_ids) or not np.array_equal(
            collection.subject_ids, reliability_second.subject_ids
        ):
            raise ValueError("Reliability calibration subject order differs")
        first_auto, first_cross, first_topography = shell._raw_blocks(
            reliability_first.csd[train_indices]
        )
        second_auto, second_cross, second_topography = shell._raw_blocks(
            reliability_second.csd[train_indices]
        )
        first_auto = (first_auto - auto_mean) / auto_scale
        second_auto = (second_auto - auto_mean) / auto_scale
        first_cross = (first_cross - cross_mean) / cross_scale
        second_cross = (second_cross - cross_mean) / cross_scale
        first_topography = (
            first_topography - topography_mean
        ) / topography_scale
        second_topography = (
            second_topography - topography_mean
        ) / topography_scale
        auto_components, auto_reliability = _reliable_coordinate_components(
            first_auto,
            second_auto,
            settings.reliability_threshold,
            settings.reliability_max_auto_coordinates,
            settings.reliability_min_coordinates,
        )
        if settings.cross_metric == "complex_coherency":
            cross_components, cross_reliability = _balanced_complex_components(
                first_cross,
                second_cross,
                settings.reliability_threshold,
                settings.reliability_max_cross_coordinates,
                settings.reliability_min_coordinates,
            )
        else:
            cross_components, cross_reliability = _reliable_coordinate_components(
                first_cross,
                second_cross,
                settings.reliability_threshold,
                settings.reliability_max_cross_coordinates,
                settings.reliability_min_coordinates,
            )
        topography_components, topography_reliability = (
            _reliable_coordinate_components(
                first_topography,
                second_topography,
                settings.reliability_threshold,
                settings.reliability_max_topography_coordinates,
                settings.reliability_min_coordinates,
            )
        )
        # Reliability is used for selection, not again as a weight. Saved
        # score scales retain the selected coefficients for auditing.
        auto_score_scale = np.ones(auto_components.shape[0])
        cross_score_scale = np.ones(cross_components.shape[0])
        topography_score_scale = np.ones(topography_components.shape[0])
    else:
        auto_components, auto_score_scale = _pca(auto_standard, settings.auto_components)
        cross_components, cross_score_scale = _pca(cross_standard, settings.cross_components)
        topography_components = np.empty((0, topography.shape[1]))
        topography_score_scale = np.empty(0)
    return SpectralFeatureTransformer(
        sensor_basis=basis,
        frequency_hz=collection.frequency_hz.copy(),
        auto_mean=auto_mean,
        auto_scale=auto_scale,
        auto_components=auto_components,
        auto_score_scale=auto_score_scale,
        cross_mean=cross_mean,
        cross_scale=cross_scale,
        cross_components=cross_components,
        cross_score_scale=cross_score_scale,
        topography_mean=topography_mean,
        topography_scale=topography_scale,
        topography_components=topography_components,
        topography_score_scale=topography_score_scale,
        cross_metric=settings.cross_metric,
        diagonal_shrinkage=settings.diagonal_shrinkage,
        auto_weight=settings.auto_weight,
        cross_weight=settings.cross_weight,
        topography_weight=settings.topography_weight,
    )


def save_cross_spectral_collection(path: Path, collection: CrossSpectralCollection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **collection.__dict__)


def load_cross_spectral_collection(path: Path) -> CrossSpectralCollection:
    with np.load(path) as arrays:
        return CrossSpectralCollection(
            **{name: arrays[name] for name in CrossSpectralCollection.__dataclass_fields__}
        )


def save_spectral_transformer(path: Path, transformer: SpectralFeatureTransformer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **transformer.__dict__)


def load_spectral_transformer(path: Path) -> SpectralFeatureTransformer:
    with np.load(path) as arrays:
        values = {
            name: arrays[name]
            for name in SpectralFeatureTransformer.__dataclass_fields__
            if name in arrays
            and name
            not in {
                "cross_metric",
                "diagonal_shrinkage",
                "auto_weight",
                "cross_weight",
                "topography_weight",
            }
        }
        for name in (
            "diagonal_shrinkage",
            "auto_weight",
            "cross_weight",
            "topography_weight",
        ):
            values[name] = float(arrays[name])
        values["cross_metric"] = (
            str(arrays["cross_metric"])
            if "cross_metric" in arrays
            else "complex_coherency"
        )
    return SpectralFeatureTransformer(**values)
