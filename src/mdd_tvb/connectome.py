"""Load, audit, scale, and construct the TVB connectivity datatype."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tvb.datatypes.connectivity import Connectivity

from .config import ConnectivityConfig, PathsConfig


@dataclass(frozen=True)
class ConnectomeBundle:
    connectivity: Connectivity
    raw_counts: np.ndarray
    transformed_weights: np.ndarray
    weights: np.ndarray
    tract_lengths: np.ndarray
    region_labels: np.ndarray
    centres: np.ndarray
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_square_matrix(path: Path, expected_regions: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} matrix not found: {path}")
    matrix = np.loadtxt(path, dtype=float)
    expected = (expected_regions, expected_regions)
    if matrix.shape != expected:
        raise ValueError(f"{name} matrix has shape {matrix.shape}; expected {expected}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} matrix contains non-finite values")
    return matrix


def _load_atlas(path: Path, expected_regions: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Schaefer centroid file not found: {path}")
    table = pd.read_csv(path)
    required = {"ROI Label", "ROI Name", "R", "A", "S"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Centroid file is missing columns: {sorted(missing)}")
    table = table.sort_values("ROI Label")
    expected_ids = np.arange(1, expected_regions + 1)
    if len(table) != expected_regions or not np.array_equal(
        table["ROI Label"].to_numpy(), expected_ids
    ):
        raise ValueError("Centroid ROI labels must be consecutive and match the matrix size")
    labels = table["ROI Name"].astype(str).to_numpy(dtype="U128")
    centres = table[["R", "A", "S"]].to_numpy(dtype=float)
    return labels, centres


def transform_weights(raw_counts: np.ndarray, transform: str, normalization: str) -> tuple[np.ndarray, np.ndarray]:
    """Transform counts, then apply one declared global normalization."""

    if np.any(raw_counts < 0):
        raise ValueError("Streamline counts must be non-negative")
    if transform == "none":
        transformed = raw_counts.copy()
    elif transform == "log1p":
        transformed = np.log1p(raw_counts)
    elif transform == "sqrt":
        transformed = np.sqrt(raw_counts)
    else:
        raise ValueError(f"Unknown weight transform: {transform}")

    if normalization == "none":
        factor = 1.0
    elif normalization == "edge_max":
        factor = float(np.max(np.abs(transformed)))
    elif normalization == "region_sum":
        factor = float(np.max(np.sum(np.abs(transformed), axis=1)))
    elif normalization == "frobenius":
        factor = float(np.linalg.norm(transformed))
    else:
        raise ValueError(f"Unknown connectivity normalization: {normalization}")
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError("Connectivity normalization factor must be positive and finite")
    scaled = transformed / factor
    return transformed, scaled


def _radial_orientations(centres: np.ndarray) -> np.ndarray:
    centered = centres - centres.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Cannot derive radial orientations from coincident centroids")
    return centered / norms


def load_connectome(paths: PathsConfig, settings: ConnectivityConfig) -> ConnectomeBundle:
    """Create a validated 200-region cortical TVB connectivity object."""

    counts = _read_square_matrix(paths.count_matrix, settings.expected_regions, "count")
    lengths = _read_square_matrix(paths.distance_matrix, settings.expected_regions, "distance")
    tol = settings.symmetry_tolerance
    if not np.allclose(counts, counts.T, atol=tol, rtol=0):
        raise ValueError("Count matrix is not symmetric")
    if not np.allclose(lengths, lengths.T, atol=tol, rtol=0):
        raise ValueError("Distance matrix is not symmetric")
    if not np.allclose(np.diag(counts), 0.0, atol=tol, rtol=0):
        raise ValueError("Count matrix diagonal must be zero")
    if not np.allclose(np.diag(lengths), 0.0, atol=tol, rtol=0):
        raise ValueError("Distance matrix diagonal must be zero")
    if np.any(lengths < 0):
        raise ValueError("Tract lengths must be non-negative")

    count_support = counts > 0
    length_support = lengths > 0
    if not np.array_equal(count_support, length_support):
        mismatch = int(np.count_nonzero(count_support != length_support))
        raise ValueError(f"Count and distance support differ at {mismatch} entries")

    labels, centres = _load_atlas(paths.centroids, settings.expected_regions)
    transformed, weights = transform_weights(
        counts, settings.weight_transform, settings.normalization
    )
    orientations = _radial_orientations(centres)
    hemispheres = np.char.find(labels, "_RH_") >= 0
    cortical = np.ones(settings.expected_regions, dtype=bool)

    conn = Connectivity(
        weights=weights,
        tract_lengths=lengths,
        centres=centres,
        region_labels=labels,
        orientations=orientations,
        cortical=cortical,
        hemispheres=hemispheres,
        speed=np.array([settings.speed_mm_per_ms], dtype=float),
    )
    conn.configure()

    n_components, _ = connected_components(csr_matrix(count_support), directed=False)
    positive_counts = counts[count_support]
    positive_lengths = lengths[length_support]
    n = settings.expected_regions
    audit: dict[str, Any] = {
        "regions": n,
        "undirected_edges": int(np.count_nonzero(np.triu(count_support, k=1))),
        "directed_density_excluding_diagonal": float(np.count_nonzero(count_support) / (n * (n - 1))),
        "connected_components": int(n_components),
        "count_min_positive": float(positive_counts.min()),
        "count_median_positive": float(np.median(positive_counts)),
        "count_max": float(positive_counts.max()),
        "tract_length_min_positive_mm": float(positive_lengths.min()),
        "tract_length_median_positive_mm": float(np.median(positive_lengths)),
        "tract_length_max_mm": float(positive_lengths.max()),
        "max_delay_ms": float(positive_lengths.max() / settings.speed_mm_per_ms),
        "weight_transform": settings.weight_transform,
        "normalization": settings.normalization,
        "scaled_weight_max": float(weights.max()),
        "scaled_max_region_input": float(np.max(weights.sum(axis=1))),
        "speed_mm_per_ms": settings.speed_mm_per_ms,
        "atlas_first_label": str(labels[0]),
        "atlas_last_label": str(labels[-1]),
        "count_matrix_sha256": _sha256(paths.count_matrix),
        "distance_matrix_sha256": _sha256(paths.distance_matrix),
        "centroids_sha256": _sha256(paths.centroids),
    }
    return ConnectomeBundle(
        connectivity=conn,
        raw_counts=counts,
        transformed_weights=transformed,
        weights=weights,
        tract_lengths=lengths,
        region_labels=labels,
        centres=centres,
        audit=audit,
    )
