"""TDBRAIN ingestion and feature extraction for M5."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .config import load_config
from .eeg import apply_surface_laplacian
from .features import (
    EEGFeatureVector,
    FeatureCollection,
    extract_eeg_features,
    save_feature_collection,
)
from .fit_config import FeatureConfig, M5Config


def load_empirical_eeg(
    source_file: str,
    baseline_config_path: str,
    max_duration_s: float,
    use_surface_laplacian: bool,
) -> tuple[np.ndarray, float]:
    """Load one recording with the exact observation preprocessing used by M5."""

    baseline = load_config(baseline_config_path)
    path = Path(source_file)
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    missing = [name for name in baseline.monitor.channels if name not in raw.ch_names]
    if missing:
        raise ValueError(f"missing expected channels: {missing}")
    picks = [raw.ch_names.index(name) for name in baseline.monitor.channels]
    sfreq_hz = float(raw.info["sfreq"])
    maximum_samples = min(
        raw.n_times, int(round(max_duration_s * sfreq_hz))
    )
    eeg = raw.get_data(picks=picks, start=0, stop=maximum_samples).T
    eeg -= eeg.mean(axis=1, keepdims=True)
    if use_surface_laplacian:
        eeg = apply_surface_laplacian(
            eeg, baseline.monitor.channels, sfreq_hz, baseline.monitor
        )
    return eeg, sfreq_hz


def _extract_one(
    source_file: str,
    group: str,
    baseline_config_path: str,
    feature_settings: FeatureConfig,
    max_duration_s: float,
    use_surface_laplacian: bool,
) -> dict[str, Any]:
    path = Path(source_file)
    eeg, sfreq_hz = load_empirical_eeg(
        source_file,
        baseline_config_path,
        max_duration_s,
        use_surface_laplacian,
    )
    # A temporal split gives every subject an honest unseen-recording check.
    # The split is made only after all linear preprocessing so both halves use
    # exactly the same observation model.
    midpoint = eeg.shape[0] // 2
    fit_features: EEGFeatureVector = extract_eeg_features(
        eeg[:midpoint], sfreq_hz, feature_settings
    )
    validation_features: EEGFeatureVector = extract_eeg_features(
        eeg[midpoint:], sfreq_hz, feature_settings
    )
    row: dict[str, Any] = {
        "subject_id": path.parent.name,
        "group": group,
        "source_file": str(path.resolve()),
        "duration_s": eeg.shape[0] / sfreq_hz,
        "sfreq_hz": sfreq_hz,
    }
    for prefix, features in (
        ("fit", fit_features),
        ("validation", validation_features),
    ):
        for name in EEGFeatureVector.__dataclass_fields__:
            row[f"{prefix}_{name}"] = getattr(features, name)
    return row


def discover_empirical_files(
    config: M5Config,
    max_subjects_per_group: int | None = None,
) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for group in config.empirical.groups:
        group_files = sorted((config.paths.dataset_root / group).glob("*/*.set"))
        if max_subjects_per_group is not None:
            group_files = group_files[:max_subjects_per_group]
        files.extend((path, group) for path in group_files)
    if not files:
        raise FileNotFoundError(
            f"No grouped EEGLAB files found below {config.paths.dataset_root}"
        )
    return files


def extract_empirical_collection(
    config: M5Config,
    max_subjects_per_group: int | None = None,
) -> FeatureCollection:
    tasks = discover_empirical_files(config, max_subjects_per_group)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    arguments = [
        (
            str(path),
            group,
            str(config.paths.baseline_config),
            config.features,
            config.empirical.max_duration_s,
            config.empirical.apply_surface_laplacian,
        )
        for path, group in tasks
    ]
    if config.empirical.n_jobs == 1:
        for index, values in enumerate(arguments, start=1):
            try:
                rows.append(_extract_one(*values))
            except Exception as error:  # retain a complete audit instead of hiding failures
                failures.append({"source_file": values[0], "error": repr(error)})
            if index % 10 == 0 or index == len(arguments):
                print(f"Empirical features: {index}/{len(arguments)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.empirical.n_jobs) as pool:
            futures = {
                pool.submit(_extract_one, *values): values[0] for values in arguments
            }
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    rows.append(future.result())
                except Exception as error:
                    failures.append({
                        "source_file": futures[future],
                        "error": repr(error),
                    })
                if index % 10 == 0 or index == len(futures):
                    print(f"Empirical features: {index}/{len(futures)}", flush=True)

    output_dir = config.paths.output_dir / "empirical"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["source_file", "error"]).to_csv(
        output_dir / "extraction_failures.csv", index=False
    )
    if failures:
        raise RuntimeError(
            f"Feature extraction failed for {len(failures)} files; see "
            f"{output_dir / 'extraction_failures.csv'}"
        )
    rows.sort(key=lambda row: (row["group"], row["subject_id"]))
    def make_collection(prefix: str) -> FeatureCollection:
        frequency = rows[0][f"{prefix}_frequency_hz"]
        if any(
            not np.array_equal(row[f"{prefix}_frequency_hz"], frequency)
            for row in rows
        ):
            raise RuntimeError("Empirical feature frequency grids differ")
        return FeatureCollection(
            subject_ids=np.asarray([row["subject_id"] for row in rows], dtype="U32"),
            groups=np.asarray([row["group"] for row in rows], dtype="U32"),
            source_files=np.asarray(
                [row["source_file"] for row in rows], dtype="U512"
            ),
            durations_s=np.asarray([row["duration_s"] / 2.0 for row in rows]),
            psd_shape=np.stack([row[f"{prefix}_psd_shape"] for row in rows]),
            alpha_topography=np.stack(
                [row[f"{prefix}_alpha_topography"] for row in rows]
            ),
            coherence=np.stack([row[f"{prefix}_coherence"] for row in rows]),
            frequency_hz=frequency,
            alpha_peak_hz=np.asarray(
                [row[f"{prefix}_alpha_peak_hz"] for row in rows]
            ),
            alpha_power_fraction=np.asarray(
                [row[f"{prefix}_alpha_power_fraction"] for row in rows]
            ),
            spectral_entropy=np.asarray(
                [row[f"{prefix}_spectral_entropy"] for row in rows]
            ),
            total_log_power=np.asarray(
                [row[f"{prefix}_total_log_power"] for row in rows]
            ),
        )

    collection = make_collection("fit")
    validation = make_collection("validation")
    save_feature_collection(output_dir / "empirical_features.npz", collection)
    save_feature_collection(
        output_dir / "empirical_validation_features.npz", validation
    )
    pd.DataFrame({
        "subject_id": collection.subject_ids,
        "group": collection.groups,
        "source_file": collection.source_files,
        "total_duration_s": collection.durations_s * 2.0,
        "fit_duration_s": collection.durations_s,
        "fit_alpha_peak_hz": collection.alpha_peak_hz,
        "validation_alpha_peak_hz": validation.alpha_peak_hz,
        "fit_alpha_power_fraction": collection.alpha_power_fraction,
        "validation_alpha_power_fraction": validation.alpha_power_fraction,
        "fit_spectral_entropy_2_45": collection.spectral_entropy,
        "validation_spectral_entropy_2_45": validation.spectral_entropy,
        "total_log_power_nuisance_only": collection.total_log_power,
    }).to_csv(output_dir / "subjects.csv", index=False)
    return collection
