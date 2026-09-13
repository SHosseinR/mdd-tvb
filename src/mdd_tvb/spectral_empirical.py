"""TDBRAIN extraction for the complex cross-spectral M5 pipeline."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .empirical import load_empirical_eeg
from .spectral_config import SpectralM5Config
from .spectral_features import (
    CrossSpectralCollection,
    estimate_cross_spectrum,
    save_cross_spectral_collection,
)


def discover_spectral_eeg_files(
    config: SpectralM5Config,
    max_subjects_per_group: int | None = None,
) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for group in config.empirical.groups:
        selected = sorted((config.paths.dataset_root / group).glob("*/*.set"))
        if max_subjects_per_group is not None:
            selected = selected[:max_subjects_per_group]
        files.extend((path, group) for path in selected)
    if not files:
        raise FileNotFoundError(f"No grouped EEGLAB files found below {config.paths.dataset_root}")
    return files


def _extract_one(
    source_file: str,
    group: str,
    baseline_config_path: str,
    config: SpectralM5Config,
) -> dict[str, Any]:
    eeg, sfreq_hz = load_empirical_eeg(
        source_file,
        baseline_config_path,
        config.empirical.max_duration_s,
        config.empirical.apply_surface_laplacian,
    )
    midpoint = eeg.shape[0] // 2
    fit_frequency, fit_csd, fit_epochs = estimate_cross_spectrum(
        eeg[:midpoint], sfreq_hz, config.spectral
    )
    calibration_midpoint = midpoint // 2
    _, reliability_a_csd, reliability_a_epochs = estimate_cross_spectrum(
        eeg[:calibration_midpoint], sfreq_hz, config.spectral
    )
    _, reliability_b_csd, reliability_b_epochs = estimate_cross_spectrum(
        eeg[calibration_midpoint:midpoint], sfreq_hz, config.spectral
    )
    validation_frequency, validation_csd, validation_epochs = estimate_cross_spectrum(
        eeg[midpoint:], sfreq_hz, config.spectral
    )
    if not np.array_equal(fit_frequency, validation_frequency):
        raise RuntimeError("Fitting and validation frequency grids differ")
    path = Path(source_file)
    return {
        "subject_id": path.parent.name,
        "group": group,
        "source_file": str(path.resolve()),
        "duration_s": eeg.shape[0] / sfreq_hz,
        "frequency_hz": fit_frequency,
        "fit_csd": fit_csd,
        "validation_csd": validation_csd,
        "reliability_a_csd": reliability_a_csd,
        "reliability_b_csd": reliability_b_csd,
        "fit_epochs": fit_epochs,
        "validation_epochs": validation_epochs,
        "reliability_a_epochs": reliability_a_epochs,
        "reliability_b_epochs": reliability_b_epochs,
    }


def extract_cross_spectral_collection(
    config: SpectralM5Config,
    max_subjects_per_group: int | None = None,
) -> tuple[CrossSpectralCollection, CrossSpectralCollection]:
    tasks = discover_spectral_eeg_files(config, max_subjects_per_group)
    arguments = [
        (str(path), group, str(config.paths.baseline_config), config)
        for path, group in tasks
    ]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if config.empirical.n_jobs == 1:
        for index, values in enumerate(arguments, start=1):
            try:
                rows.append(_extract_one(*values))
            except Exception as error:
                failures.append({"source_file": values[0], "error": repr(error)})
            if index % 10 == 0 or index == len(arguments):
                print(f"Complex empirical spectra: {index}/{len(arguments)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.empirical.n_jobs) as pool:
            futures = {pool.submit(_extract_one, *values): values[0] for values in arguments}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    rows.append(future.result())
                except Exception as error:
                    failures.append({"source_file": futures[future], "error": repr(error)})
                if index % 10 == 0 or index == len(futures):
                    print(f"Complex empirical spectra: {index}/{len(futures)}", flush=True)

    output = config.paths.output_dir / "empirical"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["source_file", "error"]).to_csv(
        output / "extraction_failures.csv", index=False
    )
    if failures:
        raise RuntimeError(f"Cross-spectral extraction failed for {len(failures)} recordings")
    rows.sort(key=lambda row: (row["group"], row["subject_id"]))
    frequency = rows[0]["frequency_hz"]
    if any(not np.array_equal(row["frequency_hz"], frequency) for row in rows):
        raise RuntimeError("Empirical cross-spectral frequency grids differ")
    channels = np.asarray(load_config(config.paths.baseline_config).monitor.channels, dtype="U16")

    def collection(prefix: str) -> CrossSpectralCollection:
        duration_divisor = 4.0 if prefix.startswith("reliability_") else 2.0
        return CrossSpectralCollection(
            subject_ids=np.asarray([row["subject_id"] for row in rows], dtype="U32"),
            groups=np.asarray([row["group"] for row in rows], dtype="U32"),
            source_files=np.asarray([row["source_file"] for row in rows], dtype="U512"),
            durations_s=np.asarray([row["duration_s"] / duration_divisor for row in rows]),
            epoch_counts=np.asarray([row[f"{prefix}_epochs"] for row in rows], dtype=int),
            frequency_hz=frequency.copy(),
            channel_names=channels.copy(),
            csd=np.stack([row[f"{prefix}_csd"] for row in rows]),
        )

    fitting = collection("fit")
    validation = collection("validation")
    reliability_a = collection("reliability_a")
    reliability_b = collection("reliability_b")
    save_cross_spectral_collection(output / "cross_spectra_fit.npz", fitting)
    save_cross_spectral_collection(output / "cross_spectra_validation.npz", validation)
    save_cross_spectral_collection(output / "cross_spectra_reliability_a.npz", reliability_a)
    save_cross_spectral_collection(output / "cross_spectra_reliability_b.npz", reliability_b)
    pd.DataFrame(
        {
            "subject_id": fitting.subject_ids,
            "group": fitting.groups,
            "source_file": fitting.source_files,
            "total_duration_s": fitting.durations_s + validation.durations_s,
            "fit_epochs": fitting.epoch_counts,
            "validation_epochs": validation.epoch_counts,
            "reliability_a_epochs": reliability_a.epoch_counts,
            "reliability_b_epochs": reliability_b.epoch_counts,
        }
    ).to_csv(output / "subjects.csv", index=False)
    return fitting, validation
