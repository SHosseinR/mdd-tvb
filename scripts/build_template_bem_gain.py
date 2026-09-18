"""Build and audit the distributed fsaverage BEM gain for M5.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.eeg import build_eeg_monitor, regularize_analytic_eeg_gain
from mdd_tvb.template_bem import build_template_bem_gain, save_template_bem_gain


def _column_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=0)
    denominator = np.linalg.norm(first, axis=0) * np.linalg.norm(second, axis=0)
    return numerator / np.maximum(denominator, np.finfo(float).tiny)


def _upper_correlation(first: np.ndarray, second: np.ndarray) -> float:
    indices = np.triu_indices(first.shape[0], 1)
    return float(np.corrcoef(first[indices], second[indices])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path("C:/Users/hosei/mne_data/MNE-fsaverage-data"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/m52_template_bem")
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    config = load_config(root / "configs" / "baseline.toml")
    connectome = load_connectome(config.paths, config.connectivity)
    gain, metadata, region_audit = build_template_bem_gain(
        subjects_dir=args.subjects_dir,
        coordinate_file=root
        / "data"
        / "sensors"
        / "TDBRAIN_Table3_electrode_coordinates.csv",
        atlas_file=root
        / "data"
        / "atlas"
        / "Schaefer2018_200Parcels_7Networks_centroids.csv",
        annotation_dir=root / "data" / "atlas" / "fsaverage",
        channel_names=config.monitor.channels,
        n_jobs=args.n_jobs,
    )
    region_names = pd.read_csv(
        root / "data" / "atlas" / "Schaefer2018_200Parcels_7Networks_centroids.csv"
    )["ROI Name"].astype(str).tolist()
    save_template_bem_gain(
        output_dir,
        gain,
        config.monitor.channels,
        region_names,
        metadata,
        region_audit,
    )

    analytic_monitor, _ = build_eeg_monitor(
        config.monitor,
        config.connectivity.expected_regions,
        config.simulation.monitor_period_ms,
    )
    analytic_monitor.configure()
    regularize_analytic_eeg_gain(
        analytic_monitor,
        connectome.centres,
        connectome.connectivity.orientations,
        config.monitor.minimum_source_sensor_distance_mm,
    )
    analytic = np.asarray(analytic_monitor.gain, dtype=float)
    average_reference = np.eye(len(config.monitor.channels)) - np.ones(
        (len(config.monitor.channels), len(config.monitor.channels))
    ) / len(config.monitor.channels)
    analytic = average_reference @ analytic
    analytic /= np.sqrt(np.mean(analytic**2))

    cosines = _column_cosines(analytic, gain)
    analytic_covariance = analytic @ analytic.T
    template_covariance = gain @ gain.T
    audit = {
        "analytic_and_bem_gain_flat_correlation": float(
            np.corrcoef(analytic.ravel(), gain.ravel())[0, 1]
        ),
        "parcel_topography_cosine": {
            "median": float(np.median(cosines)),
            "p05": float(np.quantile(cosines, 0.05)),
            "p95": float(np.quantile(cosines, 0.95)),
            "negative_fraction": float(np.mean(cosines < 0.0)),
        },
        "sensor_covariance_upper_triangle_correlation": _upper_correlation(
            analytic_covariance, template_covariance
        ),
        "analytic_rank": int(np.linalg.matrix_rank(analytic)),
        "template_bem_rank": int(np.linalg.matrix_rank(gain)),
        "interpretation": (
            "Geometry audit only. Empirical superiority must be evaluated with "
            "nested M5.2 development folds, never the consumed M5.1 holdout."
        ),
    }
    (output_dir / "analytic_vs_template_bem_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].hist(cosines, bins=30, color="#3b82f6", alpha=0.85)
    axes[0].axvline(np.median(cosines), color="black", linestyle="--")
    axes[0].set_xlabel("Per-parcel sensor-topography cosine")
    axes[0].set_ylabel("Parcels")
    axes[0].set_title("Analytic centroid vs distributed BEM")
    image = axes[1].imshow(analytic_covariance, cmap="coolwarm", aspect="auto")
    axes[1].set_title("Analytic gain spatial Gram")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Channel")
    figure.colorbar(image, ax=axes[1], shrink=0.75)
    image = axes[2].imshow(template_covariance, cmap="coolwarm", aspect="auto")
    axes[2].set_title("Template-BEM gain spatial Gram")
    axes[2].set_xlabel("Channel")
    axes[2].set_ylabel("Channel")
    figure.colorbar(image, ax=axes[2], shrink=0.75)
    figure.suptitle("M5.2 forward-model geometry audit")
    figure.savefig(output_dir / "template_bem_gain_audit.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"metadata": metadata, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
