"""Select alpha-topography objective weight using training subjects only."""

from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import load_cross_spectral_collection
from mdd_tvb.spectral_fit import fit_spectral_subjects


def main() -> None:
    config = load_spectral_m5_config("configs/m5_spectral_calibration.toml")
    empirical = config.paths.output_dir / "empirical"
    bank = load_spectral_simulation_bank(
        config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    fitting = load_cross_spectral_collection(empirical / "cross_spectra_fit.npz")
    validation = load_cross_spectral_collection(
        empirical / "cross_spectra_validation.npz"
    )
    reliability_a = load_cross_spectral_collection(
        empirical / "cross_spectra_reliability_a.npz"
    )
    reliability_b = load_cross_spectral_collection(
        empirical / "cross_spectra_reliability_b.npz"
    )
    # Equal residual weight is retained for spectral shape and coherency. The
    # direct alpha-topography block is varied without using diagnosis labels.
    variants = (
        (0.00, 8),
        (0.05, 8),
        (0.10, 8),
        (0.10, 16),
        (0.15, 16),
        (0.20, 16),
        (0.20, 24),
    )
    root = config.project_root / "outputs" / "m5_spectral_topography_audit"
    rows: list[dict[str, float | int]] = []
    for topography_weight, coordinates in variants:
        residual = (1.0 - topography_weight) / 2.0
        spectral = replace(
            config.spectral,
            reliability_max_auto_coordinates=16,
            reliability_max_cross_coordinates=16,
            reliability_max_topography_coordinates=coordinates,
            reliability_min_coordinates=min(8, coordinates),
            auto_weight=residual,
            cross_weight=residual,
            topography_weight=topography_weight,
        )
        output = root / f"weight_{topography_weight:.2f}_coordinates_{coordinates}"
        working = replace(
            config,
            spectral=spectral,
            paths=replace(config.paths, output_dir=output),
        )
        fit_spectral_subjects(
            working,
            fitting,
            validation,
            bank,
            reliability_a,
            reliability_b,
        )
        summary = json.loads((output / "fit" / "fit_summary.json").read_text())
        train = summary["validation"]["train"]
        rows.append(
            {
                "topography_weight": topography_weight,
                "topography_coordinates": coordinates,
                "training_unseen_cost_ratio": train[
                    "median_validation_cost_ratio_to_pooled_null"
                ],
                "training_unseen_topography_cost_ratio": train[
                    "median_alpha_topography_validation_cost_ratio_to_pooled_null"
                ],
                "training_fraction_beating_null": train[
                    "fraction_beating_pooled_null_on_validation"
                ],
            }
        )
        print(
            f"Completed weight={topography_weight:.2f}, coordinates={coordinates}",
            flush=True,
        )
    table = pd.DataFrame(rows).sort_values("training_unseen_cost_ratio")
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "training_only_topography_ranking.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
