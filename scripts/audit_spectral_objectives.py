"""Choose CSD objective resolution using training subjects only."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

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
    resolutions = ((16, 16), (32, 32), (32, 64), (64, 64), (64, 128))
    rows: list[dict[str, float | int | str]] = []
    audit_root = config.project_root / "outputs" / "m5_spectral_objective_audit"
    for auto_count, cross_count in resolutions:
        output = audit_root / f"auto_{auto_count}_cross_{cross_count}"
        spectral = replace(
            config.spectral,
            sensor_basis_method="empirical",
            reliability_max_auto_coordinates=auto_count,
            reliability_max_cross_coordinates=cross_count,
            reliability_min_coordinates=min(8, auto_count, cross_count),
        )
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
                "auto_coordinates": auto_count,
                "cross_coordinates": cross_count,
                "training_unseen_cost_ratio": train[
                    "median_validation_cost_ratio_to_pooled_null"
                ],
                "training_unseen_map_cost_ratio": train[
                    "median_map_validation_cost_ratio_to_pooled_null"
                ],
                "training_unseen_oracle_cost_ratio": train[
                    "median_oracle_validation_cost_ratio_to_pooled_null_not_a_fit"
                ],
                "training_fraction_beating_null": train[
                    "fraction_beating_pooled_null_on_validation"
                ],
            }
        )
        print(f"Completed auto={auto_count}, cross={cross_count}", flush=True)
    table = pd.DataFrame(rows).sort_values("training_unseen_cost_ratio")
    audit_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(audit_root / "training_only_objective_ranking.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
