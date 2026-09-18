"""Compare leakage-robust connectivity objectives on development subjects.

The original 65-subject holdout is excluded entirely.  Each variant is fitted
and evaluated in a fresh nested split of the original 262 development subjects.
This script may therefore guide M5.1 design without repeatedly consulting the
reported production holdout.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import (
    CrossSpectralCollection,
    load_cross_spectral_collection,
)
from mdd_tvb.spectral_fit import _stratified_split, fit_spectral_subjects


SUBJECT_FIELDS = {
    "subject_ids",
    "groups",
    "source_files",
    "durations_s",
    "epoch_counts",
    "csd",
}


def _subset(
    collection: CrossSpectralCollection, indices: object
) -> CrossSpectralCollection:
    return CrossSpectralCollection(
        **{
            name: value[indices] if name in SUBJECT_FIELDS else value.copy()
            for name, value in collection.__dict__.items()
        }
    )


def main() -> None:
    config = load_spectral_m5_config("configs/m5_spectral.toml")
    empirical_dir = config.paths.output_dir / "empirical"
    bank = load_spectral_simulation_bank(
        config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    fitting_full = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_fit.npz"
    )
    validation_full = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_validation.npz"
    )
    reliability_a_full = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_a.npz"
    )
    reliability_b_full = load_cross_spectral_collection(
        empirical_dir / "cross_spectra_reliability_b.npz"
    )
    development, _ = _stratified_split(
        fitting_full.groups,
        config.spectral.holdout_fraction,
        config.spectral.split_seed,
    )
    fitting = _subset(fitting_full, development)
    validation = _subset(validation_full, development)
    reliability_a = _subset(reliability_a_full, development)
    reliability_b = _subset(reliability_b_full, development)

    variants = (
        ("complex_coherency", "complex_coherency", (0.0,), (1.0,)),
        ("imaginary_coherency", "imaginary_coherency", (0.0,), (1.0,)),
        ("lagged_coherency", "lagged_coherency", (0.0,), (1.0,)),
        (
            "lagged_coherency_source_background",
            "lagged_coherency",
            (0.0, 0.30, 0.60),
            (1.0,),
        ),
    )
    root = config.project_root / "outputs" / "m51_connectivity_audit"
    rows: list[dict[str, object]] = []
    for name, metric, source_fractions, source_exponents in variants:
        output = root / name
        spectral = replace(
            config.spectral,
            cross_metric=metric,
            # A second deterministic seed makes this a genuinely nested
            # development split rather than recreating the old subject split.
            split_seed=config.spectral.split_seed + 101,
        )
        working = replace(
            config,
            spectral=spectral,
            posterior=replace(
                config.posterior,
                source_background_fractions=source_fractions,
                source_background_exponents=source_exponents,
            ),
            paths=replace(config.paths, output_dir=output),
        )
        summary_path = output / "fit" / "fit_summary.json"
        if not summary_path.is_file():
            fit_spectral_subjects(
                working,
                fitting,
                validation,
                bank,
                reliability_a,
                reliability_b,
            )
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        nested = summary["validation"]["holdout"]
        rows.append(
            {
                "cross_metric": metric,
                "variant": name,
                "source_background_fractions": ",".join(
                    str(value) for value in source_fractions
                ),
                "nested_development_subjects": nested["n"],
                "nested_unseen_total_cost_ratio": nested[
                    "median_validation_cost_ratio_to_pooled_null"
                ],
                "nested_unseen_auto_cost_ratio": nested[
                    "median_auto_spectrum_validation_cost_ratio_to_pooled_null"
                ],
                "nested_unseen_cross_cost_ratio": nested[
                    "median_complex_coherency_validation_cost_ratio_to_pooled_null"
                ],
                "nested_unseen_topography_cost_ratio": nested[
                    "median_alpha_topography_validation_cost_ratio_to_pooled_null"
                ],
                "nested_fraction_beating_null": nested[
                    "fraction_beating_pooled_null_on_validation"
                ],
                "nested_median_candidate_ess": nested[
                    "median_candidate_posterior_ess"
                ],
                "selected_temperature": summary["posterior"][
                    "selected_temperature_training_subjects_only"
                ],
            }
        )
        print(f"Completed nested development metric: {metric}", flush=True)
    table = pd.DataFrame(rows).sort_values(
        ["nested_unseen_total_cost_ratio", "nested_unseen_cross_cost_ratio"]
    )
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "nested_development_ranking.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
