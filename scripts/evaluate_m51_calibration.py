"""Evaluate the M5.1 GPU calibration bank without reading the old holdout."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_m5_observability import run_audit
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
    collection: CrossSpectralCollection, indices: np.ndarray
) -> CrossSpectralCollection:
    return CrossSpectralCollection(
        **{
            name: value[indices] if name in SUBJECT_FIELDS else value.copy()
            for name, value in collection.__dict__.items()
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m5_spectral_calibration.toml"),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--empirical-dir",
        type=Path,
        default=Path("outputs/m5_spectral/empirical"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    config = load_spectral_m5_config(args.config)
    bank_path = args.bank or (
        config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    output_dir = args.output_dir or (config.paths.output_dir / "evaluation")
    fitting_full = load_cross_spectral_collection(
        args.empirical_dir / "cross_spectra_fit.npz"
    )
    validation_full = load_cross_spectral_collection(
        args.empirical_dir / "cross_spectra_validation.npz"
    )
    reliability_a_full = load_cross_spectral_collection(
        args.empirical_dir / "cross_spectra_reliability_a.npz"
    )
    reliability_b_full = load_cross_spectral_collection(
        args.empirical_dir / "cross_spectra_reliability_b.npz"
    )
    development, old_holdout = _stratified_split(
        fitting_full.groups,
        config.spectral.holdout_fraction,
        config.spectral.split_seed,
    )
    fitting = _subset(fitting_full, development)
    validation = _subset(validation_full, development)
    reliability_a = _subset(reliability_a_full, development)
    reliability_b = _subset(reliability_b_full, development)
    bank = load_spectral_simulation_bank(bank_path)

    working = replace(
        config,
        spectral=replace(
            config.spectral,
            split_seed=config.spectral.split_seed + 101,
        ),
        paths=replace(config.paths, output_dir=output_dir),
    )
    fit_spectral_subjects(
        working,
        fitting,
        validation,
        bank,
        reliability_a,
        reliability_b,
    )
    summary_path = output_dir / "fit" / "fit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nested = summary["validation"]["holdout"]

    audit = run_audit(
        args.config,
        empirical_dir=args.empirical_dir,
        bank_path=bank_path,
        output_dir=output_dir / "observability",
        fit_summary_path=summary_path,
    )
    baseline_table = pd.read_csv(
        PROJECT_ROOT
        / "outputs"
        / "m51_connectivity_audit"
        / "nested_development_ranking.csv"
    )
    baseline = baseline_table[
        baseline_table["variant"] == "lagged_coherency"
    ].iloc[0]
    structural = audit["selected_prior_recovery"]
    structural_minimum = min(
        float(structural["default_incident_weight_contrast"]),
        float(structural["dorsattn_salventattn_weight_balance"]),
    )
    gates = {
        "nested_total_not_worse_than_old_lagged_bank": bool(
            nested["median_validation_cost_ratio_to_pooled_null"]
            <= float(baseline["nested_unseen_total_cost_ratio"]) * 1.02
        ),
        "nested_cross_improves_old_lagged_bank": bool(
            nested.get(
                "median_selected_connectivity_validation_cost_ratio_to_pooled_null",
                nested.get(
                    "median_complex_coherency_validation_cost_ratio_to_pooled_null"
                ),
            )
            < float(baseline["nested_unseen_cross_cost_ratio"])
        ),
        "nested_topography_improves_by_five_percent": bool(
            nested["median_alpha_topography_validation_cost_ratio_to_pooled_null"]
            < float(baseline["nested_unseen_topography_cost_ratio"]) * 0.95
        ),
        "both_structural_modes_recoverable": bool(structural_minimum >= 0.30),
        "majority_of_nested_subjects_beat_null": bool(
            nested["fraction_beating_pooled_null_on_validation"] > 0.50
        ),
    }
    fit_gates_pass = all(
        passed
        for name, passed in gates.items()
        if name != "both_structural_modes_recoverable"
    )
    if all(gates.values()):
        status = "promote_to_production"
    elif fit_gates_pass and not gates["both_structural_modes_recoverable"]:
        status = "promote_fixed_connectome"
    else:
        status = "revise"
    decision = {
        "status": status,
        "old_65_subject_holdout_read": False,
        "development_subjects": int(len(development)),
        "excluded_old_holdout_subjects": int(len(old_holdout)),
        "nested_evaluation_subjects": int(nested["n"]),
        "bank_candidates": int(len(bank.parameters)),
        "bank_replicates": int(bank.csd_replicates.shape[1]),
        "metrics": nested,
        "baseline_nested_lagged": baseline.to_dict(),
        "minimum_structural_recovery_correlation": structural_minimum,
        "production_structural_policy": (
            "fit_bounded_modes"
            if gates["both_structural_modes_recoverable"]
            else "fix_common_connectome_at_reference"
        ),
        "gates": gates,
    }
    decision_path = output_dir / "CALIBRATION_DECISION.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        f"# M5.1 calibration: {decision['status']}",
        "",
        "The original 65-subject holdout was not read.",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}"
        for name, passed in gates.items()
    )
    (output_dir / "CALIBRATION_DECISION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
