"""Create a broad-plus-local M5.1 production design from development fits."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_parameterization import (
    PARAMETER_NAMES,
    candidates_from_parameter_matrix,
    denormalized_spectral_parameters,
    make_spectral_design,
    normalized_spectral_parameters,
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
        "--fit-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/m5_spectral_m51_adaptive_candidates.csv"),
    )
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--broad-samples", type=int, default=512)
    parser.add_argument("--anchors", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_spectral_m5_config(args.config)
    bank_path = args.bank or (
        config.paths.output_dir / "bank" / "spectral_simulation_bank.npz"
    )
    fit_dir = args.fit_dir or (config.paths.output_dir / "evaluation" / "fit")
    decision_path = args.decision or (
        config.paths.output_dir / "evaluation" / "CALIBRATION_DECISION.json"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "promote_to_production" and not args.force:
        raise RuntimeError(
            "Calibration did not pass promotion gates; adaptive production "
            "design was not generated"
        )
    if not 2 <= args.broad_samples < args.samples:
        raise ValueError("broad-samples must be between 2 and total samples")

    bank = load_spectral_simulation_bank(bank_path)
    top = pd.read_csv(fit_dir / "subject_top_posterior_states.csv")
    per_subject_candidate = (
        top.groupby(["subject_id", "candidate_index"], as_index=False)[
            "posterior_weight"
        ]
        .sum()
    )
    aggregate = (
        per_subject_candidate.groupby("candidate_index", as_index=False)
        .agg(
            aggregate_posterior_weight=("posterior_weight", "sum"),
            subjects_supported=("subject_id", "nunique"),
        )
        .sort_values(
            ["aggregate_posterior_weight", "subjects_supported"],
            ascending=False,
        )
    )
    anchor_indices = aggregate["candidate_index"].to_numpy(dtype=int)[
        : args.anchors
    ]
    if len(anchor_indices) < 2:
        raise RuntimeError("Too few diagnosis-blind posterior anchors")

    broad_global = 16
    if args.broad_samples % broad_global:
        raise ValueError("broad-samples must be divisible by 16")
    broad_settings = replace(
        config.design,
        samples=args.broad_samples,
        strategy="factorized",
        global_samples=broad_global,
        spatial_samples=args.broad_samples // broad_global,
        seed=args.seed,
    )
    broad = np.stack(
        [candidate.numeric_vector() for candidate in make_spectral_design(broad_settings)]
    )

    normalized_bank = normalized_spectral_parameters(bank.parameters, config.design)
    anchor_values = normalized_bank[anchor_indices]
    local_count = args.samples - args.broad_samples
    engine = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True, seed=args.seed + 1)
    exponent = int(np.ceil(np.log2(local_count)))
    offsets = 2.0 * engine.random_base2(exponent)[:local_count] - 1.0
    radius = np.asarray(
        [0.12] * 9
        + [0.18] * 6
        + [0.25] * 2,
        dtype=float,
    )
    local_normalized = np.empty_like(offsets)
    local_anchor = np.empty(local_count, dtype=int)
    for row in range(local_count):
        anchor_position = row % len(anchor_indices)
        local_anchor[row] = int(anchor_indices[anchor_position])
        local_normalized[row] = np.clip(
            anchor_values[anchor_position] + offsets[row] * radius,
            -1.0,
            1.0,
        )
    local = denormalized_spectral_parameters(local_normalized, config.design)
    combined = np.vstack((broad, local))

    # Clipping can create duplicate boundary rows. Keep the first occurrence,
    # then fill any deficit with a reproducible broad Sobol sequence.
    rounded = np.round(combined, decimals=12)
    _, unique_index = np.unique(rounded, axis=0, return_index=True)
    keep = np.sort(unique_index)
    combined = combined[keep]
    origins = np.asarray(
        ["broad"] * args.broad_samples + ["adaptive_local"] * local_count,
        dtype=object,
    )[keep]
    anchor_column = np.concatenate(
        (
            np.full(args.broad_samples, -1, dtype=int),
            local_anchor,
        )
    )[keep]
    if len(combined) < args.samples:
        missing = args.samples - len(combined)
        filler_engine = qmc.Sobol(
            d=len(PARAMETER_NAMES), scramble=True, seed=args.seed + 2
        )
        filler_exponent = int(np.ceil(np.log2(missing)))
        filler_normalized = (
            2.0 * filler_engine.random_base2(filler_exponent)[:missing] - 1.0
        )
        filler = denormalized_spectral_parameters(filler_normalized, config.design)
        combined = np.vstack((combined, filler))
        origins = np.concatenate((origins, np.asarray(["broad_fill"] * missing)))
        anchor_column = np.concatenate(
            (anchor_column, np.full(missing, -1, dtype=int))
        )
    combined = combined[: args.samples]
    origins = origins[: args.samples]
    anchor_column = anchor_column[: args.samples]
    candidates_from_parameter_matrix(combined, config.design)

    table = pd.DataFrame(combined, columns=PARAMETER_NAMES)
    table.insert(0, "anchor_calibration_candidate", anchor_column)
    table.insert(0, "proposal_origin", origins)
    table.insert(0, "candidate_index", np.arange(len(table)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    aggregate.to_csv(
        args.output.with_name(f"{args.output.stem}_anchor_ranking.csv"), index=False
    )
    summary = {
        "status": "completed",
        "diagnosis_labels_used": False,
        "old_65_subject_holdout_used": False,
        "samples": int(len(table)),
        "broad_samples": int(np.sum(origins == "broad")),
        "adaptive_local_samples": int(np.sum(origins == "adaptive_local")),
        "anchors": anchor_indices.tolist(),
        "normalized_local_radius": {
            name: float(radius[index]) for index, name in enumerate(PARAMETER_NAMES)
        },
        "candidate_table": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
