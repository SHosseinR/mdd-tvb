"""Run the empirical/simulation audit of the current M5 fitting features."""

from __future__ import annotations

import argparse
from pathlib import Path

from mdd_tvb.m5_feature_audit import load_audit_inputs, run_feature_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m5-output",
        type=Path,
        default=Path("outputs/m5"),
        help="M5 output root containing empirical/, bank/, and fit/.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/m5/feature_audit"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path(
            "D:/university/projects/graph-opt/eeg-cluster-py/"
            "classification_score/data/tdbrain_participants.csv"
        ),
    )
    args = parser.parse_args()
    root = args.m5_output
    inputs = load_audit_inputs(
        root / "empirical" / "empirical_features.npz",
        root / "empirical" / "empirical_validation_features.npz",
        root / "bank" / "simulation_bank.npz",
        root / "fit" / "feature_transformer.npz",
        root / "fit" / "subject_fits.csv",
        args.participants,
    )
    summary = run_feature_audit(
        inputs, args.output, seed=args.seed, repeats=args.repeats
    )
    print(f"Wrote {args.output / 'feature_audit_summary.json'}")
    print(summary["parameter_identifiability"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
