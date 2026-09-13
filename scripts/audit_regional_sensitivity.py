"""Audit network-local drive, time-scale, and neural-noise perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

from mdd_tvb.regional_sensitivity import run_regional_sensitivity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m5_fit_v2.toml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m5_v2/feature_audit"))
    parser.add_argument("--candidate", type=int, default=3)
    parser.add_argument("--deviation", type=float, default=0.10)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    summary = run_regional_sensitivity(
        args.config,
        args.output,
        candidate_index=args.candidate,
        deviation=args.deviation,
        n_jobs=args.jobs,
    )
    ranked = (
        summary[summary["block"].isin(["Alpha topography", "Coherence alpha", "Coherence beta"])]
        .assign(score=lambda table: table["standardized_sensitivity_per_unit_gain"] * table["cosine_to_empirical_mdd_direction"].abs())
        .groupby("mode", as_index=False)["score"].mean()
        .sort_values("score", ascending=False)
    )
    print(ranked.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
