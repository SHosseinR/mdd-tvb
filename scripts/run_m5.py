"""Run M5 empirical extraction, direct simulation bank, and hierarchical fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from mdd_tvb.m5 import run_m5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m5_fit.toml"))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("extract", "bank", "fit"),
        default=("extract", "bank", "fit"),
    )
    parser.add_argument("--max-subjects-per-group", type=int, default=None)
    parser.add_argument("--design-samples", type=int, default=None)
    args = parser.parse_args()
    run_m5(
        args.config,
        tuple(args.stages),
        max_subjects_per_group=args.max_subjects_per_group,
        design_samples=args.design_samples,
    )


if __name__ == "__main__":
    main()
