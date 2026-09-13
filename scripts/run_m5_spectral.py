"""Run the redesigned resting-state complex cross-spectral M5 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from mdd_tvb.spectral_bank import build_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_empirical import extract_cross_spectral_collection
from mdd_tvb.spectral_fit import fit_spectral_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m5_spectral_pilot.toml")
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("extract", "bank", "fit"),
        default=("extract", "bank", "fit"),
    )
    parser.add_argument("--max-subjects-per-group", type=int, default=None)
    parser.add_argument("--design-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_spectral_m5_config(args.config)
    fitting = validation = bank = None
    if "extract" in args.stages:
        fitting, validation = extract_cross_spectral_collection(
            config, args.max_subjects_per_group
        )
    if "bank" in args.stages:
        bank = build_spectral_simulation_bank(config, args.design_samples)
    if "fit" in args.stages:
        fit_spectral_subjects(config, fitting, validation, bank)


if __name__ == "__main__":
    main()
