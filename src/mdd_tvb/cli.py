"""Command-line interface for validation and baseline simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .connectome import load_connectome
from .eeg import sensor_locations
from .reporting import save_run
from .simulation import run_baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdd-tvb")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("validate", "Audit inputs and the EEG montage without simulating."),
        ("run", "Run and save the baseline whole-brain simulation."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument(
            "--config", type=Path, default=Path("configs/baseline.toml")
        )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    connectome = load_connectome(config.paths, config.connectivity)
    locations = sensor_locations(config.monitor)
    if args.command == "validate":
        payload = {
            "connectome": connectome.audit,
            "sensors": {
                "count": len(config.monitor.channels),
                "labels": list(config.monitor.channels),
                "montage": config.monitor.montage,
                "unit_norm_max_error": float(abs((locations**2).sum(axis=1) ** 0.5 - 1).max()),
            },
        }
        print(json.dumps(payload, indent=2))
        return

    result = run_baseline(config, connectome)
    output_dir = save_run(config, connectome, result)
    print(json.dumps({
        "output_dir": str(output_dir),
        "simulation": result.metadata,
    }, indent=2))

