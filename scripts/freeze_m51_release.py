"""Create and verify a checksum manifest for the frozen M5.1 production run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _structural_values(path: Path) -> dict[str, list[float]]:
    names = (
        "default_incident_weight_contrast",
        "dorsattn_salventattn_weight_balance",
    )
    unique = {name: set() for name in names}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for name in names:
                unique[name].add(float(row[name]))
    return {name: sorted(values) for name, values in unique.items()}


def _git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    production = root / "outputs" / "m5_spectral_m51_production"
    if not production.is_dir():
        raise FileNotFoundError(production)

    summary_path = production / "m51_production_summary.json"
    fit_summary_path = production / "fit" / "fit_summary.json"
    production_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit_summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))

    candidates_path = production / "bank" / "candidate_parameters.csv"
    replicates_path = production / "bank" / "replicates.csv"
    failures_path = production / "bank" / "simulation_failures.csv"
    structural_values = _structural_values(candidates_path)

    candidate_rows = _row_count(candidates_path)
    replicate_rows = _row_count(replicates_path)
    failure_rows = _row_count(failures_path)
    expected_candidates = int(production_summary["parameters_shape"][0])
    expected_replicates = expected_candidates * int(
        production_summary["csd_replicates_shape"][1]
    )

    checks = {
        "source_commit_is_a302fdf": production_summary["git_commit"] == "a302fdf",
        "candidate_rows_are_2048": candidate_rows == 2048 == expected_candidates,
        "replicate_rows_are_6144": replicate_rows == 6144 == expected_replicates,
        "zero_simulation_failures": failure_rows == 0,
        "production_arrays_declared_finite": bool(production_summary["finite"]),
        "structural_fitting_disabled": not bool(
            production_summary["fit_structural_modes"]
        ),
        "both_structural_columns_fixed_zero": all(
            values == [0.0] for values in structural_values.values()
        ),
        "fit_is_production_scale": fit_summary["run_scale"] == "production",
        "fit_status_recorded": fit_summary["status"] in {"accepted", "not_accepted"},
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"M5.1 release validation failed: {failed}")

    artifact_paths = sorted(path for path in production.rglob("*") if path.is_file())
    artifact_paths.append(root / "configs" / "m5_spectral.toml")
    artifacts = []
    for path in sorted(set(artifact_paths)):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "release": "m5.1-fixed-connectome-production",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_freeze": _git_value(root, "rev-parse", "HEAD"),
        "repository_branch_at_freeze": _git_value(root, "branch", "--show-current"),
        "bank_source_commit": production_summary["git_commit"],
        "scientific_status": fit_summary["status"],
        "artifact_root": production.relative_to(root).as_posix(),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(item["bytes"] for item in artifacts),
        "validation": {
            "checks": checks,
            "candidate_rows": candidate_rows,
            "replicate_rows": replicate_rows,
            "simulation_failure_rows": failure_rows,
            "structural_unique_values": structural_values,
            "acceptance_gates": fit_summary["acceptance_gates"],
        },
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/M51_RELEASE_MANIFEST.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = build_manifest(root, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_count": manifest["artifact_count"],
                "artifact_bytes": manifest["artifact_bytes"],
                "scientific_status": manifest["scientific_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
