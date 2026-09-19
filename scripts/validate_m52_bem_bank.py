"""Validate and checksum the downloaded M5.2 template-BEM production bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


RUNS = {
    "legacy": {
        "commit": "9a1b6ad",
        "gain_sha256": "299efec33c8a38879ef480e5aa4757ff9b62e48636067d23dd61f6ba4c9ebf8b",
        "directory": "m52_template_bem_production",
        "summary": "m52_template_bem_summary.json",
        "release": "m5.2-template-bem-production-bank-invalid-registration",
    },
    "corrected": {
        "commit": "a8f4bfc",
        "gain_sha256": "e0e0c4bbf453ed0d81ce27867a92ee974f9da477b210fa3375d4d2fca92ac3ac",
        "directory": "m52_template_bem_corrected_production",
        "summary": "m52_template_bem_corrected_summary.json",
        "release": "m5.2-template-bem-corrected-production-bank",
    },
}
EXPECTED_CANDIDATE_SHA256 = (
    "1f2b5a79abb98ddf7c496aea9ee735fb354ed9d0ae2d47cf13faceea0b40b4da"
)
STRUCTURAL_COLUMNS = (
    "default_incident_weight_contrast",
    "dorsattn_salventattn_weight_balance",
)


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    major, minor = np.lib.format.read_magic(stream)
    if major == 1:
        return np.lib.format.read_array_header_1_0(stream)
    if major in {2, 3}:
        return np.lib.format.read_array_header_2_0(stream)
    raise RuntimeError(f"Unsupported NPY version {major}.{minor}")


def _stream_array_finite(
    archive: zipfile.ZipFile,
    member: str,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> tuple[tuple[int, ...], np.dtype, bool]:
    """Inspect one NPZ member without retaining its potentially large array."""

    with archive.open(member) as stream:
        shape, _fortran_order, dtype = _read_npy_header(stream)
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise RuntimeError(f"Object dtype is not permitted in {member}")
        expected_values = int(np.prod(shape, dtype=np.int64))
        values_read = 0
        finite = True
        remainder = b""
        while chunk := stream.read(chunk_bytes):
            raw = remainder + chunk
            usable = len(raw) - (len(raw) % dtype.itemsize)
            if usable:
                values = np.frombuffer(raw[:usable], dtype=dtype)
                values_read += values.size
                finite = finite and bool(np.isfinite(values).all())
            remainder = raw[usable:]
        if remainder or values_read != expected_values:
            raise RuntimeError(
                f"Truncated {member}: read {values_read} of {expected_values} values"
            )
    return tuple(int(value) for value in shape), dtype, finite


def validate_bank(
    source: Path,
    *,
    variant: str = "legacy",
    expected_commit: str | None = None,
    expected_candidates: int = 2048,
    expected_replicates: int = 3,
    expected_candidate_sha256: str = EXPECTED_CANDIDATE_SHA256,
    expected_gain_sha256: str | None = None,
) -> dict[str, Any]:
    run = RUNS[variant]
    expected_commit = expected_commit or run["commit"]
    expected_gain_sha256 = expected_gain_sha256 or run["gain_sha256"]
    source = source.resolve()
    nested = source / run["directory"] / "bank"
    if nested.is_dir():
        bank_dir = nested
    elif (source / "bank").is_dir():
        bank_dir = source / "bank"
    else:
        bank_dir = source
    summary_path = source / run["summary"]
    if not summary_path.is_file():
        summary_path = bank_dir.parent / run["summary"]

    required = {
        "summary": summary_path,
        "bank": bank_dir / "spectral_simulation_bank.npz",
        "candidates": bank_dir / "candidate_parameters.csv",
        "replicates": bank_dir / "replicates.csv",
        "failures": bank_dir / "simulation_failures.csv",
        "backend": bank_dir / "backend_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing production artifacts: {missing}")

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    backend = json.loads(required["backend"].read_text(encoding="utf-8"))
    candidates = _csv_rows(required["candidates"])
    replicates = _csv_rows(required["replicates"])
    failures = _csv_rows(required["failures"])

    structural_unique = {
        name: sorted({float(row[name]) for row in candidates})
        for name in STRUCTURAL_COLUMNS
    }
    with zipfile.ZipFile(required["bank"]) as archive:
        members = set(archive.namelist())
        required_members = {
            "parameters.npy",
            "parameter_names.npy",
            "frequency_hz.npy",
            "channel_names.npy",
            "csd_replicates.npy",
        }
        missing_members = sorted(required_members - members)
        if missing_members:
            raise RuntimeError(f"Bank is missing arrays: {missing_members}")
        csd_shape, csd_dtype, csd_finite = _stream_array_finite(
            archive, "csd_replicates.npy"
        )
        parameters_shape, parameters_dtype, parameters_finite = _stream_array_finite(
            archive, "parameters.npy"
        )

    with np.load(required["bank"]) as arrays:
        parameters = np.asarray(arrays["parameters"])
        parameter_names = [str(value) for value in arrays["parameter_names"]]
        frequency_hz = np.asarray(arrays["frequency_hz"])
        channel_names = [str(value) for value in arrays["channel_names"]]
    candidate_matrix = np.asarray(
        [[float(row[name]) for name in parameter_names] for row in candidates]
    )
    replicate_pairs = {
        (int(row["candidate_index"]), int(row["replicate"])) for row in replicates
    }
    expected_pairs = {
        (candidate_index, replicate)
        for candidate_index in range(expected_candidates)
        for replicate in range(expected_replicates)
    }

    checks = {
        "source_commit": summary.get("git_commit") == expected_commit,
        "candidate_hash": summary.get("candidate_sha256")
        == expected_candidate_sha256,
        "gain_hash_summary": summary.get("observation_gain_sha256")
        == expected_gain_sha256,
        "gain_hash_backend": backend.get("observation_gain_sha256")
        == expected_gain_sha256,
        "configured_gain_used": summary.get("observation_gain")
        == backend.get("observation_gain")
        == "configured_regional_gain",
        "candidate_count": len(candidates) == expected_candidates,
        "replicate_count": len(replicates)
        == expected_candidates * expected_replicates,
        "zero_failures": len(failures) == 0,
        "parameters_shape": parameters_shape == (expected_candidates, 17),
        "csd_shape": csd_shape
        == (expected_candidates, expected_replicates, 39, 26, 26),
        "summary_parameters_shape": summary.get("parameters_shape")
        == [expected_candidates, 17],
        "summary_csd_shape": summary.get("csd_replicates_shape")
        == list(csd_shape),
        "arrays_finite": parameters_finite and csd_finite,
        "summary_finite": summary.get("finite") is True,
        "structural_fitting_disabled": summary.get("fit_structural_modes") is False,
        "structural_columns_zero": all(
            values == [0.0] for values in structural_unique.values()
        ),
        "summary_structural_columns_zero": summary.get(
            "structural_columns_unique"
        )
        == [[0.0, 0.0]],
        "candidate_table_matches_bank": candidate_matrix.shape == parameters.shape
        and bool(np.allclose(candidate_matrix, parameters, rtol=0.0, atol=1e-12)),
        "candidate_indices_ordered": [int(row["candidate_index"]) for row in candidates]
        == list(range(expected_candidates)),
        "replicate_design_complete": replicate_pairs == expected_pairs
        and len(replicates) == len(replicate_pairs),
        "frequency_axis": frequency_hz.shape == (39,)
        and bool(np.all(np.diff(frequency_hz) > 0.0)),
        "channel_axis": len(channel_names) == 26
        and len(set(channel_names)) == 26,
        "parameter_axis": len(parameter_names) == 17
        and parameter_names == list(candidates[0].keys())[1:],
        "gpu_backend": backend.get("jax_backend") == "gpu",
        "backend_simulation_count": backend.get("simulations")
        == expected_candidates * expected_replicates,
        "bank_file_size_summary": summary.get("bank_file_bytes")
        == required["bank"].stat().st_size,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"M5.2 BEM bank validation failed: {failed}")

    artifact_paths = sorted(path for path in source.rglob("*") if path.is_file())
    repository_root = Path(__file__).resolve().parents[1]
    try:
        artifact_root = source.relative_to(repository_root).as_posix()
    except ValueError:
        artifact_root = source.as_posix()
    return {
        "schema_version": 1,
        "release": run["release"],
        "artifact_root": artifact_root,
        "artifact_count": len(artifact_paths),
        "artifact_bytes": sum(path.stat().st_size for path in artifact_paths),
        "validation": {
            "checks": checks,
            "candidate_rows": len(candidates),
            "replicate_rows": len(replicates),
            "failure_rows": len(failures),
            "parameters_shape": list(parameters_shape),
            "parameters_dtype": str(parameters_dtype),
            "csd_shape": list(csd_shape),
            "csd_dtype": str(csd_dtype),
            "structural_unique_values": structural_unique,
            "bank_source_commit": summary["git_commit"],
        },
        "artifacts": [
            {
                "path": path.relative_to(source).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--variant", choices=tuple(RUNS), default="legacy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = validate_bank(args.source, variant=args.variant)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["validation"], indent=2))


if __name__ == "__main__":
    main()
