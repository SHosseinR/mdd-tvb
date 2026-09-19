from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_m52_bem_bank import validate_bank


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_validate_small_complete_bank(tmp_path: Path) -> None:
    source = tmp_path / "production"
    bank_dir = source / "bank"
    bank_dir.mkdir(parents=True)
    parameter_names = [f"parameter_{index}" for index in range(15)] + [
        "default_incident_weight_contrast",
        "dorsattn_salventattn_weight_balance",
    ]
    parameters = np.arange(34, dtype=np.float64).reshape(2, 17)
    parameters[:, -2:] = 0.0
    frequency_hz = np.arange(1.0, 40.0)
    channel_names = np.asarray([f"E{index:02d}" for index in range(26)])
    csd = np.ones((2, 2, 39, 26, 26), dtype=np.complex64)
    bank_path = bank_dir / "spectral_simulation_bank.npz"
    np.savez_compressed(
        bank_path,
        parameters=parameters,
        parameter_names=np.asarray(parameter_names),
        frequency_hz=frequency_hz,
        channel_names=channel_names,
        csd_replicates=csd,
    )

    candidate_rows = []
    for candidate_index, values in enumerate(parameters):
        row: dict[str, object] = {"candidate_index": candidate_index}
        row.update(dict(zip(parameter_names, values, strict=True)))
        candidate_rows.append(row)
    _write_csv(
        bank_dir / "candidate_parameters.csv",
        ["candidate_index", *parameter_names],
        candidate_rows,
    )
    _write_csv(
        bank_dir / "replicates.csv",
        ["candidate_index", "replicate", "spectral_epoch_count"],
        [
            {
                "candidate_index": candidate_index,
                "replicate": replicate,
                "spectral_epoch_count": 10,
            }
            for candidate_index in range(2)
            for replicate in range(2)
        ],
    )
    _write_csv(
        bank_dir / "simulation_failures.csv",
        ["candidate_index", "replicate", "error"],
        [],
    )
    (bank_dir / "backend_metadata.json").write_text(
        json.dumps(
            {
                "jax_backend": "gpu",
                "simulations": 4,
                "observation_gain": "configured_regional_gain",
                "observation_gain_sha256": "gain",
            }
        ),
        encoding="utf-8",
    )
    (source / "m52_template_bem_summary.json").write_text(
        json.dumps(
            {
                "git_commit": "abc",
                "candidate_sha256": "cand",
                "observation_gain_sha256": "gain",
                "observation_gain": "configured_regional_gain",
                "parameters_shape": [2, 17],
                "csd_replicates_shape": [2, 2, 39, 26, 26],
                "finite": True,
                "fit_structural_modes": False,
                "structural_columns_unique": [[0.0, 0.0]],
                "bank_file_bytes": bank_path.stat().st_size,
            }
        ),
        encoding="utf-8",
    )

    manifest = validate_bank(
        source,
        expected_commit="abc",
        expected_candidates=2,
        expected_replicates=2,
        expected_candidate_sha256="cand",
        expected_gain_sha256="gain",
    )

    assert all(manifest["validation"]["checks"].values())
    assert manifest["validation"]["csd_shape"] == [2, 2, 39, 26, 26]

    candidate_rows[0]["default_incident_weight_contrast"] = 0.1
    _write_csv(
        bank_dir / "candidate_parameters.csv",
        ["candidate_index", *parameter_names],
        candidate_rows,
    )
    with pytest.raises(RuntimeError, match="structural_columns_zero"):
        validate_bank(
            source,
            expected_commit="abc",
            expected_candidates=2,
            expected_replicates=2,
            expected_candidate_sha256="cand",
            expected_gain_sha256="gain",
        )
