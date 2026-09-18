"""Create deterministic nested M5.2 splits without reusing the M5.1 holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _stratified_folds(
    rows: list[dict[str, str]], folds: int, seed: int
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    groups = sorted({row["group"] for row in rows})
    for group in groups:
        subjects = sorted(
            row["subject_id"] for row in rows if row["group"] == group
        )
        order = rng.permutation(len(subjects))
        for rank, index in enumerate(order):
            assignment[subjects[int(index)]] = rank % folds
    return assignment


def build_splits(
    source: Path,
    output: Path,
    metadata_output: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
) -> dict[str, object]:
    with source.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    required = {"subject_id", "group", "subject_split"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} in {source}")

    development = [row for row in source_rows if row["subject_split"] == "train"]
    consumed_holdout = [
        row for row in source_rows if row["subject_split"] == "holdout"
    ]
    if len(development) != 262 or len(consumed_holdout) != 65:
        raise ValueError(
            "Expected the frozen M5.1 split with 262 development and 65 holdout subjects"
        )

    outer_assignment = _stratified_folds(development, outer_folds, seed)
    output_rows: list[dict[str, object]] = []
    for outer_fold in range(outer_folds):
        outer_training = [
            row
            for row in development
            if outer_assignment[row["subject_id"]] != outer_fold
        ]
        inner_assignment = _stratified_folds(
            outer_training, inner_folds, seed + 1000 + outer_fold
        )
        for row in development:
            subject_id = row["subject_id"]
            is_validation = outer_assignment[subject_id] == outer_fold
            output_rows.append(
                {
                    "subject_id": subject_id,
                    "group": row["group"],
                    "outer_fold": outer_fold,
                    "outer_role": "validation" if is_validation else "training",
                    "inner_fold": "" if is_validation else inner_assignment[subject_id],
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    group_counts = {
        group: sum(row["group"] == group for row in development)
        for group in sorted({row["group"] for row in development})
    }
    outer_counts = []
    for outer_fold in range(outer_folds):
        selected = [
            row
            for row in development
            if outer_assignment[row["subject_id"]] == outer_fold
        ]
        outer_counts.append(
            {
                "outer_fold": outer_fold,
                "n": len(selected),
                "groups": {
                    group: sum(row["group"] == group for row in selected)
                    for group in group_counts
                },
            }
        )

    try:
        source_display = source.relative_to(Path.cwd()).as_posix()
    except ValueError:
        source_display = source.as_posix()
    metadata: dict[str, object] = {
        "schema_version": 1,
        "source": source_display,
        "seed": seed,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "development_subjects": len(development),
        "development_groups": group_counts,
        "excluded_consumed_m51_holdout_subjects": len(consumed_holdout),
        "outer_validation_counts": outer_counts,
        "rules": [
            "Only original M5.1 training subjects appear in these folds.",
            "Original M5.1 holdout subjects remain excluded from M5.2 selection.",
            "Group labels are used for split balance and post-fit audits, never individual fitting.",
        ],
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "outputs/m5_spectral_m51_production/fit/data_split.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/m52_nested_splits.csv")
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("configs/m52_nested_splits.json"),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    metadata = build_splits(
        resolved(args.source),
        resolved(args.output),
        resolved(args.metadata_output),
        args.outer_folds,
        args.inner_folds,
        args.seed,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
