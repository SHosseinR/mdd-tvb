from __future__ import annotations

import csv
from pathlib import Path


def test_m52_nested_split_is_balanced_and_excludes_frozen_holdout() -> None:
    with Path("configs/m52_nested_splits.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    with Path(
        "outputs/m5_spectral_m51_production/fit/data_split.csv"
    ).open("r", encoding="utf-8", newline="") as stream:
        frozen = list(csv.DictReader(stream))

    development_ids = {
        row["subject_id"] for row in frozen if row["subject_split"] == "train"
    }
    consumed_holdout_ids = {
        row["subject_id"] for row in frozen if row["subject_split"] == "holdout"
    }
    split_ids = {row["subject_id"] for row in rows}
    assert split_ids == development_ids
    assert split_ids.isdisjoint(consumed_holdout_ids)
    assert len(rows) == 262 * 5

    for subject_id in split_ids:
        subject_rows = [row for row in rows if row["subject_id"] == subject_id]
        assert len(subject_rows) == 5
        assert sum(row["outer_role"] == "validation" for row in subject_rows) == 1
        assert sum(row["outer_role"] == "training" for row in subject_rows) == 4

    validation_sizes = []
    for fold in range(5):
        selected = [
            row
            for row in rows
            if int(row["outer_fold"]) == fold
            and row["outer_role"] == "validation"
        ]
        validation_sizes.append(len(selected))
        groups = {group: 0 for group in ("Healthy", "MDD")}
        for row in selected:
            groups[row["group"]] += 1
        assert abs(groups["Healthy"] - groups["MDD"]) <= 4
    assert validation_sizes == [54, 52, 52, 52, 52]
