"""Paired leakage-safe comparison of two M5.2 nested model evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mdd_tvb.spectral_features import load_cross_spectral_collection
from mdd_tvb.spectral_fit import (
    _channel_log_power,
    _connectivity_metric,
    _safe_correlation,
)


COST_COLUMNS = {
    "total": "validation_cost_ratio_to_pooled_null",
    "autospectrum": "validation_auto_spectrum_cost_ratio_to_pooled_null",
    "selected_connectivity": (
        "validation_complex_coherency_cost_ratio_to_pooled_null"
    ),
    "alpha_topography": (
        "validation_alpha_topography_cost_ratio_to_pooled_null"
    ),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_oof_predictions(directory: Path, table: pd.DataFrame) -> np.ndarray:
    combined = directory / "out_of_fold_posterior_predictive_csd.npz"
    if combined.is_file():
        with np.load(combined) as payload:
            ids = payload["subject_ids"].astype(str)
            csd = payload["csd"]
        lookup = {value: index for index, value in enumerate(ids)}
        return np.stack([csd[lookup[value]] for value in table.subject_id.astype(str)])

    fold_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    predictions: list[np.ndarray] = []
    for row in table.itertuples(index=False):
        fold = int(row.outer_fold)
        if fold not in fold_payloads:
            path = directory / f"outer_{fold}" / "posterior_predictive_csd.npz"
            with np.load(path) as payload:
                fold_payloads[fold] = (
                    payload["subject_ids"].astype(str),
                    payload["csd"],
                )
        ids, csd = fold_payloads[fold]
        match = np.flatnonzero(ids == str(row.subject_id))
        if len(match) != 1:
            raise ValueError(f"Missing unique prediction for {row.subject_id}")
        predictions.append(csd[int(match[0])])
    return np.stack(predictions)


def _align_table(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if reference.subject_id.duplicated().any() or candidate.subject_id.duplicated().any():
        raise ValueError("Every model must contain one out-of-fold row per subject")
    reference_ids = reference.subject_id.astype(str)
    candidate_indexed = candidate.assign(
        subject_id=candidate.subject_id.astype(str)
    ).set_index("subject_id")
    if set(reference_ids) != set(candidate_indexed.index):
        raise ValueError("Reference and candidate subject sets differ")
    aligned = candidate_indexed.loc[reference_ids].reset_index()
    if not np.array_equal(reference.group.astype(str), aligned.group.astype(str)):
        raise ValueError("Reference and candidate group labels differ")
    return aligned


def _bootstrap_median_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    difference = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    draws = rng.integers(0, len(difference), size=(samples, len(difference)))
    estimates = np.median(difference[draws], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(np.median(difference)), float(low), float(high)


def _feature_blocks(csd: np.ndarray, frequency: np.ndarray, metric: str) -> dict[str, np.ndarray]:
    power = _channel_log_power(csd)
    alpha = (frequency >= 8.0) & (frequency < 13.0)
    topography = power[:, alpha].mean(axis=1)
    topography -= topography.mean(axis=1, keepdims=True)
    return {
        "power": power.reshape(len(power), -1),
        "alpha_topography": topography,
        "selected_connectivity": _connectivity_metric(csd, metric).reshape(
            len(csd), -1
        ),
    }


def _effect_statistics(empirical: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    correlation = _safe_correlation(empirical, prediction)
    retention = float(
        np.linalg.norm(prediction)
        / max(np.linalg.norm(empirical), np.finfo(float).tiny)
    )
    return correlation, retention


def _bootstrap_effect_comparison(
    empirical: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    groups: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    group_names = list(dict.fromkeys(groups.astype(str).tolist()))
    if len(group_names) != 2:
        raise ValueError("Group-effect comparison requires exactly two groups")
    first = np.flatnonzero(groups == group_names[0])
    second = np.flatnonzero(groups == group_names[1])
    weights = np.zeros((samples, len(groups)), dtype=float)
    for bootstrap in range(samples):
        first_draw = rng.choice(first, size=len(first), replace=True)
        second_draw = rng.choice(second, size=len(second), replace=True)
        weights[bootstrap] += np.bincount(
            second_draw, minlength=len(groups)
        ) / len(second)
        weights[bootstrap] -= np.bincount(
            first_draw, minlength=len(groups)
        ) / len(first)

    result: dict[str, Any] = {}
    for name in empirical:
        point_empirical = empirical[name][second].mean(axis=0) - empirical[name][first].mean(axis=0)
        point_reference = reference[name][second].mean(axis=0) - reference[name][first].mean(axis=0)
        point_candidate = candidate[name][second].mean(axis=0) - candidate[name][first].mean(axis=0)
        reference_point = _effect_statistics(point_empirical, point_reference)
        candidate_point = _effect_statistics(point_empirical, point_candidate)
        correlation_difference: list[np.ndarray] = []
        retention_difference: list[np.ndarray] = []
        for start in range(0, samples, batch_size):
            selected = weights[start : start + batch_size]
            empirical_effect = selected @ empirical[name]
            reference_effect = selected @ reference[name]
            candidate_effect = selected @ candidate[name]
            empirical_centered = empirical_effect - empirical_effect.mean(
                axis=1, keepdims=True
            )
            reference_centered = reference_effect - reference_effect.mean(
                axis=1, keepdims=True
            )
            candidate_centered = candidate_effect - candidate_effect.mean(
                axis=1, keepdims=True
            )
            empirical_norm = np.linalg.norm(empirical_centered, axis=1)
            reference_norm = np.linalg.norm(reference_centered, axis=1)
            candidate_norm = np.linalg.norm(candidate_centered, axis=1)
            denominator_reference = np.maximum(
                empirical_norm * reference_norm, np.finfo(float).tiny
            )
            denominator_candidate = np.maximum(
                empirical_norm * candidate_norm, np.finfo(float).tiny
            )
            reference_correlation = np.sum(
                empirical_centered * reference_centered, axis=1
            ) / denominator_reference
            candidate_correlation = np.sum(
                empirical_centered * candidate_centered, axis=1
            ) / denominator_candidate
            correlation_difference.append(
                candidate_correlation - reference_correlation
            )
            raw_empirical_norm = np.maximum(
                np.linalg.norm(empirical_effect, axis=1), np.finfo(float).tiny
            )
            retention_difference.append(
                np.linalg.norm(candidate_effect, axis=1) / raw_empirical_norm
                - np.linalg.norm(reference_effect, axis=1) / raw_empirical_norm
            )
        correlation_delta = np.concatenate(correlation_difference)
        retention_delta = np.concatenate(retention_difference)
        result[name] = {
            "reference": {
                "effect_correlation": reference_point[0],
                "effect_norm_retained": reference_point[1],
            },
            "candidate": {
                "effect_correlation": candidate_point[0],
                "effect_norm_retained": candidate_point[1],
            },
            "candidate_minus_reference": {
                "effect_correlation": float(candidate_point[0] - reference_point[0]),
                "effect_correlation_bootstrap_95_ci": np.quantile(
                    correlation_delta, [0.025, 0.975]
                ).tolist(),
                "effect_norm_retained": float(candidate_point[1] - reference_point[1]),
                "effect_norm_retained_bootstrap_95_ci": np.quantile(
                    retention_delta, [0.025, 0.975]
                ).tolist(),
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-name", default="M51")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--empirical-file",
        type=Path,
        default=Path(
            "outputs/m5_spectral_m51_production/empirical/cross_spectra_validation.npz"
        ),
    )
    parser.add_argument("--cross-metric", default="lagged_coherency")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    reference_dir = resolved(args.reference_dir)
    candidate_dir = resolved(args.candidate_dir)
    output_dir = resolved(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(reference_dir / "out_of_fold_subject_posteriors.csv")
    candidate_raw = pd.read_csv(candidate_dir / "out_of_fold_subject_posteriors.csv")
    candidate = _align_table(reference, candidate_raw)
    reference_prediction = _load_oof_predictions(reference_dir, reference)
    candidate_prediction_raw = _load_oof_predictions(candidate_dir, candidate_raw)
    candidate_lookup = {
        subject_id: index
        for index, subject_id in enumerate(candidate_raw.subject_id.astype(str))
    }
    candidate_prediction = np.stack(
        [candidate_prediction_raw[candidate_lookup[value]] for value in reference.subject_id.astype(str)]
    )

    empirical_collection = load_cross_spectral_collection(resolved(args.empirical_file))
    empirical_lookup = {
        subject_id: index
        for index, subject_id in enumerate(empirical_collection.subject_ids.astype(str))
    }
    empirical_csd = np.stack(
        [empirical_collection.csd[empirical_lookup[value]] for value in reference.subject_id.astype(str)]
    )
    rng = np.random.default_rng(args.seed)
    paired_rows: list[dict[str, Any]] = []
    costs: dict[str, Any] = {}
    for name, column in COST_COLUMNS.items():
        reference_value = reference[column].to_numpy(dtype=float)
        candidate_value = candidate[column].to_numpy(dtype=float)
        point, low, high = _bootstrap_median_difference(
            candidate_value,
            reference_value,
            rng,
            args.bootstrap_samples,
        )
        costs[name] = {
            "reference_median": float(np.median(reference_value)),
            "candidate_median": float(np.median(candidate_value)),
            "candidate_minus_reference_paired_median": point,
            "paired_subject_bootstrap_95_ci": [low, high],
            "fraction_of_subjects_candidate_better": float(
                np.mean(candidate_value < reference_value)
            ),
        }
        for subject_index, subject_id in enumerate(reference.subject_id.astype(str)):
            paired_rows.append(
                {
                    "subject_id": subject_id,
                    "group": reference.group.iloc[subject_index],
                    "metric": name,
                    "reference": reference_value[subject_index],
                    "candidate": candidate_value[subject_index],
                    "candidate_minus_reference": (
                        candidate_value[subject_index] - reference_value[subject_index]
                    ),
                }
            )

    frequency = empirical_collection.frequency_hz
    empirical_blocks = _feature_blocks(empirical_csd, frequency, args.cross_metric)
    reference_blocks = _feature_blocks(
        reference_prediction, frequency, args.cross_metric
    )
    candidate_blocks = _feature_blocks(
        candidate_prediction, frequency, args.cross_metric
    )
    effects = _bootstrap_effect_comparison(
        empirical_blocks,
        reference_blocks,
        candidate_blocks,
        reference.group.astype(str).to_numpy(),
        rng,
        args.bootstrap_samples,
    )
    summary = {
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "subjects": int(len(reference)),
        "cross_metric": args.cross_metric,
        "bootstrap_samples": args.bootstrap_samples,
        "lower_cost_is_better": True,
        "costs": costs,
        "group_effects": effects,
    }
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "paired_subject_costs.csv", index=False
    )
    (output_dir / "model_comparison.json").write_text(
        json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    paired = pd.DataFrame(paired_rows)
    distributions = [
        paired.loc[paired.metric == name, "candidate_minus_reference"].to_numpy()
        for name in COST_COLUMNS
    ]
    axes[0].boxplot(distributions, tick_labels=list(COST_COLUMNS), showfliers=False)
    axes[0].axhline(0.0, color="black", linestyle=":")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Candidate - reference unseen cost ratio")
    axes[0].set_title("Paired subject changes; lower is better")
    feature_names = list(effects)
    correlation_delta = [
        effects[name]["candidate_minus_reference"]["effect_correlation"]
        for name in feature_names
    ]
    norm_delta = [
        effects[name]["candidate_minus_reference"]["effect_norm_retained"]
        for name in feature_names
    ]
    positions = np.arange(len(feature_names))
    axes[1].bar(positions - 0.18, correlation_delta, width=0.36, label="correlation")
    axes[1].bar(positions + 0.18, norm_delta, width=0.36, label="norm retention")
    axes[1].axhline(0.0, color="black", linestyle=":")
    axes[1].set_xticks(positions, feature_names, rotation=20)
    axes[1].set_ylabel("Candidate - reference")
    axes[1].set_title("Healthy–MDD effect preservation")
    axes[1].legend()
    figure.suptitle(f"{args.candidate_name} versus {args.reference_name}")
    figure.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(_json_value(summary), indent=2))


if __name__ == "__main__":
    main()
