"""Audit M5 features for reliability, group information, and identifiability.

The group-discrimination analysis is a diagnostic audit only.  Diagnosis is
never supplied to the subject-level TVB parameter-selection objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .features import (
    FeatureCollection,
    FeatureTransformer,
    load_feature_collection,
    load_feature_transformer,
)


@dataclass(frozen=True)
class AuditInputs:
    fitting: FeatureCollection
    validation: FeatureCollection
    bank: dict[str, np.ndarray]
    transformer: FeatureTransformer
    selected_candidates: np.ndarray
    age: np.ndarray
    sex: np.ndarray


def _flatten(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(array.shape[0], -1)


def _blocks(source: FeatureCollection | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def value(name: str) -> np.ndarray:
        return np.asarray(
            getattr(source, name) if isinstance(source, FeatureCollection) else source[name],
            dtype=float,
        )

    spectral_summary = np.column_stack(
        (value("alpha_peak_hz"), value("alpha_power_fraction"), value("spectral_entropy"))
    )
    coherence = np.asarray(value("coherence"), dtype=float)
    result = {
        "PSD shape": _flatten(value("psd_shape")),
        "Spectral summary": spectral_summary,
        "Alpha topography": _flatten(value("alpha_topography")),
        "Coherence": _flatten(coherence),
    }
    result["All current"] = np.concatenate(tuple(result.values()), axis=1)
    for band_index, band_name in enumerate(("delta", "theta", "alpha", "beta")):
        result[f"Coherence {band_name}"] = coherence[:, band_index, :]

    frequency = value("frequency_hz")
    if frequency.ndim != 1:
        frequency = frequency[0]
    log_psd = value("psd_shape")
    aperiodic_mask = (
        ((frequency >= 2.0) & (frequency < 7.0))
        | ((frequency > 14.0) & (frequency <= 45.0))
    )
    centered_log_frequency = np.log10(frequency[aperiodic_mask])
    centered_log_frequency -= centered_log_frequency.mean()
    aperiodic_exponent = -(
        log_psd[:, aperiodic_mask] @ centered_log_frequency
    ) / float(centered_log_frequency @ centered_log_frequency)
    relative_power = np.exp(log_psd)
    total_power = np.trapezoid(relative_power, frequency, axis=1)
    summaries = [aperiodic_exponent]
    for low, high in ((2, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        mask = (frequency >= low) & (frequency < high)
        summaries.append(
            np.trapezoid(relative_power[:, mask], frequency[mask], axis=1)
            / np.maximum(total_power, np.finfo(float).tiny)
        )
    alpha = (frequency >= 8.0) & (frequency < 13.0)
    background = (
        ((frequency >= 6.0) & (frequency < 8.0))
        | ((frequency >= 13.0) & (frequency < 16.0))
    )
    summaries.append(
        np.max(log_psd[:, alpha], axis=1)
        - np.mean(log_psd[:, background], axis=1)
    )
    result["Mechanistic spectrum"] = np.column_stack(summaries)
    return result


def _transformed_blocks(
    source: FeatureCollection | dict[str, np.ndarray],
    transformer: FeatureTransformer,
) -> dict[str, np.ndarray]:
    def value(name: str) -> np.ndarray:
        return np.asarray(
            getattr(source, name) if isinstance(source, FeatureCollection) else source[name],
            dtype=float,
        )

    blocks = transformer.transform_blocks(
        value("psd_shape"),
        value("alpha_peak_hz"),
        value("alpha_power_fraction"),
        value("spectral_entropy"),
        value("alpha_topography"),
        value("coherence"),
    )
    coherence_label = (
        f"Coherence PCA{transformer.coherence_components.shape[0]} (actual fit)"
        if transformer.coherence_transform == "pca"
        else "Coherence reliable edges (actual fit)"
    )
    return {
        coherence_label: blocks[3],
        "Weighted objective (actual fit)": transformer.transform(
            value("psd_shape"),
            value("alpha_peak_hz"),
            value("alpha_power_fraction"),
            value("spectral_entropy"),
            value("alpha_topography"),
            value("coherence"),
        ),
    }


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def _dimension_reliability(fitting: np.ndarray, validation: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_safe_correlation(fitting[:, i], validation[:, i]) for i in range(fitting.shape[1])]
    )


def _profile_reliability(fitting: np.ndarray, validation: np.ndarray) -> np.ndarray:
    if fitting.shape[1] == 1:
        return np.full(fitting.shape[0], np.nan)
    return np.asarray(
        [_safe_correlation(fitting[i], validation[i]) for i in range(fitting.shape[0])]
    )


def _hedges_g(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    healthy = values[labels == 0]
    mdd = values[labels == 1]
    n0, n1 = len(healthy), len(mdd)
    variance = (
        (n0 - 1) * np.var(healthy, axis=0, ddof=1)
        + (n1 - 1) * np.var(mdd, axis=0, ddof=1)
    ) / max(n0 + n1 - 2, 1)
    pooled = np.sqrt(np.maximum(variance, np.finfo(float).eps))
    d = (np.mean(mdd, axis=0) - np.mean(healthy, axis=0)) / pooled
    correction = 1.0 - 3.0 / max(4.0 * (n0 + n1) - 9.0, 1.0)
    return correction * d


def _residualize(
    train: np.ndarray,
    test: np.ndarray,
    train_age: np.ndarray,
    test_age: np.ndarray,
    train_sex: np.ndarray,
    test_sex: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    age_mean = float(np.mean(train_age))
    age_scale = float(np.std(train_age, ddof=1)) or 1.0
    train_design = np.column_stack(
        (np.ones(len(train_age)), (train_age - age_mean) / age_scale, train_sex)
    )
    test_design = np.column_stack(
        (np.ones(len(test_age)), (test_age - age_mean) / age_scale, test_sex)
    )
    coefficients = np.linalg.lstsq(train_design, train, rcond=None)[0]
    # Preserve the training intercept while removing linear age and sex terms.
    return (
        train - train_design[:, 1:] @ coefficients[1:],
        test - test_design[:, 1:] @ coefficients[1:],
    )


def _require_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
        from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        raise RuntimeError(
            "The feature audit requires scikit-learn; install the project 'analysis' extra."
        ) from exc
    return locals()


def _group_signal(
    fitting: np.ndarray,
    validation: np.ndarray,
    labels: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    *,
    residualize: bool,
    seed: int,
    repeats: int,
) -> list[dict[str, float | int]]:
    sk = _require_sklearn()
    splitter = sk["RepeatedStratifiedKFold"](
        n_splits=5, n_repeats=repeats, random_state=seed
    )
    predictions = np.zeros((repeats, len(labels)), dtype=float)
    repeat_folds = np.zeros(repeats, dtype=int)
    for split_number, (train_index, test_index) in enumerate(splitter.split(fitting, labels)):
        repeat = split_number // 5
        x_train = fitting[train_index]
        # A different temporal half is used for every held-out participant.
        x_test = validation[test_index]
        if residualize:
            x_train, x_test = _residualize(
                x_train,
                x_test,
                age[train_index],
                age[test_index],
                sex[train_index],
                sex[test_index],
            )
        candidate_k: list[int | str] = []
        for value in (30, 120):
            if value < x_train.shape[1]:
                candidate_k.append(value)
        candidate_k.append("all")
        pipeline = sk["Pipeline"](
            [
                ("scale", sk["StandardScaler"]()),
                ("select", sk["SelectKBest"](score_func=sk["f_classif"])),
                (
                    "model",
                    sk["LogisticRegression"](
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed + split_number,
                    ),
                ),
            ]
        )
        inner = sk["StratifiedKFold"](
            n_splits=3, shuffle=True, random_state=seed + split_number
        )
        search = sk["GridSearchCV"](
            pipeline,
            {
                "select__k": candidate_k,
                "model__C": [0.03, 0.3, 3.0],
            },
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, labels[train_index])
        predictions[repeat, test_index] = search.predict_proba(x_test)[:, 1]
        repeat_folds[repeat] += 1

    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        if repeat_folds[repeat] != 5:
            raise RuntimeError("Incomplete outer cross-validation repeat")
        probability = predictions[repeat]
        rows.append(
            {
                "repeat": repeat,
                "roc_auc": float(sk["roc_auc_score"](labels, probability)),
                "balanced_accuracy": float(
                    sk["balanced_accuracy_score"](labels, probability >= 0.5)
                ),
                "brier": float(sk["brier_score_loss"](labels, probability)),
            }
        )
    return rows


def _parameter_learnability(
    features: np.ndarray,
    parameters: np.ndarray,
    parameter_names: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, float | str]]:
    sk = _require_sklearn()
    varying = np.ptp(parameters, axis=0) > np.finfo(float).eps
    parameters = parameters[:, varying]
    parameter_names = parameter_names[varying]
    if parameters.shape[1] == 0:
        return []
    splitter = sk["KFold"](n_splits=5, shuffle=True, random_state=seed)
    predictions = np.zeros_like(parameters, dtype=float)
    for fold, (train_index, test_index) in enumerate(splitter.split(features)):
        model = sk["ExtraTreesRegressor"](
            n_estimators=300,
            max_features=0.7,
            min_samples_leaf=2,
            random_state=seed + fold,
            n_jobs=1,
        )
        model.fit(features[train_index], parameters[train_index])
        predictions[test_index] = model.predict(features[test_index])
    rows: list[dict[str, float | str]] = []
    for column, name in enumerate(parameter_names):
        observed = parameters[:, column]
        predicted = predictions[:, column]
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        r2 = 1.0 - float(np.sum((observed - predicted) ** 2)) / denominator
        value_range = float(np.ptp(observed)) or 1.0
        rows.append(
            {
                "parameter": str(name),
                "cross_validated_r2": r2,
                "normalized_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2)) / value_range),
            }
        )
    return rows


def load_audit_inputs(
    fitting_path: Path,
    validation_path: Path,
    bank_path: Path,
    transformer_path: Path,
    subject_fits_path: Path,
    participant_path: Path,
) -> AuditInputs:
    fitting = load_feature_collection(fitting_path)
    validation = load_feature_collection(validation_path)
    if not np.array_equal(fitting.subject_ids, validation.subject_ids):
        raise ValueError("Fitting and validation feature subjects are not aligned")
    with np.load(bank_path) as arrays:
        bank = {name: arrays[name] for name in arrays.files}
    participants = pd.read_csv(participant_path, dtype={"subject_id": str})
    participants = participants.set_index("subject_id").loc[fitting.subject_ids]
    expected = np.where(fitting.groups == "Healthy", "Healthy", "Patient")
    if not np.array_equal(participants["group"].to_numpy(), expected):
        raise ValueError("Participant metadata group labels do not match M5 features")
    subject_fits = pd.read_csv(subject_fits_path, dtype={"subject_id": str})
    if not np.array_equal(subject_fits["subject_id"].to_numpy(), fitting.subject_ids):
        raise ValueError("Subject fit rows do not match M5 feature order")
    return AuditInputs(
        fitting=fitting,
        validation=validation,
        bank=bank,
        transformer=load_feature_transformer(transformer_path),
        selected_candidates=subject_fits["candidate_index"].to_numpy(dtype=int),
        age=participants["age"].to_numpy(dtype=float),
        sex=participants["gender"].to_numpy(dtype=float),
    )


def _plot_audit(
    path: Path,
    reliability: pd.DataFrame,
    group_summary: pd.DataFrame,
    effect_replication: pd.DataFrame,
    learnability: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    block_order = list(reliability["block"])
    colors = ["#3568a8", "#7a5195", "#ef8354", "#3b8c6e", "#555555"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].bar(
        np.arange(len(block_order)),
        reliability["median_dimension_split_half_r"],
        color=colors,
    )
    axes[0, 0].set_xticks(np.arange(len(block_order)), block_order, rotation=24, ha="right")
    axes[0, 0].axhline(0.60, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(-0.05, 1.0)
    axes[0, 0].set_ylabel("Median feature-dimension r")
    axes[0, 0].set_title("Split-half reliability")

    width = 0.36
    x = np.arange(len(block_order))
    raw = group_summary[group_summary["adjustment"] == "none"].set_index("block").loc[block_order]
    adjusted = group_summary[group_summary["adjustment"] == "age_sex"].set_index("block").loc[block_order]
    axes[0, 1].bar(x - width / 2, raw["mean_roc_auc"], width, label="Unadjusted", color="#d95f5f")
    axes[0, 1].bar(x + width / 2, adjusted["mean_roc_auc"], width, label="Age/sex residualized", color="#3568a8")
    axes[0, 1].set_xticks(x, block_order, rotation=24, ha="right")
    axes[0, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(0.4, 0.9)
    axes[0, 1].set_ylabel("Held-out ROC AUC")
    axes[0, 1].set_title("Group-information audit (not a fitting target)")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(
        x,
        effect_replication.set_index("block").loc[block_order, "effect_vector_fit_validation_r"],
        color=colors,
    )
    axes[1, 0].set_xticks(x, block_order, rotation=24, ha="right")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylim(-1.0, 1.0)
    axes[1, 0].set_ylabel("Correlation of Hedges-g vectors")
    axes[1, 0].set_title("Replication of Healthy–MDD feature effects")

    all_current = learnability[learnability["block"] == "All current"].copy()
    all_current = all_current.sort_values("cross_validated_r2")
    axes[1, 1].barh(
        np.arange(len(all_current)),
        all_current["cross_validated_r2"],
        color=np.where(all_current["cross_validated_r2"] > 0, "#3b8c6e", "#c65d57"),
    )
    axes[1, 1].set_yticks(np.arange(len(all_current)), all_current["parameter"])
    axes[1, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("5-fold out-of-simulation R²")
    axes[1, 1].set_title("Can current features identify bank parameters?")

    figure.suptitle("M5 feature audit: stability, group relevance, and model identifiability", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_fitted_group_effects(
    path: Path,
    validation_blocks: dict[str, np.ndarray],
    bank_blocks: dict[str, np.ndarray],
    selected_candidates: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Show whether independently fitted simulations preserve empirical effects."""

    import matplotlib.pyplot as plt

    requested = (
        "PSD shape",
        "Mechanistic spectrum",
        "Alpha topography",
        "Coherence theta",
        "Coherence alpha",
        "Coherence beta",
    )
    blocks = [name for name in requested if name in validation_blocks]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, block in zip(axes.flat, blocks, strict=False):
        empirical = _hedges_g(validation_blocks[block], labels)
        fitted = _hedges_g(
            bank_blocks[block][selected_candidates], labels
        )
        correlation = _safe_correlation(empirical, fitted)
        empirical_norm = float(np.linalg.norm(empirical))
        retention = float(np.linalg.norm(fitted) / max(empirical_norm, np.finfo(float).eps))
        limit = max(float(np.max(np.abs(empirical))), float(np.max(np.abs(fitted))), 0.1)
        axis.scatter(empirical, fitted, s=13, alpha=0.55, color="#3568a8")
        axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1)
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.axvline(0, color="#777777", linewidth=0.7)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_title(f"{block}\nr={correlation:.2f}; effect norm retained={retention:.2f}")
        axis.set_xlabel("Empirical unseen-half Hedges g")
        axis.set_ylabel("Fitted-simulation Hedges g")
    for axis in axes.flat[len(blocks):]:
        axis.axis("off")
    figure.suptitle(
        "Do diagnosis-blind individual fits preserve Healthy–MDD feature effects?",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_feature_audit(
    inputs: AuditInputs,
    output_dir: Path,
    *,
    seed: int = 20260912,
    repeats: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fitting_blocks = _blocks(inputs.fitting)
    validation_blocks = _blocks(inputs.validation)
    bank_blocks = _blocks(inputs.bank)
    fitting_blocks.update(_transformed_blocks(inputs.fitting, inputs.transformer))
    validation_blocks.update(_transformed_blocks(inputs.validation, inputs.transformer))
    bank_blocks.update(_transformed_blocks(inputs.bank, inputs.transformer))
    labels = (inputs.fitting.groups == "MDD").astype(int)

    reliability_rows: list[dict[str, float | int | str]] = []
    effect_rows: list[dict[str, float | int | str]] = []
    group_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | str]] = []
    for block, fitting in fitting_blocks.items():
        validation = validation_blocks[block]
        dimension_r = _dimension_reliability(fitting, validation)
        profile_r = _profile_reliability(fitting, validation)
        reliability_rows.append(
            {
                "block": block,
                "dimensions": fitting.shape[1],
                "median_dimension_split_half_r": float(np.nanmedian(dimension_r)),
                "dimension_fraction_r_at_least_0_60": float(np.mean(dimension_r >= 0.60)),
                "median_within_subject_profile_r": float(np.nanmedian(profile_r)),
            }
        )
        fit_g = _hedges_g(fitting, labels)
        validation_g = _hedges_g(validation, labels)
        strong = np.abs(fit_g) >= 0.20
        effect_rows.append(
            {
                "block": block,
                "effect_vector_fit_validation_r": _safe_correlation(fit_g, validation_g),
                "median_absolute_fit_hedges_g": float(np.median(np.abs(fit_g))),
                "maximum_absolute_fit_hedges_g": float(np.max(np.abs(fit_g))),
                "dimensions_fit_abs_g_at_least_0_20": int(np.sum(strong)),
                "strong_effect_sign_replication_fraction": float(
                    np.mean(np.sign(fit_g[strong]) == np.sign(validation_g[strong]))
                    if np.any(strong)
                    else np.nan
                ),
            }
        )
        for adjustment, use_residuals in (("none", False), ("age_sex", True)):
            for row in _group_signal(
                fitting,
                validation,
                labels,
                inputs.age,
                inputs.sex,
                residualize=use_residuals,
                seed=seed,
                repeats=repeats,
            ):
                group_rows.append({"block": block, "adjustment": adjustment, **row})
        for row in _parameter_learnability(
            bank_blocks[block],
            np.asarray(inputs.bank["parameters"], dtype=float),
            np.asarray(inputs.bank["parameter_names"]),
            seed=seed,
        ):
            parameter_rows.append({"block": block, **row})

    reliability = pd.DataFrame(reliability_rows)
    effect_replication = pd.DataFrame(effect_rows)
    group_signal = pd.DataFrame(group_rows)
    group_summary = (
        group_signal.groupby(["block", "adjustment"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_brier=("brier", "mean"),
        )
    )
    learnability = pd.DataFrame(parameter_rows)
    reliability.to_csv(output_dir / "split_half_reliability.csv", index=False)
    effect_replication.to_csv(output_dir / "group_effect_replication.csv", index=False)
    group_signal.to_csv(output_dir / "group_signal_repeats.csv", index=False)
    group_summary.to_csv(output_dir / "group_signal_summary.csv", index=False)
    learnability.to_csv(output_dir / "simulation_parameter_learnability.csv", index=False)

    coverage_rows: list[dict[str, float | int | str | bool]] = []
    for block in fitting_blocks:
        empirical = fitting_blocks[block]
        simulated = bank_blocks[block]
        healthy_centroid = empirical[labels == 0].mean(axis=0)
        mdd_centroid = empirical[labels == 1].mean(axis=0)
        contrast = mdd_centroid - healthy_centroid
        norm = float(np.linalg.norm(contrast))
        direction = contrast / max(norm, np.finfo(float).eps)
        bank_projection = simulated @ direction
        healthy_projection = float(healthy_centroid @ direction)
        mdd_projection = float(mdd_centroid @ direction)
        coverage_rows.append(
            {
                "block": block,
                "empirical_centroid_separation": norm,
                "healthy_projection": healthy_projection,
                "mdd_projection": mdd_projection,
                "bank_projection_min": float(bank_projection.min()),
                "bank_projection_max": float(bank_projection.max()),
                "bank_covers_healthy_centroid": bool(
                    bank_projection.min() <= healthy_projection <= bank_projection.max()
                ),
                "bank_covers_mdd_centroid": bool(
                    bank_projection.min() <= mdd_projection <= bank_projection.max()
                ),
                "nearest_healthy_candidate": int(
                    np.argmin(np.mean((simulated - healthy_centroid) ** 2, axis=1))
                ),
                "nearest_mdd_candidate": int(
                    np.argmin(np.mean((simulated - mdd_centroid) ** 2, axis=1))
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "simulation_group_axis_coverage.csv", index=False)

    fitted_effect_rows: list[dict[str, float | str]] = []
    for block in fitting_blocks:
        empirical_effect = _hedges_g(validation_blocks[block], labels)
        selected_simulations = bank_blocks[block][inputs.selected_candidates]
        simulated_effect = _hedges_g(selected_simulations, labels)
        fitted_effect_rows.append(
            {
                "block": block,
                "empirical_vs_fitted_effect_vector_r": _safe_correlation(
                    empirical_effect, simulated_effect
                ),
                "empirical_group_centroid_distance": float(
                    np.linalg.norm(
                        validation_blocks[block][labels == 1].mean(axis=0)
                        - validation_blocks[block][labels == 0].mean(axis=0)
                    )
                ),
                "fitted_group_centroid_distance": float(
                    np.linalg.norm(
                        selected_simulations[labels == 1].mean(axis=0)
                        - selected_simulations[labels == 0].mean(axis=0)
                    )
                ),
                "maximum_absolute_fitted_hedges_g": float(
                    np.max(np.abs(simulated_effect))
                ),
            }
        )
    fitted_effects = pd.DataFrame(fitted_effect_rows)
    fitted_effects.to_csv(
        output_dir / "fitted_group_effect_preservation.csv", index=False
    )
    _plot_fitted_group_effects(
        output_dir / "m5_fitted_group_effects.png",
        validation_blocks,
        bank_blocks,
        inputs.selected_candidates,
        labels,
    )
    _plot_audit(
        output_dir / "m5_feature_audit.png",
        reliability,
        group_summary,
        effect_replication,
        learnability,
    )

    effect_thresholds = {
        "PSD shape": 0.40,
        "Alpha topography": 0.30,
        "Coherence alpha": 0.30,
        "Coherence beta": 0.30,
    }
    effect_lookup = fitted_effects.set_index("block")[
        "empirical_vs_fitted_effect_vector_r"
    ]
    quality_checks = {
        block: bool(effect_lookup.get(block, -np.inf) >= threshold)
        for block, threshold in effect_thresholds.items()
    }
    summary = {
        "purpose": (
            "Audit feature blocks for stability, empirical group information, and "
            "simulation-bank parameter identifiability. Diagnosis is not a fit target."
        ),
        "subjects": int(len(labels)),
        "healthy": int(np.sum(labels == 0)),
        "mdd_indication": int(np.sum(labels == 1)),
        "cross_validation": (
            "Repeated nested 5-fold subject CV; train uses fitting halves and each held-out "
            "subject is scored from its unseen temporal half."
        ),
        "repeats": repeats,
        "reliability": reliability.to_dict(orient="records"),
        "group_signal": group_summary.to_dict(orient="records"),
        "group_effect_replication": effect_replication.to_dict(orient="records"),
        "parameter_identifiability": {
            block: {
                "median_cross_validated_r2": float(rows["cross_validated_r2"].median()),
                "positive_r2_parameters": int(np.sum(rows["cross_validated_r2"] > 0)),
                "parameters": int(len(rows)),
            }
            for block, rows in learnability.groupby("block")
        },
        "simulation_group_axis_coverage": coverage.to_dict(orient="records"),
        "fitted_group_effect_preservation": fitted_effects.to_dict(orient="records"),
        "fit_quality_gate": {
            "status": "pass" if all(quality_checks.values()) else "fail",
            "exploratory_not_preregistered": True,
            "minimum_effect_vector_correlations": effect_thresholds,
            "checks": quality_checks,
            "interpretation": (
                "A failed gate forbids mechanistic Healthy–MDD or stimulation-target "
                "claims from this fitted model, even if individual fit loss improves."
            ),
        },
        "guardrail": (
            "A feature may enter the fit only if it is reproducible, represented by the "
            "simulator, and informative about at least one fitted parameter. Group "
            "information is evaluated after fitting and is never optimized directly."
        ),
    }
    (output_dir / "feature_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
