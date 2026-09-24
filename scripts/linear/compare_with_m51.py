"""Paired comparison of a linear-regime nested fit with the M5.1/M5.2 baseline.

Both methods are scored on exactly the same 262 development subjects, the same
outer folds, the same fold-specific feature transformers and pooled nulls,
and only on the untouched second halves.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.spectral_fit import _plot_group_effects  # noqa: E402

BLOCKS = (
    ("total", "validation_cost_ratio_to_pooled_null", "validation_ratio"),
    ("autospectrum", "validation_auto_spectrum_cost_ratio_to_pooled_null", "validation_auto_ratio"),
    ("lagged coherency", "validation_complex_coherency_cost_ratio_to_pooled_null", "validation_cross_ratio"),
    ("alpha topography", "validation_alpha_topography_cost_ratio_to_pooled_null", "validation_topo_ratio"),
)


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    diff = b - a
    boot = [np.median(rng.choice(diff, diff.size)) for _ in range(2000)]
    return {
        "baseline_median": float(np.median(a)),
        "new_median": float(np.median(b)),
        "paired_median_change": float(np.median(diff)),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "fraction_subjects_improved": float(np.mean(diff < 0)),
        "new_fraction_below_null": float(np.mean(b < 1.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/linear_regime/nested_tvb_network")
    ap.add_argument("--baseline", default="outputs/m52_nested_baseline")
    ap.add_argument("--label", default="linear-regime analytic (TVB lead, 7 network gains)")
    args = ap.parse_args()
    run = ROOT / args.run
    base_dir = ROOT / args.baseline
    new = pd.read_csv(run / "subject_fits.csv")
    base = pd.read_csv(base_dir / "out_of_fold_subject_posteriors.csv")
    merged = base.merge(new, on="subject_id", suffixes=("_base", ""))
    rng = np.random.default_rng(0)
    result = {"label": args.label, "subjects": int(len(merged)), "blocks": {}}
    for name, bcol, ncol in BLOCKS:
        result["blocks"][name] = paired(merged[bcol].to_numpy(), merged[ncol].to_numpy(), rng)
    per_fold = []
    for fold, rows in merged.groupby("outer_fold"):
        per_fold.append({"fold": int(fold), **{name: float(rows[ncol].median()) for name, _, ncol in BLOCKS}})
    result["per_fold_new_medians"] = per_fold

    # group-effect preservation with the unseen halves of all dev subjects
    emp = ROOT / "outputs/m5_spectral_m51_production/empirical"
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    pred = np.load(run / "predictions.npz")
    pred_ids = pred["subject_ids"].astype(str)
    order = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    idx = np.asarray([order[s] for s in pred_ids])
    from mdd_tvb.spectral_features import subset_cross_spectral_collection
    val_sub = subset_cross_spectral_collection(val, idx)
    result["group_effects_new"] = _plot_group_effects(
        run / "group_effects.png", val_sub, pred["csd"], None, "lagged_coherency")
    base_summary = json.loads((base_dir / "nested_summary.json").read_text())
    result["group_effects_baseline"] = base_summary["aggregate"]["group_effects"]

    # figure: per-block paired distributions
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, (name, bcol, ncol) in zip(axes, BLOCKS):
        a = merged[bcol].to_numpy()
        b = merged[ncol].to_numpy()
        lim = np.quantile(np.r_[a, b], [0.01, 0.99])
        ax.scatter(a, b, s=9, alpha=0.5)
        ax.plot(lim, lim, color="k", lw=0.8)
        ax.axhline(1, color="C3", ls=":", lw=0.8)
        ax.axvline(1, color="C3", ls=":", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        r = result["blocks"][name]
        ax.set_title(f"{name}\nmedian {r['baseline_median']:.3f} -> {r['new_median']:.3f}")
        ax.set_xlabel("M5.1 bank (unseen cost / pooled null)")
        ax.set_ylabel("new method")
    fig.suptitle(f"Unseen second halves, 262 development subjects, 5 outer folds: {args.label}")
    fig.savefig(run / "paired_comparison.png", dpi=150)
    plt.close(fig)
    (run / "comparison.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
