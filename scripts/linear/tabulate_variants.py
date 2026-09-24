"""Paired per-block comparison of many nested runs with the M5.1 baseline (CSV only)."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = (
    ("total", "validation_cost_ratio_to_pooled_null", "validation_ratio"),
    ("auto", "validation_auto_spectrum_cost_ratio_to_pooled_null", "validation_auto_ratio"),
    ("lagged", "validation_complex_coherency_cost_ratio_to_pooled_null", "validation_cross_ratio"),
    ("topo", "validation_alpha_topography_cost_ratio_to_pooled_null", "validation_topo_ratio"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path/to/subject_fits.csv")
    ap.add_argument("--out", default="outputs/linear_regime/variant_table.csv")
    args = ap.parse_args()
    base = pd.read_csv(ROOT / "outputs/m52_nested_baseline/out_of_fold_subject_posteriors.csv")
    rng = np.random.default_rng(0)
    rows = [{"variant": "M5.1 bank (baseline)", "n": len(base),
             **{f"{b}": float(base[c].median()) for b, c, _ in BLOCKS},
             "frac_below_null_total": float((base[BLOCKS[0][1]] < 1).mean())}]
    for spec in args.runs:
        label, path = spec.split("=", 1)
        new = pd.read_csv(ROOT / path)
        m = base.merge(new, on="subject_id", suffixes=("_base", ""))
        row = {"variant": label, "n": len(m), "frac_below_null_total": float((m.validation_ratio < 1).mean())}
        for b, bc, nc in BLOCKS:
            diff = m[nc].to_numpy() - m[bc].to_numpy()
            boot = [np.median(rng.choice(diff, diff.size)) for _ in range(2000)]
            row[b] = float(m[nc].median())
            row[f"{b}_paired_change"] = float(np.median(diff))
            row[f"{b}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"{b}_ci_high"] = float(np.quantile(boot, 0.975))
        if "observation_noise_fraction" in m:
            row["obs_noise_fraction"] = float(m.observation_noise_fraction.median())
        if "common_drive_share" in m:
            row["common_drive_share"] = float(m.common_drive_share.median())
        rows.append(row)
    table = pd.DataFrame(rows)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table[["variant", "n", "total", "auto", "lagged", "topo", "frac_below_null_total"]
                    + [c for c in ("obs_noise_fraction", "common_drive_share") if c in table]].round(3).to_string(index=False))
        print()
        for _, r in table.iloc[1:].iterrows():
            print(f"{r.variant:45s} " + "  ".join(
                f"{b}: {r[b + '_paired_change']:+.3f} [{r[b + '_ci_low']:+.3f},{r[b + '_ci_high']:+.3f}]" for b, _, _ in BLOCKS))


if __name__ == "__main__":
    main()
