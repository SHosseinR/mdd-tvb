"""Summarise the M5.4 fits of the batch-free datasets.

Per dataset (MODMA, Mumtaz, ds003478): unseen deviance vs the dataset's own
population null, interpretable blocks (m54_evaluate with the dataset's folds),
and the calibrated random-effects group comparison of every parameter
(m54_group with split-half calibration).  Then a fixed-effect inverse-variance
meta-analysis across the three datasets for every parameter, shown next to the
TDBRAIN estimates (Healthy vs MDD, and the batch-matched MDD vs other late-batch
patients) for the left dorsal-attention gain and the other parameters.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
DATASETS = {"modma": ("HC", "MDD", "outputs/m54_newds/results"), "mumtaz": ("HC", "MDD", "outputs/m54_newds/results"),
            "ds003478": ("LowBDI", "HighBDI", "outputs/m54_ds003478/results")}


def run(args):
    return subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True, env={"PYTHONIOENCODING": "utf-8",
                                                                                        **__import__("os").environ})


def main():
    out = ROOT / "outputs/m54_newds_summary"
    out.mkdir(parents=True, exist_ok=True)
    summary, effects = {}, {}
    for ds, (ref, case, res) in DATASETS.items():
        fits = ROOT / res / f"{ds}_bem_m10pop" / "subject_fits.csv"
        if not fits.is_file():
            print("missing", fits)
            continue
        s = json.loads((fits.parent / "summary.json").read_text())
        emp = f"outputs/preproc_v2/{ds}_restEC/empirical"
        ev = out / f"blocks_{ds}"
        r = run(["scripts/m54/m54_evaluate.py", f"M54={res}/{ds}_bem_m10pop", "--emp-dir", emp,
                 "--subjects-file", f"configs/m54_lists/{ds}_restEC_v2.txt", "--splits", f"configs/splits_{ds}.csv",
                 "--groups", ref, case, "--out", str(ev.relative_to(ROOT))])
        blocks = json.loads((ev / "evaluation.json").read_text())["summaries"][0] if (ev / "evaluation.json").is_file() else {}
        gdir = out / f"group_{ds}"
        swap = ROOT / res / f"{ds}_bem_m10pop_swap" / "subject_fits.csv"
        r = run(["scripts/m54/m54_group.py", "--first", str(fits.relative_to(ROOT)), "--second",
                 str(swap.relative_to(ROOT)) if swap.is_file() else "none", "--qc", f"{emp}/qc.csv",
                 "--groups", ref, case, "--out", str(gdir.relative_to(ROOT))])
        if r.returncode:
            print(r.stderr[-2000:])
        g = pd.read_csv(gdir / "group_effects.csv")
        effects[ds] = g.set_index("parameter")
        summary[ds] = {"subjects": s["subjects"], "median_validation_deviance_ratio": s["median_validation_deviance_ratio"],
                       "median_persistence_deviance_ratio": s["median_persistence_deviance_ratio"],
                       "fraction_beating_null": s["fraction_beating_null"],
                       "blocks": blocks.get("median_ratio", {}), "block_group_effects": blocks.get("group_effects", {}),
                       "strongest_parameter_effects": g.sort_values("p").head(5)[["parameter", "effect_sd_units", "p", "p_holm"]]
                       .to_dict("records")}
        print(ds, json.dumps({k: v for k, v in summary[ds].items() if k not in ("blocks", "block_group_effects")},
                             default=float)[:900])
    # meta-analysis across datasets (effects in between-subject SD units; SE rescaled the same way)
    tdb = pd.read_csv(ROOT / "outputs/m54_final/group_bem_m10pop/group_effects.csv").set_index("parameter")
    rows = []
    for p in tdb.index:
        es, ses = [], []
        for ds, g in effects.items():
            if p in g.index:
                scale = g.loc[p, "effect_sd_units"] / g.loc[p, "mdd_minus_healthy"] if g.loc[p, "mdd_minus_healthy"] else np.nan
                es.append(g.loc[p, "effect_sd_units"]); ses.append(abs(g.loc[p, "se"] * scale))
        es, ses = np.asarray(es), np.asarray(ses)
        ok = np.isfinite(es) & np.isfinite(ses) & (ses > 0)
        if ok.sum() == 0:
            continue
        w = 1 / ses[ok] ** 2
        m, se = float(np.sum(w * es[ok]) / w.sum()), float(np.sqrt(1 / w.sum()))
        rows.append({"parameter": p, "meta_effect_sd": m, "meta_ci": [m - 1.96 * se, m + 1.96 * se],
                     "meta_p": float(2 * norm.sf(abs(m / se))), "n_datasets": int(ok.sum()),
                     **{f"{ds}_effect_sd": float(g.loc[p, "effect_sd_units"]) for ds, g in effects.items() if p in g.index},
                     "tdbrain_effect_sd": float(tdb.loc[p, "effect_sd_units"]), "tdbrain_p_holm": float(tdb.loc[p, "p_holm"])})
    meta = pd.DataFrame(rows)
    if len(meta):
        from statsmodels.stats.multitest import multipletests
        meta["meta_p_holm"] = multipletests(meta.meta_p, method="holm")[1]
        meta.to_csv(out / "meta_analysis.csv", index=False)
        print(meta[["parameter", "meta_effect_sd", "meta_p", "meta_p_holm", "tdbrain_effect_sd"]].round(3).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
