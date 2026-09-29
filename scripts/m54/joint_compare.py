"""Compare joint eyes-closed + eyes-open fits that differ in which parameters may change with eyes open."""
from __future__ import annotations

import glob, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[2]


def main(pattern="outputs/m54e_joint*/results/joint_dev_eoec_compare100_bem_*/subject_fits.csv", reference="none"):
    fits = {}
    for f in sorted(glob.glob(str(ROOT / pattern))):
        name = re.search(r"_bem_([a-z_]+?)(?:_swap)?[\\/]subject_fits", f).group(1)
        fits[name] = pd.read_csv(f).set_index("subject_id")
    base = fits[reference].validation_deviance_joint
    rows = []
    for d, x in fits.items():
        diff = x.validation_deviance_joint - base.loc[x.index]
        deltas = [c for c in x.columns if c.startswith("delta_") and c != "delta_set"]
        rows.append({"delta": d, "n": len(x), "total_deviance": float(x.validation_deviance_joint.sum()),
                     "vs_none_total_pct": float(100 * diff.sum() / base.loc[x.index].sum()),
                     "better_than_none": float((diff < 0).mean()),
                     "wilcoxon_p_vs_none": float(st.wilcoxon(diff).pvalue) if d != reference else np.nan,
                     "median_ratio_ec": float(x.validation_deviance_ratio_ec.median()),
                     "median_ratio_eo": float(x.validation_deviance_ratio_eo.median()),
                     "r_power_ratio_model_data": float(np.corrcoef(x.log_eo_over_ec_power_model, x.log_eo_over_ec_power_data)[0, 1]),
                     "model_log_eo_ec": float(x.log_eo_over_ec_power_model.mean()),
                     "data_log_eo_ec": float(x.log_eo_over_ec_power_data.mean()),
                     "changes": {c[6:]: {"mean": float(x[c].mean()), "t": float(x[c].mean() / x[c].std() * np.sqrt(len(x))),
                                         "median_posterior_sd": float(x[f"sd_{c}"].median())} for c in deltas}})
    t = pd.DataFrame(rows).sort_values("total_deviance")
    out = ROOT / "outputs/m54_eoec/joint_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    pd.set_option("display.width", 220)
    print(t.drop(columns="changes").round(4).to_string(index=False))
    for r in rows:
        if r["changes"]:
            top = sorted(r["changes"].items(), key=lambda kv: -abs(kv[1]["t"]))[:8]
            print(r["delta"], {k: (round(v["mean"], 3), round(v["t"], 1), round(v["median_posterior_sd"], 2)) for k, v in top})


if __name__ == "__main__":
    main(*sys.argv[1:])
