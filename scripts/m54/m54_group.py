"""Second-level analysis of M5.4 posteriors.

1. Split-half calibration: for every parameter, z = (theta_1 - theta_2) /
   sqrt(sd_1^2 + sd_2^2) from independent fits of the two halves.  var(z) = 1 for
   calibrated Laplace posteriors; kappa = var(z) is the misspecification
   inflation applied below.  The test-retest ICC(3,1) of each parameter is
   reported too.
2. Random-effects group comparison (empirical-Bayes / PEB-style): per parameter,
   theta_i = b0 + b_group * MDD_i + b_age * age_i + b_sex * sex_i + u_i + e_i with
   e_i ~ N(0, kappa * sd_i^2) and u_i ~ N(0, tau^2), tau^2 by restricted maximum
   likelihood; Holm correction across parameters.
Parameters are the free, identifiable ones; the tied limbic gain is excluded.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")


def parameter_columns(t):
    base = ["global_coupling", "mu", "a_scale", "b_scale", "vis_time_contrast", "dorsattn_time_contrast",
            "obs_fraction", "obs_exponent", "src_fraction", "src_exponent", "common_share"]
    gains = [c for c in t.columns if c.startswith("gain_") and "Limbic" not in c]
    return [(c, f"sd_{c}") for c in base + gains]


def icc31(a, b):
    x = np.column_stack([a, b])
    n = len(x)
    ms_r = 2 * np.var(x.mean(1), ddof=1)
    ms_e = np.sum((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()) ** 2) / (n - 1)
    return (ms_r - ms_e) / (ms_r + ms_e)


def reml_meta(y, v, X):
    """Random-effects meta-regression; returns beta, cov(beta), tau^2."""
    def fit(tau2):
        w = 1.0 / (v + tau2)
        XtW = X.T * w
        cov = np.linalg.inv(XtW @ X)
        beta = cov @ (XtW @ y)
        r = y - X @ beta
        ll = -0.5 * (np.sum(np.log(v + tau2)) + np.sum(w * r**2) + np.linalg.slogdet(XtW @ X)[1])
        return beta, cov, ll
    res = minimize_scalar(lambda t: -fit(t)[2], bounds=(0.0, 10.0 * np.var(y) + 1e-9), method="bounded")
    beta, cov, _ = fit(res.x)
    return beta, cov, float(res.x)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="outputs/m54/dev_tvb/subject_fits.csv")
    ap.add_argument("--second", default="outputs/m54/dev_tvb_swap/subject_fits.csv")
    ap.add_argument("--out", default="outputs/m54/group_tvb")
    ap.add_argument("--min-kappa", type=float, default=1.0)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    t1 = pd.read_csv(ROOT / args.first).set_index("subject_id")
    t2 = pd.read_csv(ROOT / args.second).set_index("subject_id") if (ROOT / args.second).is_file() else None
    cols = parameter_columns(t1)
    calib = {}
    if t2 is not None:
        common = t1.index.intersection(t2.index)
        for c, s in cols:
            z = (t1.loc[common, c] - t2.loc[common, c]) / np.sqrt(t1.loc[common, s] ** 2 + t2.loc[common, s] ** 2)
            z = z.replace([np.inf, -np.inf], np.nan).dropna()
            calib[c] = {"var_z": float(np.var(z)), "robust_var_z": float((1.4826 * np.median(np.abs(z - np.median(z)))) ** 2),
                        "icc_test_retest": float(icc31(t1.loc[common, c], t2.loc[common, c])),
                        "median_posterior_sd": float(t1.loc[common, s].median()), "n": int(len(z))}
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    d = t1[t1.group.isin(["Healthy", "MDD"])].copy()
    d["mdd"] = (d.group == "MDD").astype(float)
    d["age"] = T.loc[d.index, "age"].astype(float).to_numpy()
    d["sex"] = T.loc[d.index, "gender"].astype(float).to_numpy()
    d = d.dropna(subset=["age", "sex"])
    X = np.column_stack([np.ones(len(d)), d.mdd, (d.age - d.age.mean()) / d.age.std(), d.sex - d.sex.mean()])
    rows = []
    for c, s in cols:
        kappa = max(args.min_kappa, calib.get(c, {}).get("robust_var_z", 1.0))
        y = d[c].to_numpy(float)
        v = kappa * d[s].to_numpy(float) ** 2
        beta, cov, tau2 = reml_meta(y, v, X)
        sd_between = np.sqrt(tau2 + np.mean(v))
        se = np.sqrt(np.diag(cov))
        zval = beta[1] / se[1]
        rows.append({"parameter": c, "kappa": kappa, "tau": np.sqrt(tau2), "mdd_minus_healthy": beta[1],
                     "effect_sd_units": beta[1] / sd_between, "se": se[1], "z": zval,
                     "p": 2 * norm.sf(abs(zval)), "age_effect_per_sd": beta[2], "age_p": 2 * norm.sf(abs(beta[2] / se[2])),
                     "icc": calib.get(c, {}).get("icc_test_retest", np.nan)})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.to_numpy())
    res.to_csv(out / "group_effects.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    if calib:
        print(pd.DataFrame(calib).T.round(3).to_string())


if __name__ == "__main__":
    main()
