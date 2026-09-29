"""T2 thalamic model (per-person loop delay) on every cohort, with wide-band certification.

Per cohort: paired unseen deviance vs M5.4, the loop delay's relation to the
individual alpha frequency and to age (positive controls), and group effects of
the loop parameters (TDBRAIN: batch/clinical/depression decomposition; batch-free
datasets: calibrated random-effects comparison where split halves exist, else OLS).
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/m54"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from m54_evaluate import features  # noqa: E402
from batch_confound import add_covariates, demographics  # noqa: E402

T2 = "outputs/m54t_thal2/results"
COHORTS = {  # name: (T2 run, M5.4 run, emp dir)
    "dev": (f"{T2}/dev_bem_m10popthal2", "outputs/m54b_kaggle_bem/results/dev_bem_m10pop", "dev"),
    "rtms": (f"{T2}/ext_bem_m10popthal2", "outputs/m54b_kaggle_bem/results/ext_bem_m10pop", "rtms"),
    "controls": (f"{T2}/ctrl_controls_bem_m10popthal2", "outputs/m54_ctrl/results/ctrl_bem_m10pop", "controls"),
    "smc": (f"{T2}/ctrl_smc_bem_m10popthal2", "outputs/m54_smc/results/ctrl_smc_bem_m10pop", "smc"),
    "modma": (f"{T2}/modma_bem_m10popthal2", "outputs/m54_newds/results/modma_bem_m10pop", "modma"),
    "mumtaz": (f"{T2}/mumtaz_bem_m10popthal2", "outputs/m54_newds/results/mumtaz_bem_m10pop", "mumtaz"),
    "ds003478": (f"{T2}/ds003478_bem_m10popthal2", "outputs/m54_ds003478/results/ds003478_bem_m10pop", "ds003478"),
}
CONTRAST = {"modma": ("HC", "MDD"), "mumtaz": ("HC", "MDD"), "ds003478": ("LowBDI", "HighBDI")}


def iaf_of(emp, ids):
    c = load_cross_spectral_collection(ROOT / f"outputs/preproc_v2/{emp}_restEC/empirical/cross_spectra_fit.npz")
    lab = list(c.channel_names.astype(str))
    fr = np.asarray(c.frequency_hz, float)
    present = np.flatnonzero(np.real(np.einsum("nfii->ni", c.csd[:5])).min(0) > 0)
    lab_p = [lab[i] for i in present]
    idx = {s: i for i, s in enumerate(c.subject_ids.astype(str))}
    return pd.Series({s: features(c.csd[idx[s]][:, present][:, :, present], fr, lab_p)["iaf"][0] for s in ids if s in idx})


def ages(emp, ids):
    if emp in ("dev", "rtms", "controls", "smc"):
        T = demographics()
        return pd.Series(T.loc[[s for s in ids if s in T.index], "age"].astype(float))
    q = pd.read_csv(ROOT / f"outputs/preproc_v2/{emp}_restEC/empirical/qc.csv").set_index("subject_id")
    return pd.to_numeric(q["age"], errors="coerce") if "age" in q.columns else pd.Series(dtype=float)


def main():
    out = {}
    frames = []
    for name, (t2, m54, emp) in COHORTS.items():
        a = pd.read_csv(ROOT / t2 / "subject_fits.csv").set_index("subject_id")
        b = pd.read_csv(ROOT / m54 / "subject_fits.csv").set_index("subject_id")
        c = a.index.intersection(b.index)
        d = np.log(a.loc[c, "validation_deviance_ratio"]) - np.log(b.loc[c, "validation_deviance_ratio"])
        tot = float((a.loc[c, "validation_deviance_ratio"] * a.loc[c, "null_deviance_per_dof"]).sum()
                    / (b.loc[c, "validation_deviance_ratio"] * b.loc[c, "null_deviance_per_dof"]).sum())
        rec = {"n": int(len(c)), "median_ratio_T2": float(a.validation_deviance_ratio.median()),
               "median_ratio_M54": float(b.loc[c, "validation_deviance_ratio"].median()),
               "better_than_M54": float((d < 0).mean()), "wilcoxon_p": float(st.wilcoxon(d).pvalue),
               "total_deviance_T2_over_M54": tot, "certified_wide": float(a.certified.mean()),
               "delay_ms_mean": float(a.phys_thal_t0_ms.mean()), "delay_ms_sd": float(a.phys_thal_t0_ms.std())}
        iaf = iaf_of(emp, a.index)
        j = iaf.index.intersection(a.index)
        rec["rho_delay_iaf"] = float(st.spearmanr(a.loc[j, "phys_thal_t0_ms"], iaf.loc[j]).correlation)
        rec["p_delay_iaf"] = float(st.spearmanr(a.loc[j, "phys_thal_t0_ms"], iaf.loc[j]).pvalue)
        ag = ages(emp, a.index).dropna()
        j = ag.index.intersection(a.index)
        j = [s for s in j if ag[s] >= 18] if emp in ("dev", "rtms", "controls", "smc") else list(j)
        if len(j) > 20 and np.std(ag.loc[j]) > 2:
            r = st.spearmanr(a.loc[j, "phys_thal_t0_ms"], ag.loc[j])
            rec.update({"rho_delay_age": float(r.correlation), "p_delay_age": float(r.pvalue), "n_age": len(j),
                        "age_range": [float(ag.loc[j].min()), float(ag.loc[j].max())]})
        if name in CONTRAST:
            ref, case = CONTRAST[name]
            g = a[a.group.isin([ref, case])]
            for p in ("thal_t0_ms", "thal_gain", "thal_gamma"):
                y = (g[p] - g[p].mean()) / g[p].std()
                x1, x0 = y[g.group == case], y[g.group == ref]
                rec[f"{p}_case_minus_ref_sd"] = float(x1.mean() - x0.mean())
                rec[f"{p}_p"] = float(st.ttest_ind(x1, x0).pvalue)
        if emp in ("dev", "rtms", "controls", "smc"):
            f = a.copy()
            f["group"] = "rTMS" if name == "rtms" else f["group"]
            frames.append(f)
        out[name] = rec
    x = add_covariates(pd.concat(frames), demographics()).dropna(subset=["age", "sex"])
    dec = {}
    for p in ("thal_t0_ms", "thal_gain", "thal_gamma"):
        d = x.copy()
        d["y"] = (d[p] - d.loc[d.group == "Healthy", p].mean()) / d.loc[d.group.isin(["Healthy", "MDD"]), p].std()
        d["age10"] = d.age / 10
        f = smf.ols("y ~ early + clin + dep + age10 + sex", data=d).fit()
        dec[p] = {k: {"coef": float(f.params[k]), "p": float(f.pvalues[k])} for k in ("early", "clin", "dep", "age10")}
    out["tdbrain_decomposition"] = dec
    dest = ROOT / "outputs/m54t_thal2/replication.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    t = pd.DataFrame({k: v for k, v in out.items() if k != "tdbrain_decomposition"}).T
    pd.set_option("display.width", 250)
    print(t[["n", "median_ratio_T2", "median_ratio_M54", "better_than_M54", "wilcoxon_p", "total_deviance_T2_over_M54",
             "certified_wide", "delay_ms_mean", "rho_delay_iaf", "p_delay_iaf", "rho_delay_age", "p_delay_age"]].round(4).to_string())
    print(t[[c for c in t.columns if c.endswith("_sd") and c.startswith("thal") or c.endswith("_p") and c.startswith("thal")]].dropna(how="all").round(3).to_string())
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
