"""Specificity and acquisition-batch analysis (NEXT_STEPS report section 7).

1. Left dorsal-attention input gain (final M5.4 model) by cohort, age/sex adjusted, vs Healthy.
2. Joint model: early acquisition batch (subject ID < 88000000) + clinical + depression.
3. The raw scalp correlate (relative P3/CP3 vs F7/F8/T7/T8 power) in the same model.
4. Correlation of every Healthy-MDD EEG effect with the batch effect estimated without depressed subjects.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/m54"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from m54_evaluate import features  # noqa: E402

TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
FITS = {"dev": "outputs/m54b_kaggle_bem/results/dev_bem_m10pop/subject_fits.csv",
        "rtms": "outputs/m54b_kaggle_bem/results/ext_bem_m10pop/subject_fits.csv",
        "controls": "outputs/m54_ctrl/results/ctrl_bem_m10pop/subject_fits.csv",
        "smc": "outputs/m54_smc/results/ctrl_smc_bem_m10pop/subject_fits.csv"}
BATCH_ID = 88000000
P = "gain_DorsAttn_LH"


def demographics():
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    return T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")


def add_covariates(x, T):
    x["age"] = T.loc[x.index, "age"].astype(float).to_numpy()
    x["sex"] = T.loc[x.index, "gender"].astype(float).to_numpy()
    x["early"] = (np.array([int(s[4:]) for s in x.index]) < BATCH_ID).astype(int)
    x["clin"] = (x.group != "Healthy").astype(int)
    x["dep"] = x.group.isin(["MDD", "rTMS"]).astype(int)
    return x


def coef_table(fit, names):
    ci = fit.conf_int()
    return {n: {"coef": float(fit.params[n]), "ci": [float(ci.loc[n, 0]), float(ci.loc[n, 1])], "p": float(fit.pvalues[n])}
            for n in names}


def main():
    T = demographics()
    frames = []
    for cohort, path in FITS.items():
        t = pd.read_csv(ROOT / path).set_index("subject_id")
        g = t["group"] if "group" in t.columns else pd.Series("rTMS", index=t.index)
        frames.append(pd.DataFrame({"group": np.where(cohort == "rtms", "rTMS", g), P: t[P]}, index=t.index))
    x = add_covariates(pd.concat(frames), T)
    sd = x.loc[x.group.isin(["Healthy", "MDD"]), P].std()
    x["y"] = (x[P] - x.loc[x.group == "Healthy", P].mean()) / sd
    out = {"n": int(len(x)), "group_means_sd": x.groupby("group").y.mean().round(3).to_dict()}
    f = smf.ols("y ~ C(group, Treatment('Healthy')) + age + sex", data=x.dropna(subset=["age", "sex"])).fit()
    out["vs_healthy"] = coef_table(f, [k for k in f.params.index if "group" in k])
    f = smf.ols("y ~ early + clin + dep + age + sex", data=x.dropna(subset=["age", "sex"])).fit()
    out["joint_parameter"] = coef_table(f, ["early", "clin", "dep"])
    # raw scalp correlate and the effect-correlation table from the v2 spectra
    rows, feats = [], []
    dev_ids = set(pd.read_csv(ROOT / "configs/m52_nested_splits.csv").subject_id.astype(str))
    for cohort in FITS:
        c = load_cross_spectral_collection(ROOT / f"outputs/preproc_v2/{cohort}_restEC/empirical/cross_spectra_fit.npz")
        lab = list(c.channel_names.astype(str)); fr = np.asarray(c.frequency_hz, float)
        for s, g, csd in zip(c.subject_ids.astype(str), c.groups.astype(str), c.csd):
            if cohort == "dev" and s not in dev_ids:
                continue
            ft = features(csd, fr, lab)
            pw = np.real(np.einsum("fii->fi", csd))
            rel = {}
            for band, (lo, hi) in {"alpha": (8, 12), "beta": (13, 19)}.items():
                bp = np.log(pw[(fr >= lo) & (fr <= hi)].mean(0)); bp -= bp.mean()
                rel[band] = bp[[lab.index("P3"), lab.index("CP3")]].mean() - bp[[lab.index(k) for k in ("F7", "F8", "T7", "T8")]].mean()
            rows.append({"subject_id": s, "group": "rTMS" if cohort == "rtms" else g, **rel})
            feats.append(ft)
    m = add_covariates(pd.DataFrame(rows).set_index("subject_id"), T)
    for band in ("alpha", "beta"):
        m[f"z_{band}"] = (m[band] - m.loc[m.group == "Healthy", band].mean()) / m.loc[m.group.isin(["Healthy", "MDD"]), band].std()
        f = smf.ols(f"z_{band} ~ early + clin + dep + age + sex", data=m.dropna(subset=["age", "sex"])).fit()
        out[f"joint_scalp_{band}"] = coef_table(f, ["early", "clin", "dep"])
    cov = m[["age", "sex"]].fillna(m[["age", "sex"]].mean()).to_numpy()
    X = np.column_stack([np.ones(len(cov)), cov])
    g = m.group.to_numpy(); e = m.early.to_numpy().astype(bool); nondep = ~np.isin(g, ["MDD", "rTMS"])
    table = {}
    for k in ["topo_alpha", "topo_beta", "topo_theta", "zerolag_alpha", "zerolag_beta", "lagged_alpha", "spectrum"]:
        A = np.stack([ft[k] for ft in feats])
        b, *_ = np.linalg.lstsq(X, A, rcond=None)
        A = A - X[:, 1:] @ b[1:]
        dep_eff = A[g == "MDD"].mean(0) - A[g == "Healthy"].mean(0)
        batch = A[nondep & ~e].mean(0) - A[nondep & e].mean(0)
        within = A[np.isin(g, ["MDD", "rTMS"])].mean(0) - A[nondep & ~e & (g != "Healthy")].mean(0)
        table[k] = {"r_mdd_batch": float(np.corrcoef(dep_eff, batch)[0, 1]),
                    "batch_over_mdd_norm": float(np.linalg.norm(batch) / np.linalg.norm(dep_eff)),
                    "r_mdd_batch_matched": float(np.corrcoef(dep_eff, within)[0, 1]),
                    "batch_matched_over_mdd_norm": float(np.linalg.norm(within) / np.linalg.norm(dep_eff))}
    out["effect_vs_batch"] = table
    dest = ROOT / "outputs/m54_final/batch_confound.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
