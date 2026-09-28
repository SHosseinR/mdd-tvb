"""Positive control: do the fitted M5.4 parameters track age?

Age changes the resting EEG in well-known ways (slower and weaker alpha, weaker
alpha reactivity), and van Albada et al. (2010) recovered age trends from 1498
eyes-closed spectra with a corticothalamic model.  If our parameters carry
biology, they should show age trends that (a) replicate across cohorts and
(b) predict age in unseen people about as well as plain spectral features.

1. TDBRAIN adults (>= 18 y; development, rTMS, late clinical controls, SMC):
   parameter ~ age + sex + group + early batch, standardised age slope per decade.
2. Replication in MODMA (other EEG system, ages 16-56): parameter ~ age + sex + group.
3. Out-of-sample age prediction (ridge, penalty chosen by inner CV) trained on
   TDBRAIN development + controls + SMC adults, tested on the rTMS cohort, from
   (i) model parameters, (ii) descriptive spectral features, (iii) both.
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
from batch_confound import add_covariates, demographics  # noqa: E402
from m54_evaluate import features  # noqa: E402

FITS = {"dev": "outputs/m54b_kaggle_bem/results/dev_bem_m10pop/subject_fits.csv",
        "rtms": "outputs/m54b_kaggle_bem/results/ext_bem_m10pop/subject_fits.csv",
        "controls": "outputs/m54_ctrl/results/ctrl_bem_m10pop/subject_fits.csv",
        "smc": "outputs/m54_smc/results/ctrl_smc_bem_m10pop/subject_fits.csv"}
MODMA = "outputs/m54_newds/results/modma_bem_m10pop/subject_fits.csv"
NEURAL = ["global_coupling", "mu", "a_scale", "b_scale", "vis_time_contrast", "dorsattn_time_contrast"]
NUIS = ["common_share", "pop_share", "obs_fraction", "src_fraction"]


def params(t):
    gains = [c for c in t.columns if c.startswith("gain_") and not c.startswith("sd_")]
    return NEURAL + gains + NUIS


def descriptive(cohort, ids):
    """Plain spectral features of the first half (the half the parameters were fitted on)."""
    c = load_cross_spectral_collection(ROOT / f"outputs/preproc_v2/{cohort}_restEC/empirical/cross_spectra_fit.npz")
    lab = list(c.channel_names.astype(str))
    fr = np.asarray(c.frequency_hz, float)
    idx = {s: i for i, s in enumerate(c.subject_ids.astype(str))}
    rows = {}
    for s in ids:
        if s not in idx:
            continue
        csd = c.csd[idx[s]]
        ft = features(csd, fr, lab)
        pw = np.maximum(np.real(np.einsum("fii->fi", csd)), 1e-30)
        tot = pw.mean(1)
        rec = {"iaf": float(ft["iaf"][0]), "log_total": float(np.log(tot.sum()))}
        for band, (lo, hi) in {"delta": (2, 3), "theta": (4, 7), "alpha": (8, 12), "beta": (13, 25), "gamma": (26, 40)}.items():
            sel = (fr >= lo) & (fr <= hi)
            rec[f"rel_{band}"] = float(np.log(tot[sel].sum() / tot.sum()))
            rec[f"abs_{band}"] = float(np.log(tot[sel].mean()))
        rows[s] = rec
    return pd.DataFrame(rows).T


def ridge_fit_predict(Xtr, ytr, Xte, alphas=np.logspace(-2, 4, 25), k=5, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(ytr)) % k
    err = []
    for a in alphas:
        e = 0.0
        for f in range(k):
            tr, va = fold != f, fold == f
            A = Ztr[tr]
            w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ (ytr[tr] - ytr[tr].mean()))
            e += np.sum((Ztr[va] @ w + ytr[tr].mean() - ytr[va]) ** 2)
        err.append(e)
    a = alphas[int(np.argmin(err))]
    w = np.linalg.solve(Ztr.T @ Ztr + a * np.eye(Ztr.shape[1]), Ztr.T @ (ytr - ym))
    return Zte @ w + ym, float(a)


def main():
    T = demographics()
    frames = []
    for cohort, path in FITS.items():
        t = pd.read_csv(ROOT / path).set_index("subject_id")
        t["cohort"] = cohort
        t["group"] = "rTMS" if cohort == "rtms" else t["group"]
        frames.append(t)
    x = add_covariates(pd.concat(frames), T)
    names = params(x)
    x = x[(x.age >= 18)].dropna(subset=["age", "sex"])
    out = {"n_tdbrain_adults": int(len(x)), "age_range": [float(x.age.min()), float(x.age.max())]}
    rows = {}
    for p in names:
        d = x.copy()
        d["y"] = (d[p] - d[p].mean()) / d[p].std()
        d["age10"] = d.age / 10.0
        f = smf.ols("y ~ age10 + sex + C(group) + early", data=d).fit()
        rows[p] = {"tdbrain_sd_per_decade": float(f.params["age10"]), "tdbrain_p": float(f.pvalues["age10"])}
    # MODMA replication
    m = pd.read_csv(ROOT / MODMA).set_index("subject_id")
    from mdd_tvb.new_datasets import modma_subjects
    meta = modma_subjects().set_index("subject_id")
    m = m.join(meta[["age", "sex"]], how="left").dropna(subset=["age"])
    m["sex"] = m["sex"].fillna(m["sex"].mean())
    out["n_modma"] = int(len(m))
    for p in names:
        if p not in m.columns:
            continue
        d = m.copy()
        d["y"] = (d[p] - d[p].mean()) / d[p].std()
        d["age10"] = d.age / 10.0
        f = smf.ols("y ~ age10 + sex + C(group)", data=d).fit()
        rows[p].update({"modma_sd_per_decade": float(f.params["age10"]), "modma_p": float(f.pvalues["age10"])})
    tab = pd.DataFrame(rows).T
    from statsmodels.stats.multitest import multipletests
    tab["tdbrain_p_holm"] = multipletests(tab.tdbrain_p, method="holm")[1]
    both = tab.dropna(subset=["modma_sd_per_decade"])
    out["sign_agreement_tdbrain_modma"] = float(np.mean(np.sign(both.tdbrain_sd_per_decade) == np.sign(both.modma_sd_per_decade)))
    out["r_slopes_tdbrain_modma"] = float(np.corrcoef(both.tdbrain_sd_per_decade, both.modma_sd_per_decade)[0, 1])
    sig = both[both.tdbrain_p_holm < 0.05]
    out["holm_significant_in_tdbrain"] = sig.index.tolist()
    out["of_those_same_sign_in_modma"] = int(np.sum(np.sign(sig.tdbrain_sd_per_decade) == np.sign(sig.modma_sd_per_decade)))
    out["of_those_p05_same_sign_in_modma"] = int(np.sum((np.sign(sig.tdbrain_sd_per_decade) == np.sign(sig.modma_sd_per_decade))
                                                        & (sig.modma_p < 0.05)))
    # prediction: train on dev + controls + smc, test on rTMS
    desc = pd.concat([descriptive(c, x.index[x.cohort == c]) for c in FITS])
    x = x.join(desc, how="inner")
    dcols = list(desc.columns)
    train, test = x[x.cohort != "rtms"], x[x.cohort == "rtms"]
    pred = {}
    for label, cols in (("parameters", names), ("descriptive", dcols), ("both", names + dcols)):
        yhat, a = ridge_fit_predict(train[cols].to_numpy(float), train.age.to_numpy(float), test[cols].to_numpy(float))
        y = test.age.to_numpy(float)
        pred[label] = {"r": float(np.corrcoef(yhat, y)[0, 1]), "mae_years": float(np.mean(np.abs(yhat - y))),
                       "r2": float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)), "ridge": a,
                       "n_train": int(len(train)), "n_test": int(len(test))}
    out["age_prediction_on_rtms"] = pred
    out["slopes"] = tab.round(4).to_dict(orient="index")
    dest = ROOT / "outputs/m54_eoec/age_control.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(tab.sort_values("tdbrain_p").round(3).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "slopes"}, indent=1))


if __name__ == "__main__":
    main()
