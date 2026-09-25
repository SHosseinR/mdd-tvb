"""Pre-specified rTMS response prediction (docs/RTMS_PREDICTION_PLAN.md).

Feature blocks: C (clinical), E (standard EEG markers), M3 (frozen M5.3
parameters), M4 (M5.4 parameters).  Nested repeated stratified CV with an
L2 logistic regression; label-permutation and block-permutation tests.  H3 tests
the pre-specified group difference in the frozen M5.3 parameters.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402

TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
POST = ("P3", "Pz", "P4", "O1", "Oz", "O2")


def clinical():
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    r = T[(T["Dataset"] == "MDD-rTMS") & (T["sessID"] == 1)].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    out = pd.DataFrame({"age": r["age"], "sex": r["gender"], "bdi_pre": r["BDI_pre"],
                        "protocol": r["rTMS PROTOCOL"], "responder": r["Responder"], "remitter": r["Remitter"],
                        "bdi_change": (r["BDI_post"] - r["BDI_pre"]) / r["BDI_pre"]})
    out.index = out.index.astype(str)
    return out


def band(power, freq, lo, hi):
    sel = (freq >= lo) & (freq <= hi)
    return power[sel].mean(0)


def eeg_markers(ec_dir, eo_dir):
    ec = load_cross_spectral_collection(ROOT / ec_dir / "cross_spectra_fit.npz")
    labels = list(ec.channel_names.astype(str))
    ch = {c: i for i, c in enumerate(labels)}
    f = np.asarray(ec.frequency_hz, float)
    eo_alpha = {}
    if eo_dir and (ROOT / eo_dir / "cross_spectra_fit.npz").is_file():
        eo = load_cross_spectral_collection(ROOT / eo_dir / "cross_spectra_fit.npz")
        for s, c in zip(eo.subject_ids.astype(str), eo.csd):
            p = np.real(np.einsum("fii->fi", c))
            eo_alpha[s] = band(p, f, 8, 12)[[ch[x] for x in POST]].mean()
    rows = {}
    for s, c in zip(ec.subject_ids.astype(str), ec.csd):
        p = np.real(np.einsum("fii->fi", c))
        total = p.sum(0)
        a, th, be = band(p, f, 8, 12), band(p, f, 4, 7), band(p, f, 13, 25)
        post = [ch[x] for x in POST]
        pp = p[:, post].mean(1)
        sel = np.flatnonzero((f >= 7) & (f <= 13))
        j = sel[np.argmax(pp[sel])]
        y0, y1, y2 = np.log(pp[j - 1:j + 2])
        den = y0 - 2 * y1 + y2
        iaf = f[j] + (np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5) if den < 0 else 0.0)
        bg = (f <= 7) | (f >= 30)
        slope = np.polyfit(np.log(f[bg]), np.log(p[bg, ch["Cz"]]), 1)[0]
        rows[s] = {"faa": np.log(a[ch["F4"]]) - np.log(a[ch["F3"]]),
                   "frontal_theta_rel": np.log(th[ch["Fz"]] * 4 / total[ch["Fz"]]),
                   "iaf": iaf,
                   "posterior_alpha_rel": np.log(np.mean(a[post] * 5 / total[post])),
                   "frontal_beta_rel": np.log(np.mean(be[[ch["F3"], ch["Fz"], ch["F4"]]] * 13 /
                                                      total[[ch["F3"], ch["Fz"], ch["F4"]]])),
                   "tbr_fz": np.log(th[ch["Fz"]] / be[ch["Fz"]]),
                   "alpha_reactivity": (np.log(a[post].mean() / eo_alpha[s]) if s in eo_alpha else np.nan),
                   "aperiodic_exponent_cz": -slope}
    return pd.DataFrame(rows).T


M3_PARAMS = ["global_coupling", "mu", "a_scale", "b_scale", "vis_time_contrast", "dorsattn_time_contrast",
             "common_drive_share", "observation_noise_fraction", "source_background_fraction"]


def m3_params(path):
    t = pd.read_csv(ROOT / path).set_index("subject_id")
    gains = [c for c in t.columns if c.startswith("log_noise_gain_") and "Limbic" not in c and "Cont" not in c]
    return t[M3_PARAMS + gains]


def m4_params(path):
    t = pd.read_csv(ROOT / path).set_index("subject_id")
    cols = ["global_coupling", "mu", "a_scale", "b_scale", "vis_time_contrast", "dorsattn_time_contrast",
            "obs_fraction", "obs_exponent", "src_fraction", "src_exponent", "common_share"]
    cols += [c for c in t.columns if c.startswith("gain_") and "Limbic" not in c]
    return t[cols]


def _model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegressionCV(Cs=np.logspace(-3, 2, 10), cv=5, scoring="roc_auc", max_iter=5000))


def cv_scores(X, y, reps, seed=0):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    aucs, baccs = [], []
    for r in range(reps):
        prob = np.zeros(len(y))
        pred = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(X, y):
            m = _model().fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
            pred[te] = prob[te] > y[tr].mean()
        aucs.append(roc_auc_score(y, prob))
        baccs.append(balanced_accuracy_score(y, pred))
    return float(np.mean(aucs)), float(np.mean(baccs))


def permutation_p(X, y, observed, reps, n_perm, block=None, seed=1):
    """Label permutation (block=None) or permutation of the columns in ``block`` only."""
    rng = np.random.default_rng(seed)

    def one(k):
        r = np.random.default_rng(seed + 1000 + k)
        if block is None:
            return cv_scores(X, r.permutation(y), reps, seed=seed + k)[0]
        Xp = X.copy()
        Xp[:, block] = X[r.permutation(len(X))][:, block]
        return cv_scores(Xp, y, reps, seed=seed + k)[0]

    null = np.asarray(Parallel(n_jobs=-1)(delayed(one)(k) for k in range(n_perm)))
    return float((1 + np.sum(null >= observed)) / (1 + n_perm)), float(null.mean())


def continuous(X, y, reps=20, seed=0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    r2, rho = [], []
    for r in range(reps):
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(X):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 20)))
            pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
        r2.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        rho.append(spearmanr(y, pred).correlation)
    return float(np.mean(r2)), float(np.mean(rho))


def h3(dev_fits, ext_fits, param="log_noise_gain_DorsAttn_LH"):
    import statsmodels.formula.api as smf
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    d = pd.read_csv(ROOT / dev_fits)
    d = d[d.group == "Healthy"][["subject_id", param]].assign(mdd=0)
    e = pd.read_csv(ROOT / ext_fits)[["subject_id", param]].assign(mdd=1)
    x = pd.concat([d, e]).set_index("subject_id")
    x["age"] = T.loc[x.index, "age"].astype(float).to_numpy()
    x["sex"] = T.loc[x.index, "gender"].astype(float).to_numpy()
    x["y"] = (x[param] - x[param].mean()) / x[param].std()
    fit = smf.ols("y ~ mdd + age + sex", data=x.dropna()).fit()
    b, p2 = float(fit.params["mdd"]), float(fit.pvalues["mdd"])
    return {"parameter": param, "effect_sd": b, "ci95": fit.conf_int().loc["mdd"].tolist(),
            "p_one_sided_lower": p2 / 2 if b < 0 else 1 - p2 / 2, "n_healthy": int((x.mdd == 0).sum()),
            "n_rtms": int((x.mdd == 1).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec", default="outputs/preproc_v2/rtms_restEC/empirical")
    ap.add_argument("--eo", default="outputs/preproc_v2/rtms_restEO/empirical")
    ap.add_argument("--m3", default="outputs/external/m53/ext_tvb_refined/subject_fits.csv")
    ap.add_argument("--m3-dev", default="outputs/linear_regime/kaggle/final/final_tvb_refined/subject_fits.csv")
    ap.add_argument("--m4", default="outputs/m54/ext_tvb/subject_fits.csv")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--perm-reps", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--out", default="outputs/rtms_prediction")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    C = clinical()
    E = eeg_markers(args.ec, args.eo)
    blocks = {"C": pd.get_dummies(C[["age", "sex", "bdi_pre", "protocol"]].astype({"protocol": "category"}),
                                  columns=["protocol"], dummy_na=True).astype(float),
              "E": E.astype(float)}
    if (ROOT / args.m3).is_file():
        blocks["M3"] = m3_params(args.m3)
    if (ROOT / args.m4).is_file():
        blocks["M4"] = m4_params(args.m4)
    ids = sorted(set(C.index[C.responder.notna()]).intersection(*[set(b.index) for b in blocks.values()]))
    y = C.loc[ids, "responder"].astype(int).to_numpy()
    print(f"subjects {len(ids)}, responders {y.sum()}", flush=True)
    sets = [("C",), ("E",), ("C", "E")]
    if "M3" in blocks:
        sets += [("M3",)]
    if "M4" in blocks:
        sets += [("M4",), ("C", "E", "M4")]
    results = {}
    for s in sets:
        X = np.column_stack([blocks[b].loc[ids].to_numpy(float) for b in s])
        auc, bacc = cv_scores(X, y, args.reps)
        auc_p, _ = cv_scores(X, y, args.perm_reps)
        p, null_mean = permutation_p(X, y, auc_p, args.perm_reps, args.n_perm)
        res = {"auc": auc, "balanced_accuracy": bacc, "perm_p": p, "perm_null_mean_auc": null_mean,
               "n_features": X.shape[1]}
        if s == ("C", "E", "M4"):
            base = np.column_stack([blocks[b].loc[ids].to_numpy(float) for b in ("C", "E")])
            nb = base.shape[1]
            p_inc, _ = permutation_p(X, y, auc_p, args.perm_reps, args.n_perm, block=np.arange(nb, X.shape[1]))
            res["incremental_perm_p_vs_CE"] = p_inc
        yc = C.loc[ids, "bdi_change"].to_numpy(float)
        ok = np.isfinite(yc)
        res["bdi_change_r2"], res["bdi_change_spearman"] = continuous(X[ok], yc[ok])
        results["+".join(s)] = res
        print("+".join(s), json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
    summary = {"subjects": len(ids), "responders": int(y.sum()), "results": results}
    if "M3" in blocks:
        summary["H3"] = h3(args.m3_dev, args.m3)
        print("H3", summary["H3"])
    (out / "prediction_summary.json").write_text(json.dumps(summary, indent=1))
    for name, b in blocks.items():
        b.loc[[i for i in ids if i in b.index]].to_csv(out / f"features_{name}.csv")


if __name__ == "__main__":
    main()
