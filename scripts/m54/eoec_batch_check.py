"""Does the eyes-closed / eyes-open contrast escape TDBRAIN's acquisition batch?

Premise of the joint EO+EC fit: a within-subject condition contrast cancels what
differs between recording set-ups (amplifier gain, electrodes, skull), whereas
absolute measures carry it.  Data-level test, same logic as batch_confound.py:

* depression effect  = MDD - Healthy (dev subjects; Healthy recorded early, MDD late);
* batch effect       = late - early among non-depressed subjects
                       (late: OCD/insomnia/tinnitus/ADHD adults; early: Healthy + SMC);
* batch-matched depression effect = (MDD + rTMS) - late non-depressed.

For each feature family, absolute (eyes closed) vs contrast (log EC - log EO),
report ||batch|| / ||depression effect||, the correlation of the two effect
patterns, and scalar regressions (early + clinical + depression + age + sex).
Only QC-usable recordings with both conditions are used.
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

COHORTS = ("dev", "rtms", "controls", "smc")
EMG = ("Fp1", "Fp2", "F7", "F8", "T7", "T8")
BANDS = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 25)}
USABLE = ("ok", "qc_flag")


def load(cohort, task):
    emp = ROOT / f"outputs/preproc_v2/{cohort}_{task}/empirical"
    c = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    v = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    qc = pd.read_csv(emp / "qc.csv").set_index("subject_id")
    ok = set(qc.index[qc.status.isin(USABLE) & (qc.note.fillna("") != "unrepairable channel neighbourhood")])
    ids = c.subject_ids.astype(str)
    # both halves together: epoch-weighted mean of the fit and validation CSDs
    w1, w2 = c.epoch_counts[:, None, None, None], v.epoch_counts[:, None, None, None]
    csd = (c.csd * w1 + v.csd * w2) / (w1 + w2)
    return {s: (g, x) for s, g, x in zip(ids, c.groups.astype(str), csd) if s in ok}, np.asarray(c.frequency_hz, float), \
        list(c.channel_names.astype(str))


def main():
    dev_ids = set(pd.read_csv(ROOT / "configs/m52_nested_splits.csv").subject_id.astype(str))
    rows, fam = [], {}
    for cohort in COHORTS:
        ec, freq, lab = load(cohort, "restEC")
        eo, _, _ = load(cohort, "restEO")
        keep = [lab.index(c) for c in lab if c not in EMG]
        for s in sorted(set(ec) & set(eo)):
            if cohort == "dev" and s not in dev_ids:
                continue
            g = "rTMS" if cohort == "rtms" else ec[s][0]
            pe = np.maximum(np.real(np.einsum("fii->fi", ec[s][1])), 1e-30)[:, keep]
            po = np.maximum(np.real(np.einsum("fii->fi", eo[s][1])), 1e-30)[:, keep]
            le, lo = np.log(pe), np.log(po)
            rec = {"subject_id": s, "group": g}
            vec = {}
            for band, (a, b) in BANDS.items():
                sel = (freq >= a) & (freq <= b)
                be, bo = np.log(pe[sel].mean(0)), np.log(po[sel].mean(0))
                rec[f"abs_{band}"] = float(be.mean())          # absolute log power, eyes closed
                rec[f"react_{band}"] = float((be - bo).mean())  # log EC/EO ratio
                vec[f"abs_topo_{band}"] = be                   # absolute per-channel log power
                vec[f"react_topo_{band}"] = be - bo
            vec["abs_spectrum"] = le.ravel()
            vec["react_spectrum"] = (le - lo).ravel()
            rows.append(rec)
            for k, v in vec.items():
                fam.setdefault(k, []).append(v)
    T = demographics()
    x = add_covariates(pd.DataFrame(rows).set_index("subject_id"), T)
    print("subjects by group", x.group.value_counts().to_dict())
    out = {"n": int(len(x)), "groups": x.group.value_counts().to_dict()}
    cov = x[["age", "sex"]].fillna(x[["age", "sex"]].mean()).to_numpy()
    X = np.column_stack([np.ones(len(cov)), cov])
    g = x.group.to_numpy()
    e = x.early.to_numpy().astype(bool)
    nondep = ~np.isin(g, ["MDD", "rTMS"])
    table = {}
    for k, vals in fam.items():
        A = np.stack(vals)
        b, *_ = np.linalg.lstsq(X, A, rcond=None)
        A = A - X[:, 1:] @ b[1:]
        sd = A[np.isin(g, ["Healthy", "MDD"])].std(0)
        A = A / np.where(sd > 0, sd, 1.0)  # effects in between-subject SD units per element
        dep = A[g == "MDD"].mean(0) - A[g == "Healthy"].mean(0)
        batch = A[nondep & ~e].mean(0) - A[nondep & e].mean(0)
        matched = A[np.isin(g, ["MDD", "rTMS"])].mean(0) - A[nondep & ~e & (g != "Healthy")].mean(0)
        table[k] = {"rms_depression_sd": float(np.sqrt(np.mean(dep ** 2))),
                    "rms_batch_sd": float(np.sqrt(np.mean(batch ** 2))),
                    "rms_batch_matched_depression_sd": float(np.sqrt(np.mean(matched ** 2))),
                    "r_depression_vs_batch": float(np.corrcoef(dep, batch)[0, 1]) if len(dep) > 2 else None,
                    "r_depression_vs_matched": float(np.corrcoef(dep, matched)[0, 1]) if len(dep) > 2 else None}
    out["patterns"] = table
    scal = {}
    for col in [c for c in x.columns if c.startswith(("abs_", "react_"))]:
        d = x.dropna(subset=["age", "sex"]).copy()
        d["y"] = (d[col] - d.loc[d.group == "Healthy", col].mean()) / d.loc[d.group.isin(["Healthy", "MDD"]), col].std()
        f = smf.ols("y ~ early + clin + dep + age + sex", data=d).fit()
        scal[col] = {k: {"coef": float(f.params[k]), "p": float(f.pvalues[k])} for k in ("early", "clin", "dep", "age")}
    out["scalar"] = scal
    dest = ROOT / "outputs/m54_eoec/batch_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(table).T.round(3).to_string())
    print(pd.DataFrame({k: {f"{t}_{s}": v[t][s] for t in ("early", "dep", "age") for s in ("coef", "p")}
                        for k, v in scal.items()}).T.round(3).to_string())


if __name__ == "__main__":
    main()
