"""Second-level analysis of the joint eyes-closed + eyes-open fits (m54_joint, delta = vis).

The eyes-open changes (delta_*) are within-person contrasts, so acquisition set-up
should largely cancel from them.  Tests:
1. Reliability: split-half ICC and posterior calibration (m54_group, --prefix delta_).
2. TDBRAIN, all cohorts (development Healthy/MDD, rTMS, late clinical controls, SMC):
   delta ~ early batch + clinical + depression + age + sex  (same model as batch_confound.py),
   for the eyes-open changes AND for the shared eyes-closed parameters of the same fits.
3. Healthy vs MDD (development) and the batch-free datasets (ds003478 high vs low BDI,
   Mumtaz MDD vs HC): calibrated random-effects group comparison of the changes.
4. Age as a positive control (alpha reactivity falls with age at the data level).
"""
from __future__ import annotations

import json, subprocess, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/m54"))
from batch_confound import add_covariates, demographics  # noqa: E402

RES = {"a": "outputs/m54e_joint3a/results", "b": "outputs/m54e_joint3b/results"}
TDB = {"dev": ("a", "joint_dev_eoec_v2_bem_vis"), "rtms": ("a", "joint_rtms_bem_vis"),
       "controls": ("a", "joint_controls_bem_vis"), "smc": ("a", "joint_smc_bem_vis")}
NEWDS = {"ds003478": ("b", "joint_ds003478_bem_vis", ("LowBDI", "HighBDI")),
         "mumtaz": ("b", "joint_mumtaz_bem_vis", ("HC", "MDD"))}
SHARED = ["global_coupling", "mu", "a_scale", "b_scale", "common_share"]


def fits(which, name, swap=False):
    p = ROOT / RES[which] / (name + ("_swap" if swap else "")) / "subject_fits.csv"
    return pd.read_csv(p).set_index("subject_id") if p.is_file() else None


def run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode:
        print(r.stderr[-1500:])
    return r.returncode == 0


def main():
    out = ROOT / "outputs/m54_eoec"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    # 1 + 3. reliability and group comparisons with the calibrated random-effects script
    for label, (w, name, groups) in {"dev": (*TDB["dev"], ("Healthy", "MDD")), **NEWDS}.items():
        first = ROOT / RES[w] / name / "subject_fits.csv"
        if not first.is_file():
            print("missing", first)
            continue
        second = ROOT / RES[w] / (name + "_swap") / "subject_fits.csv"
        args = ["scripts/m54/m54_group.py", "--first", str(first.relative_to(ROOT)),
                "--second", str(second.relative_to(ROOT)) if second.is_file() else "none",
                "--prefix", "delta_", "--groups", *groups, "--out", f"outputs/m54_eoec/group_{label}"]
        if label != "dev":
            args += ["--qc", f"outputs/preproc_v2/{label}_restEC/empirical/qc.csv"]
        if run(args):
            g = pd.read_csv(out / f"group_{label}" / "group_effects.csv")
            summary[f"group_{label}"] = g[["parameter", "effect_sd_units", "p", "p_holm", "icc", "age_p"]].to_dict("records")
    # 2 + 4. batch / clinical / depression / age decomposition in TDBRAIN
    frames = []
    for cohort, (w, name) in TDB.items():
        t = fits(w, name)
        if t is None:
            print("missing", cohort)
            continue
        t = t.copy()
        t["group"] = "rTMS" if cohort == "rtms" else t["group"]
        t["cohort"] = cohort
        frames.append(t)
    if frames:
        x = add_covariates(pd.concat(frames), demographics())
        x = x.dropna(subset=["age", "sex"])
        cols = [c for c in x.columns if c.startswith("delta_") and c != "delta_set" and not c.startswith("sd_")]
        rows = {}
        for c in cols + SHARED + [g for g in x.columns if g.startswith("gain_") and not g.startswith("sd_")]:
            d = x.copy()
            ref = d.loc[d.group.isin(["Healthy", "MDD"]), c]
            d["y"] = (d[c] - d.loc[d.group == "Healthy", c].mean()) / ref.std()
            d["age10"] = d.age / 10
            f = smf.ols("y ~ early + clin + dep + age10 + sex", data=d).fit()
            rows[c] = {k: {"coef": float(f.params[k]), "p": float(f.pvalues[k])} for k in ("early", "clin", "dep", "age10")}
        summary["tdbrain_decomposition"] = rows
        summary["tdbrain_n"] = x.cohort.value_counts().to_dict()
        tab = pd.DataFrame({k: {f"{t}_{s}": v[t][s] for t in ("early", "dep", "age10") for s in ("coef", "p")}
                            for k, v in rows.items()}).T
        print(tab.round(3).to_string())
        # mean eyes-open change per cohort (sanity: direction of alpha blocking)
        summary["mean_changes"] = x.groupby("cohort")[cols].mean().round(3).to_dict()
    (out / "eoec_group.json").write_text(json.dumps(summary, indent=1, default=float))
    for k, v in summary.items():
        if k.startswith("group_"):
            print(k, pd.DataFrame(v).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
