"""Do batch-free datasets show TDBRAIN's Healthy-MDD EEG differences?

For each new dataset: the depression effect (case - reference, age/sex adjusted
when available) on the interpretable features of scripts/m54/m54_evaluate.py,
computed from both recording halves, restricted to the channels the dataset has.
Compared with (a) the TDBRAIN development MDD - Healthy effect and (b) the TDBRAIN
acquisition-batch effect estimated without depressed subjects (§7.1 of the report),
on the same channels.  Bootstrap CIs resample subjects of the new dataset.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/m54"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from m54_evaluate import features  # noqa: E402

TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
BLOCKS = ["topo_alpha", "topo_beta", "topo_theta", "zerolag_alpha", "zerolag_beta", "lagged_alpha", "spectrum"]
DATASETS = {"modma": ("HC", "MDD"), "mumtaz": ("HC", "MDD"), "ds003478": ("LowBDI", "HighBDI")}


def feats_of(emp_dir, ids, present):
    fit = load_cross_spectral_collection(ROOT / emp_dir / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(ROOT / emp_dir / "cross_spectra_validation.npz")
    idx = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    labels = list(fit.channel_names.astype(str))
    f = np.asarray(fit.frequency_hz, float)
    keep = [labels.index(c) for c in present]
    out = {}
    for s in ids:
        c = 0.5 * (fit.csd[idx[s]] + val.csd[idx[s]])
        out[s] = features(c[:, keep][:, :, keep], f, present)
    return out


def resid(A, cov):
    if cov is None:
        return A
    X = np.column_stack([np.ones(len(cov)), cov])
    b, *_ = np.linalg.lstsq(X, A, rcond=None)
    return A - X[:, 1:] @ b[1:]


def effect(F, ids, groups, ref, case, cov, k):
    A = resid(np.stack([F[s][k] for s in ids]), cov)
    g = np.asarray(groups)
    return A[g == case].mean(0) - A[g == ref].mean(0)


def main():
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    results = {}
    for ds, (ref, case) in DATASETS.items():
        emp = ROOT / f"outputs/preproc_v2/{ds}_restEC/empirical"
        if not (emp / "cross_spectra_fit.npz").is_file():
            continue
        qc = pd.read_csv(emp / "qc.csv").set_index("subject_id")
        ids = [s for s in Path(ROOT / f"configs/m54_lists/{ds}_restEC_v2.txt").read_text().split()
               if qc.loc[s, "group"] in (ref, case)]
        missing = qc.loc[ids[0], "missing_channels"] if "missing_channels" in qc.columns else np.nan
        missing = set(missing.split(";")) if isinstance(missing, str) else set()
        labels = list(load_cross_spectral_collection(emp / "cross_spectra_fit.npz").channel_names.astype(str))
        present = [c for c in labels if c not in missing]
        F = feats_of(f"outputs/preproc_v2/{ds}_restEC/empirical", ids, present)
        g = qc.loc[ids, "group"].to_numpy()
        cov = None
        if {"age", "sex"} <= set(qc.columns) and qc.loc[ids, ["age", "sex"]].notna().all().all():
            cov = qc.loc[ids, ["age", "sex"]].to_numpy(float)
        # TDBRAIN reference effects on the same channels
        dev_ids = Path(ROOT / "configs/m54_lists/dev_restEC_v2.txt").read_text().split()
        Ft = feats_of("outputs/preproc_v2/dev_restEC/empirical", dev_ids, present)
        dq = pd.read_csv(ROOT / "outputs/preproc_v2/dev_restEC/empirical/qc.csv").set_index("subject_id")
        tg = dq.loc[dev_ids, "group"].to_numpy()
        tcov = np.column_stack([T.loc[dev_ids, "age"].astype(float), T.loc[dev_ids, "gender"].astype(float)])
        # batch effect without depressed subjects: late non-depressed patients vs early Healthy + SMC
        others = {}
        for coh in ("controls", "smc"):
            q = pd.read_csv(ROOT / f"outputs/preproc_v2/{coh}_restEC/empirical/qc.csv").set_index("subject_id")
            oid = Path(ROOT / f"configs/m54_lists/{coh}_restEC_v2.txt").read_text().split()
            others.update({s: q.loc[s, "group"] for s in oid})
            Ft.update(feats_of(f"outputs/preproc_v2/{coh}_restEC/empirical", oid, present))
        nd_ids = [s for s in dev_ids if dq.loc[s, "group"] == "Healthy"] + list(others)
        early = np.array([int(s[4:]) < 88000000 for s in nd_ids])
        ncov = np.column_stack([T.loc[nd_ids, "age"].astype(float), T.loc[nd_ids, "gender"].astype(float)])
        res = {"n": {case: int((g == case).sum()), ref: int((g == ref).sum())}, "channels": len(present),
               "age_sex_adjusted": cov is not None, "blocks": {}}
        rng = np.random.default_rng(0)
        for k in BLOCKS:
            e_new = effect(F, ids, g, ref, case, cov, k)
            e_tdb = effect(Ft, dev_ids, tg, "Healthy", "MDD", tcov, k)
            A = resid(np.stack([Ft[s][k] for s in nd_ids]), ncov)
            e_batch = A[~early].mean(0) - A[early].mean(0)
            boot_t, boot_b = [], []
            for _ in range(500):
                pick = np.concatenate([rng.choice(np.flatnonzero(g == grp), (g == grp).sum()) for grp in (ref, case)])
                eb = effect(F, [ids[i] for i in pick], g[pick], ref, case, None if cov is None else cov[pick], k)
                boot_t.append(np.corrcoef(eb, e_tdb)[0, 1]); boot_b.append(np.corrcoef(eb, e_batch)[0, 1])
            res["blocks"][k] = {"r_with_tdbrain_mdd_effect": float(np.corrcoef(e_new, e_tdb)[0, 1]),
                                "ci_tdbrain": np.percentile(boot_t, [2.5, 97.5]).round(3).tolist(),
                                "r_with_tdbrain_batch_effect": float(np.corrcoef(e_new, e_batch)[0, 1]),
                                "ci_batch": np.percentile(boot_b, [2.5, 97.5]).round(3).tolist(),
                                "effect_norm_relative_to_tdbrain": float(np.linalg.norm(e_new) / np.linalg.norm(e_tdb))}
        results[ds] = res
        print(ds, json.dumps(res["n"]), "channels", len(present))
        for k, v in res["blocks"].items():
            print(f"  {k:14s} r(TDBRAIN MDD) {v['r_with_tdbrain_mdd_effect']:+.2f} {v['ci_tdbrain']} | "
                  f"r(TDBRAIN batch) {v['r_with_tdbrain_batch_effect']:+.2f} {v['ci_batch']} | size {v['effect_norm_relative_to_tdbrain']:.2f}")
    (ROOT / "outputs/preproc_v2/cross_dataset_effects.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
