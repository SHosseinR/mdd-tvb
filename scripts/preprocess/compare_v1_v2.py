"""Compare the original (v1, EEGLAB) and artefact-aware (v2) spectra on the same subjects.

Reports split-half reliability, muscle and ocular signatures, and the
replicability of the Healthy - MDD group effects between the two halves.
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

V1 = ROOT / "outputs/m5_spectral_m51_production/empirical"
V2 = ROOT / "outputs/preproc_v2/dev_restEC/empirical"
LATERAL = ("Fp1", "Fp2", "F7", "F8", "T7", "T8")
CENTRAL = ("C3", "Cz", "C4", "CP3", "CPz", "CP4", "P3", "Pz", "P4")
TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")


def load(d, ids):
    f = load_cross_spectral_collection(d / "cross_spectra_fit.npz")
    v = load_cross_spectral_collection(d / "cross_spectra_validation.npz")
    idx = {s: i for i, s in enumerate(f.subject_ids.astype(str))}
    return f.csd[[idx[s] for s in ids]], v.csd[[idx[s] for s in ids]], np.asarray(f.frequency_hz, float), list(f.channel_names.astype(str))


def slope(p, f, lo=20, hi=40):
    sel = (f >= lo) & (f <= hi)
    return np.polyfit(np.log(f[sel]), np.log(p[sel]), 1)[0]


def residualise(values, covars):
    X = np.column_stack([np.ones(len(covars)), covars])
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    return values - X[:, 1:] @ beta[1:]


def main():
    ids = Path(ROOT / "configs/m54_lists/dev_restEC_v2.txt").read_text().split()
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    groups = load_cross_spectral_collection(V2 / "cross_spectra_fit.npz")
    gmap = dict(zip(groups.subject_ids.astype(str), groups.groups.astype(str)))
    g = np.asarray([gmap[s] for s in ids])
    cov = np.column_stack([T.loc[ids, "age"].astype(float), T.loc[ids, "gender"].astype(float)])
    report = {"subjects": len(ids), "healthy": int((g == "Healthy").sum()), "mdd": int((g == "MDD").sum())}
    for name, d in (("v1", V1), ("v2", V2)):
        fit, val, f, labels = load(d, ids)
        pw_fit = np.real(np.einsum("nfii->nfi", fit))
        pw_val = np.real(np.einsum("nfii->nfi", val))
        lat = [labels.index(c) for c in LATERAL]
        cen = [labels.index(c) for c in CENTRAL]
        s_lat = np.asarray([[slope(pw_fit[n, :, c], f) for c in lat] for n in range(len(ids))])
        s_cen = np.asarray([[slope(pw_fit[n, :, c], f) for c in cen] for n in range(len(ids))])
        fp = [labels.index("Fp1"), labels.index("Fp2")]
        slow = np.log(pw_fit[:, (f >= 2) & (f <= 4)][:, :, fp].mean((1, 2)) / pw_fit[:, (f >= 2) & (f <= 4)][:, :, cen].mean((1, 2)))
        fF = [features(c, f, labels) for c in fit]
        fV = [features(c, f, labels) for c in val]
        rel = {}
        rep = {}
        for k in fF[0]:
            A = np.stack([x[k] for x in fF]); B = np.stack([x[k] for x in fV])
            if A.shape[1] == 1:
                rel[k] = float(np.corrcoef(A[:, 0], B[:, 0])[0, 1])
            else:  # median over coordinates of the across-subject test-retest correlation
                rel[k] = float(np.nanmedian([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(A.shape[1])]))
            Ar = residualise(A, cov); Br = residualise(B, cov)
            eA = Ar[g == "MDD"].mean(0) - Ar[g == "Healthy"].mean(0)
            eB = Br[g == "MDD"].mean(0) - Br[g == "Healthy"].mean(0)
            rep[k] = float(np.corrcoef(eA, eB)[0, 1]) if eA.size > 1 else float(eA[0] - eB[0])
        # subject-sampling replicability: effects in two disjoint random halves of the subjects
        rng = np.random.default_rng(0)
        split_rep = {}
        for k in fF[0]:
            A = residualise(np.stack([x[k] for x in fF]), cov)
            if A.shape[1] == 1:
                continue
            rs = []
            for _ in range(200):
                perm = rng.permutation(len(ids))
                h1, h2 = perm[: len(ids) // 2], perm[len(ids) // 2:]
                e1 = A[h1][g[h1] == "MDD"].mean(0) - A[h1][g[h1] == "Healthy"].mean(0)
                e2 = A[h2][g[h2] == "MDD"].mean(0) - A[h2][g[h2] == "Healthy"].mean(0)
                rs.append(np.corrcoef(e1, e2)[0, 1])
            split_rep[k] = float(np.median(rs))
        report.setdefault("subject_split_replicability", {})[name] = split_rep
        report[name] = {
            "median_slope_20_40_lateral": float(np.median(s_lat)), "median_slope_20_40_central": float(np.median(s_cen)),
            "fraction_lateral_channels_slope_gt_minus1": float(np.mean(s_lat > -1.0)),
            "median_fp_vs_central_2_4hz_log_ratio": float(np.median(slow)),
            "median_epochs_per_half": float(np.median(load_cross_spectral_collection(d / "cross_spectra_fit.npz").epoch_counts)),
            "split_half_reliability": rel, "group_effect_half_to_half_replicability": rep}
    (ROOT / "outputs/preproc_v2/compare_v1_v2.json").write_text(json.dumps(report, indent=1))
    rows = {k: {"v1": report["v1"]["split_half_reliability"][k], "v2": report["v2"]["split_half_reliability"][k],
                "effect_rep_v1": report["v1"]["group_effect_half_to_half_replicability"][k],
                "effect_rep_v2": report["v2"]["group_effect_half_to_half_replicability"][k]}
            for k in report["v1"]["split_half_reliability"]}
    for k in rows:
        for name in ("v1", "v2"):
            rows[k][f"subject_split_rep_{name}"] = report["subject_split_replicability"][name].get(k, np.nan)
    print(pd.DataFrame(rows).T.round(3).to_string())
    for name in ("v1", "v2"):
        print(name, {k: round(v, 3) for k, v in report[name].items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
