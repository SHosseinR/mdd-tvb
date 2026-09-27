"""How much of resting beta is an alpha harmonic?  (data check before changing the model)

For eyes-closed v2-cleaned recordings of development subjects:
* individual alpha frequency (IAF) from posterior channels;
* whether the 13-30 Hz spectral peak (above a 1/f fit) sits at 2 x IAF (+/- 1 Hz);
* bicoherence b(IAF, IAF) -> 2 IAF (quadratic phase coupling of alpha with its
  harmonic) against a within-subject surrogate: the same statistic with the
  harmonic segment taken from a different epoch (destroys phase coupling,
  keeps power);
separately for posterior (O1/Oz/O2, P3/Pz/P4) and central (C3/Cz/C4) channels.
A linear Gaussian model cannot produce harmonic beta; an independent rhythm
(e.g. rolandic beta) it can.
"""
from __future__ import annotations

import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mdd_tvb.preprocess_v2 import EEG_LABELS, clean_epoch_starts, preprocess_bdf  # noqa: E402

TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
SITES = {"posterior": ("O1", "Oz", "O2", "P3", "Pz", "P4"), "central": ("C3", "Cz", "C4")}


def one(sid: str) -> dict:
    res = preprocess_bdf(TDBRAIN / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-restEC_eeg.bdf")
    fs, n = res.fs, int(4 * res.fs)
    starts = clean_epoch_starts(res.clean, n, n)  # non-overlapping 4-s epochs
    if len(starts) < 10:
        return {"subject_id": sid, "ok": False}
    win = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / fs)
    out = {"subject_id": sid, "ok": True, "epochs": len(starts)}
    rng = np.random.default_rng(0)
    for site, chans in SITES.items():
        idx = [EEG_LABELS.index(c) for c in chans]
        seg = np.stack([res.data_uv[idx, s:s + n] for s in starts])        # (E, C, n)
        seg = seg - seg.mean(-1, keepdims=True)
        X = np.fft.rfft(seg * win, axis=-1)                                  # (E, C, F)
        P = (np.abs(X) ** 2).mean((0, 1))
        sel = (f >= 7) & (f <= 13)
        iaf = f[sel][np.argmax(P[sel])]
        # beta peak above a 1/f fit (fit on 3-7 and 30-40 Hz)
        bg = ((f >= 3) & (f <= 7)) | ((f >= 30) & (f <= 40))
        coef = np.polyfit(np.log(f[bg]), np.log(P[bg]), 1)
        resid = np.log(P) - np.polyval(coef, np.log(np.maximum(f, 1e-3)))
        bsel = (f >= 13) & (f <= 30)
        fb = f[bsel][np.argmax(resid[bsel])]
        k1 = int(np.argmin(np.abs(f - iaf)))
        k2 = int(np.argmin(np.abs(f - 2 * iaf)))

        def bic(Xa, Xb):
            num = np.abs(np.sum(Xa[..., k1] * Xa[..., k1] * np.conj(Xb[..., k2])))
            den = np.sqrt(np.sum(np.abs(Xa[..., k1] * Xa[..., k1]) ** 2) * np.sum(np.abs(Xb[..., k2]) ** 2))
            return num / den

        b = bic(X, X)
        sur = np.mean([bic(X, X[rng.permutation(len(X))]) for _ in range(50)])
        out.update({f"{site}_iaf": float(iaf), f"{site}_beta_peak": float(fb),
                    f"{site}_beta_peak_height": float(resid[bsel].max()),
                    f"{site}_beta_at_2iaf": bool(abs(fb - 2 * iaf) <= 1.0),
                    f"{site}_bicoherence": float(b), f"{site}_bicoherence_surrogate": float(sur)})
    return out


def main():
    ids = Path(ROOT / "configs/m54_lists/dev_restEC_v2.txt").read_text().split()
    ids = list(np.random.default_rng(1).choice(ids, size=min(120, len(ids)), replace=False))
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(one, ids))
    t = pd.DataFrame([r for r in rows if r.get("ok")])
    out = {"subjects": len(t)}
    for site in SITES:
        b, s = t[f"{site}_bicoherence"], t[f"{site}_bicoherence_surrogate"]
        out[site] = {"median_bicoherence": float(b.median()), "median_surrogate": float(s.median()),
                     "fraction_bicoherence_gt_2x_surrogate": float((b > 2 * s).mean()),
                     "fraction_beta_peak_at_2iaf": float(t[f"{site}_beta_at_2iaf"].mean()),
                     "median_beta_peak_hz": float(t[f"{site}_beta_peak"].median()),
                     "median_iaf": float(t[f"{site}_iaf"].median()),
                     "corr_bicoherence_beta_height": float(np.corrcoef(b, t[f"{site}_beta_peak_height"])[0, 1])}
    dest = ROOT / "outputs/preproc_v2/beta_origin_check.json"
    dest.write_text(json.dumps(out, indent=1))
    t.to_csv(dest.with_suffix(".csv"), index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
