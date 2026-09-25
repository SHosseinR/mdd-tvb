"""Interpretable unseen-half evaluation of any run that saved raw CSD predictions.

Blocks (each: per-subject mean squared error of the prediction vs the unseen
half, divided by that of the pooled population null; lower is better):

* ``spectrum``   centred log power, every channel x frequency (muscle-prone
                 channels excluded >= 20 Hz, as in the likelihood);
* ``topo_<band>`` channel-centred band log power (theta 4-7, alpha 8-12, beta 13-19 Hz);
* ``zerolag_<band>`` real coherency of all channel pairs (alpha, theta, beta);
* ``lagged_<band>``  lagged coherency Im/sqrt(1 - Re^2);
* ``iaf``        absolute error of the posterior individual alpha frequency (Hz).

Group-effect preservation: Healthy - MDD differences of each block's features in
the predictions vs the unseen empirical halves (correlation and norm ratio).
The pooled null follows the nested M5.2 folds (training subjects' first halves)
unless ``--null-emp-dir`` gives an external development collection.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.whittle import FIXED_EMG_CHANNELS, EMG_SPLIT_HZ  # noqa: E402

BANDS = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 19)}
POSTERIOR = ("P3", "Pz", "P4", "O1", "Oz", "O2")


def features(csd, freq, labels):
    """Dict of feature vectors for one (F, 26, 26) CSD."""
    power = np.maximum(np.real(np.einsum("fii->fi", csd)), 1e-30)
    logp = np.log(power)
    keep = np.ones_like(logp, bool)
    emg = [labels.index(c) for c in FIXED_EMG_CHANNELS]
    keep[np.ix_(freq >= EMG_SPLIT_HZ, emg)] = False
    spectrum = logp - logp[keep].mean()
    out = {"spectrum": spectrum[keep]}
    iu = np.triu_indices(len(labels), 1)
    for band, (lo, hi) in BANDS.items():
        sel = (freq >= lo) & (freq <= hi)
        bp = np.log(power[sel].mean(0))
        out[f"topo_{band}"] = bp - bp.mean()
        c = csd[sel].mean(0)
        d = np.sqrt(np.outer(np.real(np.diag(c)), np.real(np.diag(c))))
        coh = (c / d)[iu]
        out[f"zerolag_{band}"] = np.real(coh)
        out[f"lagged_{band}"] = np.imag(coh) / np.sqrt(np.maximum(1 - np.real(coh) ** 2, 1e-8))
    post = power[:, [labels.index(c) for c in POSTERIOR]].mean(1)
    sel = np.flatnonzero((freq >= 7) & (freq <= 13))
    j = sel[np.argmax(post[sel])]
    if 0 < j < len(freq) - 1:
        y0, y1, y2 = np.log(post[j - 1:j + 2])
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom < 0 else 0.0
    else:
        shift = 0.0
    out["iaf"] = np.asarray([freq[j] + np.clip(shift, -0.5, 0.5)])
    return out


def paired_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    boots = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(n)]
    return float(np.median(diff)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(run_dir, emp_dir, null_emp_dir=None, subjects=None, label=None):
    emp = ROOT / emp_dir
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    labels = list(fit.channel_names.astype(str))
    freq = np.asarray(fit.frequency_hz, float)
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    with np.load(ROOT / run_dir / "predictions.npz") as payload:
        pred = dict(zip(payload["subject_ids"].astype(str), payload["csd"]))
    ids = [s for s in pred if s in index and (subjects is None or s in subjects)]
    feats = lambda c: features(np.asarray(c, np.complex128), freq, labels)  # noqa: E731
    val_f = {s: feats(val.csd[index[s]]) for s in ids}
    pred_f = {s: feats(pred[s]) for s in ids}
    # null features: mean of training subjects' first-half features (nested) or development mean
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    if null_emp_dir:
        dev = load_cross_spectral_collection(ROOT / null_emp_dir / "cross_spectra_fit.npz")
        dev_ids = set(splits.subject_id.astype(str))
        rows = [feats(dev.csd[i]) for i, s in enumerate(dev.subject_ids.astype(str)) if s in dev_ids]
        null = {k: np.mean([r[k] for r in rows], 0) for k in rows[0]}
        null_of = {s: null for s in ids}
    else:
        fit_f = {s: feats(fit.csd[i]) for s, i in index.items()}
        null_of = {}
        for fold in sorted(splits.outer_fold.unique()):
            fr = splits[splits.outer_fold == fold]
            train = [s for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str) if s in fit_f]
            null = {k: np.mean([fit_f[s][k] for s in train], 0) for k in fit_f[train[0]]}
            for s in fr.loc[fr.outer_role == "validation", "subject_id"].astype(str):
                null_of[s] = null
        ids = [s for s in ids if s in null_of]
    rows = []
    for s in ids:
        row = {"subject_id": s, "group": str(fit.groups[index[s]])}
        for k in val_f[s]:
            e_model = np.mean((pred_f[s][k] - val_f[s][k]) ** 2)
            e_null = np.mean((null_of[s][k] - val_f[s][k]) ** 2)
            row[f"{k}_ratio"] = e_model / e_null if e_null > 0 else np.nan
            if k == "iaf":
                row["iaf_abs_error_hz"] = float(np.sqrt(e_model))
                row["iaf_null_abs_error_hz"] = float(np.sqrt(e_null))
        rows.append(row)
    table = pd.DataFrame(rows)
    groups = table.group.to_numpy()
    effects = {}
    if {"Healthy", "MDD"} <= set(groups):
        for k in val_f[ids[0]]:
            V = np.stack([val_f[s][k] for s in ids]); P = np.stack([pred_f[s][k] for s in ids])
            ev = V[groups == "MDD"].mean(0) - V[groups == "Healthy"].mean(0)
            ep = P[groups == "MDD"].mean(0) - P[groups == "Healthy"].mean(0)
            if ev.size > 1:
                effects[k] = {"r": float(np.corrcoef(ev, ep)[0, 1]), "norm_ratio": float(np.linalg.norm(ep) / np.linalg.norm(ev))}
            else:
                effects[k] = {"empirical": float(ev[0]), "predicted": float(ep[0])}
    summary = {"label": label or str(run_dir), "subjects": len(table),
               "median_ratio": {c.removesuffix("_ratio"): float(table[c].median()) for c in table.columns if c.endswith("_ratio")},
               "iaf_median_abs_error_hz": float(table.iaf_abs_error_hz.median()),
               "iaf_null_median_abs_error_hz": float(table.iaf_null_abs_error_hz.median()),
               "group_effects": effects}
    return table, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--emp-dir", required=True)
    ap.add_argument("--null-emp-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects-file", default=None)
    args = ap.parse_args()
    subjects = set(Path(ROOT / args.subjects_file).read_text().split()) if args.subjects_file else None
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tables, summaries = {}, []
    for spec in args.runs:
        label, run = spec.split("=", 1)
        t, s = evaluate(run, args.emp_dir, args.null_emp_dir, subjects, label)
        t.to_csv(out / f"blocks_{label}.csv", index=False)
        tables[label] = t.set_index("subject_id")
        summaries.append(s)
    common = sorted(set.intersection(*[set(t.index) for t in tables.values()]))
    labels = list(tables)
    paired = {}
    for other in labels[1:]:
        paired[other] = {}
        for c in tables[labels[0]].columns:
            if c.endswith("_ratio"):
                paired[other][c] = paired_ci(tables[other].loc[common, c], tables[labels[0]].loc[common, c])
    (out / "evaluation.json").write_text(json.dumps({"summaries": summaries, "paired_vs_first": paired,
                                                     "common_subjects": len(common)}, indent=1))
    frame = pd.DataFrame({s["label"]: s["median_ratio"] for s in summaries}).round(3)
    print(frame.to_string())
    print(json.dumps({s["label"]: s["group_effects"] for s in summaries}, indent=1)[:4000])


if __name__ == "__main__":
    main()
