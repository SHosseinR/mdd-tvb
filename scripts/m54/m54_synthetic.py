"""Synthetic recovery and posterior-coverage test for M5.4.

generate: the MAP parameters of N real subjects (not bank states) are ground
truth.  Their exact model CSD is sampled as two independent complex-Wishart
halves with each subject's own degrees of freedom, then passed through that
subject's channel-repair and average-reference operator.  The emp directory
keeps the subject's QC (muscle channels) and repair matrices, so the refit uses
identical observation operators.
summarise: truth-vs-estimate correlations and 95 % Laplace-interval coverage.
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jnp, W  # noqa: E402
from mdd_tvb.spectral_features import (CrossSpectralCollection, load_cross_spectral_collection,  # noqa: E402
                                       save_cross_spectral_collection)


def wishart(rng, S, dof):
    n = S.shape[-1]
    L = np.linalg.cholesky(S + 1e-9 * np.trace(S).real / n * np.eye(n))
    X = L @ (rng.normal(size=(n, dof)) + 1j * rng.normal(size=(n, dof))) / np.sqrt(2)
    return X @ X.conj().T / dof


def generate(args):
    fits = pd.read_csv(M.ROOT / args.fits).set_index("subject_id")
    population = json.loads((M.ROOT / args.population).read_text())
    setup = M.make_setup(args.lead, population)
    names = setup.theta_names()
    rng = np.random.default_rng(args.seed)
    ids = sorted(rng.choice(fits.index.to_numpy(), size=min(args.n, len(fits)), replace=False))
    emp = M.ROOT / args.emp_dir
    out = M.ROOT / args.out / "empirical"
    out.mkdir(parents=True, exist_ok=True)
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    repair = {}
    if (emp / "repair_matrices.npz").is_file():
        with np.load(emp / "repair_matrices.npz") as payload:
            repair = dict(zip(payload["subject_ids"].astype(str), payload["repair"].astype(float)))
    R = np.eye(26) - 1.0 / 26
    model = M.jax.jit(lambda th: M.model_csd(setup, th)[0])
    sel = [index[s] for s in ids]
    csd = {"fit": [], "validation": []}
    truth = []
    for s in ids:
        row = fits.loc[s]
        theta = jnp.asarray([row[n] for n in names])
        S = np.asarray(model(theta), np.complex128) * np.exp(row["log_scale"])
        O = R @ repair.get(s, np.eye(26))
        for key, coll in (("fit", fit), ("validation", val)):
            dof = int(round(W.DOF_PER_EPOCH * coll.epoch_counts[index[s]]))
            sample = np.stack([wishart(rng, S[f], dof) for f in range(S.shape[0])])
            csd[key].append(np.einsum("ij,fjk,lk->fil", O, sample, O))
        truth.append({"subject_id": s, **{n: float(row[n]) for n in names},
                      **{c: float(row[c]) for c in row.index if c.startswith("gain_") and not c.startswith("sd_")}})
    for key, coll in (("fit", fit), ("validation", val)):
        save_cross_spectral_collection(out / f"cross_spectra_{key}.npz", CrossSpectralCollection(
            subject_ids=coll.subject_ids[sel], groups=coll.groups[sel], source_files=coll.source_files[sel],
            durations_s=coll.durations_s[sel], epoch_counts=coll.epoch_counts[sel], frequency_hz=coll.frequency_hz,
            channel_names=coll.channel_names, csd=np.stack(csd[key])))
    for extra in ("qc.csv", "repair_matrices.npz"):
        if (emp / extra).is_file():
            shutil.copy2(emp / extra, out / extra)
    pd.DataFrame(truth).to_csv(M.ROOT / args.out / "truth.csv", index=False)
    (M.ROOT / args.out / "subjects.txt").write_text("\n".join(ids) + "\n")
    print("generated", len(ids))


def summarise(args):
    truth = pd.read_csv(M.ROOT / args.out / "truth.csv").set_index("subject_id")
    rec = pd.read_csv(M.ROOT / args.recovered).set_index("subject_id")
    common = truth.index.intersection(rec.index)
    rows = {}
    for c in truth.columns:
        if c not in rec.columns or c.startswith("log_gain_"):
            continue
        t, e = truth.loc[common, c], rec.loc[common, c]
        sd = rec.loc[common, f"sd_{c}"] if f"sd_{c}" in rec.columns else None
        rows[c] = {"r": float(np.corrcoef(t, e)[0, 1]), "bias": float(np.mean(e - t)),
                   "coverage95": float(np.mean(np.abs(e - t) <= 1.96 * sd)) if sd is not None else np.nan,
                   "rmse_over_sd": float(np.sqrt(np.mean(((e - t) / sd) ** 2))) if sd is not None else np.nan}
    table = pd.DataFrame(rows).T
    table.to_csv(M.ROOT / args.out / "recovery.csv")
    summary = {"n": int(len(common)), "median_r": float(table.r.median()), "n_r_ge_0p5": int((table.r >= 0.5).sum()),
               "n_params": int(len(table)), "median_coverage95": float(table.coverage95.median())}
    (M.ROOT / args.out / "recovery_summary.json").write_text(json.dumps(summary, indent=1))
    print(table.round(3).to_string())
    print(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "summarise"])
    ap.add_argument("--fits", default="outputs/m54/dev_tvb/subject_fits.csv")
    ap.add_argument("--population", default="outputs/m54/population_tvb.json")
    ap.add_argument("--emp-dir", default="outputs/preproc_v2/dev_restEC/empirical")
    ap.add_argument("--lead", default="tvb")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/m54/synthetic_tvb")
    ap.add_argument("--recovered", default="outputs/m54/synthetic_tvb_fit/subject_fits.csv")
    args = ap.parse_args()
    generate(args) if args.mode == "generate" else summarise(args)


if __name__ == "__main__":
    main()
