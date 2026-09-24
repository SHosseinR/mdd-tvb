"""Synthetic parameter recovery for the final linear-regime parameterisation.

``generate``: take continuously refined parameter sets of N real subjects as
ground truth (they are not bank states), compute the exact model CSD, and draw
independent complex-Wishart "first" and "second" halves (Welch-like
estimation noise).  The empirical collections are copied with those
subjects' spectra replaced, so fold transformers and pooled nulls stay valid.

``summarise``: correlate recovered with true parameters.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/linear"))

GLOBALS = ("global_coupling", "speed_mm_per_ms", "mu", "a_scale", "b_scale", "fast_ratio",
           "fast_fraction", "noise_tau_ms", "vis_time_contrast", "dorsattn_time_contrast")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def generate(args) -> None:
    import jax.numpy as jnp
    from mdd_tvb.spectral_config import load_spectral_m5_config
    from mdd_tvb.spectral_features import (CrossSpectralCollection, load_cross_spectral_collection,
                                           save_cross_spectral_collection)
    from mdd_tvb.linear_spectral import CandidateLinearizer
    from mdd_tvb import linear_jax as LJ
    import fit_subjects_nested as FSN

    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    gain = lin.gain
    lead = LJ.average_reference(gain)
    lead_np = np.asarray(gain) - np.asarray(gain).mean(0, keepdims=True)
    cov = lead_np @ lead_np.T
    FSN.SOURCE_COV = jnp.asarray(cov / np.mean(np.diag(cov)))
    FSN.MAP = jnp.eye(static.n_groups)
    FSN.USE_COMMON = True
    freq = jnp.asarray(np.arange(2.0, 41.0))
    refined = pd.read_csv(ROOT / args.refined).set_index("subject_id")
    mapfits = pd.read_csv(ROOT / args.map_fits).set_index("subject_id")
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(refined.index.to_numpy(), size=min(args.n, len(refined)), replace=False)
    gains = [c for c in refined.columns if c.startswith("log_noise_gain_")]
    emp = ROOT / "outputs/m5_spectral_m51_production/empirical"
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    fit_csd, val_csd = fit.csd.copy(), val.csd.copy()
    truths = []
    for sid in ids:
        r = refined.loc[sid]
        m = mapfits.loc[sid]
        u = LJ.physical_to_unit(np.asarray([r[g] for g in GLOBALS]))
        z = list(r[gains].to_numpy(float)) + [
            logit(r.observation_noise_fraction / 0.95), logit(m.observation_noise_exponent / 2.5),
            logit(r.source_background_fraction / 0.95), logit(m.source_background_exponent / 3.0),
            logit(r.common_drive_share / 0.95)]
        phys = LJ.unit_to_physical(jnp.asarray(u))
        p, speed = LJ.build_state(phys, static)
        C, common, psi = LJ.contributions_with_common_drive(
            p, static, lead, LJ.delays_for_speed(static, speed), jnp.asarray([0.0, -15.0, 60.0]), 3.0)
        csd = np.asarray(FSN.combine(C, jnp.asarray(z), freq, common), dtype=np.complex128)
        for target in (fit_csd, val_csd):
            sample = np.empty_like(csd)
            for f in range(csd.shape[0]):
                c = 0.5 * (csd[f] + csd[f].conj().T)
                chol = np.linalg.cholesky(c + 1e-12 * np.trace(c).real / 26 * np.eye(26))
                x = chol @ (rng.normal(size=(26, args.dof)) + 1j * rng.normal(size=(26, args.dof))) / np.sqrt(2)
                sample[f] = x @ x.conj().T / args.dof
            target[index[sid]] = sample
        truths.append({"subject_id": sid, **{g: float(r[g]) for g in GLOBALS},
                       **{g: float(r[g]) for g in gains}, "common_drive_share": float(r.common_drive_share),
                       "observation_noise_fraction": float(r.observation_noise_fraction),
                       "source_background_fraction": float(r.source_background_fraction)})
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    save_cross_spectral_collection(out / "cross_spectra_fit.npz", CrossSpectralCollection(**{**fit.__dict__, "csd": fit_csd}))
    save_cross_spectral_collection(out / "cross_spectra_validation.npz", CrossSpectralCollection(**{**val.__dict__, "csd": val_csd}))
    pd.DataFrame(truths).to_csv(out / "truth.csv", index=False)
    (out / "subjects.txt").write_text("\n".join(ids))
    print("generated", len(ids), "synthetic subjects with", args.dof, "dof")


def summarise(args) -> None:
    truth = pd.read_csv(ROOT / args.out / "truth.csv").set_index("subject_id")
    rows = {}
    for label, path in (("bank MAP", args.map_run), ("continuous refinement", args.refined_run)):
        rec = pd.read_csv(ROOT / path / "subject_fits.csv").set_index("subject_id")
        common = truth.index.intersection(rec.index)
        corr = {c: float(np.corrcoef(truth.loc[common, c], rec.loc[common, c])[0, 1])
                for c in truth.columns if c in rec.columns}
        rows[label] = corr
    table = pd.DataFrame(rows).round(3)
    table.to_csv(ROOT / args.out / "recovery_correlations.csv")
    print(table.to_string())
    (ROOT / args.out / "recovery_summary.json").write_text(json.dumps(
        {k: {"median_r": float(np.median(list(v.values()))), "min_r": float(np.min(list(v.values()))),
             "n_params_r_ge_0p5": int(sum(x >= 0.5 for x in v.values())), "n_params": len(v)}
         for k, v in rows.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "summarise"])
    ap.add_argument("--refined", default="outputs/linear_regime/kaggle/final/final_tvb_refined/subject_fits.csv")
    ap.add_argument("--map-fits", default="outputs/linear_regime/kaggle/final/final_tvb/subject_fits.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dof", type=int, default=60, help="Wishart degrees of freedom per 1-Hz bin and half")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="outputs/linear_regime/synthetic_recovery")
    ap.add_argument("--map-run", default="outputs/linear_regime/synthetic_recovery/final")
    ap.add_argument("--refined-run", default="outputs/linear_regime/synthetic_recovery/final_refined")
    args = ap.parse_args()
    generate(args) if args.mode == "generate" else summarise(args)


if __name__ == "__main__":
    main()
