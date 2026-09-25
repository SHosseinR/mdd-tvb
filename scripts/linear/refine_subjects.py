"""Continuous per-subject refinement of *all* parameters (GPU recommended).

Starts from each subject's MAP state of a nested bank fit (``subject_fits.csv``)
and runs Adam on the 10 global dynamics parameters, the spatial log-gains, the
observation nuisances and (optionally) the shared-drive share, through the
full differentiable analytic spectrum.  Stability is enforced with penalties
(damped nodes, Jansen-Rit fold margin, small-gain certificate) and the best
*certified* iterate is kept.  Scoring is identical to fit_subjects_nested.py.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/linear"))

import jax, jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_spectral import CandidateLinearizer  # noqa: E402
from mdd_tvb.linear_fit import load_transformer_arrays, feature_blocks, weighted_features  # noqa: E402
from mdd_tvb import linear_jax as LJ  # noqa: E402
import fit_subjects_nested as FSN  # noqa: E402
import plans as PL  # noqa: E402


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nested run directory with subject_fits.csv")
    ap.add_argument("--bank", nargs="+", required=True, help="bank(s) used by that run, same order")
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--common-drive", action="store_true")
    ap.add_argument("--source-background", action="store_true")
    ap.add_argument("--common-origin", type=float, nargs=3, default=[0.0, -15.0, 60.0])
    ap.add_argument("--common-speed", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--prior", type=float, default=0.02)
    ap.add_argument("--global-prior", type=float, default=0.01)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--emp-dir", default="outputs/m5_spectral_m51_production/empirical",
                    help="directory with cross_spectra_fit.npz / cross_spectra_validation.npz")
    ap.add_argument("--subjects-file", default=None, help="optional text file restricting fitted subjects")
    ap.add_argument("--out", default=None)
    PL.add_plan_arguments(ap)
    args = ap.parse_args()
    run = ROOT / args.run
    out = ROOT / (args.out or f"{args.run}_refined")
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    if args.lead == "bem":
        from mdd_tvb.template_bem import load_template_bem_gain
        gain, _, _ = load_template_bem_gain(ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    else:
        gain = lin.gain
    lead = LJ.average_reference(gain)
    if args.source_background:
        lead_np = np.asarray(gain) - np.asarray(gain).mean(0, keepdims=True)
        cov = lead_np @ lead_np.T
        FSN.SOURCE_COV = jnp.asarray(cov / np.mean(np.diag(cov)))
    units = np.concatenate([np.load(ROOT / b)["unit"] for b in args.bank])
    fits = pd.read_csv(run / "subject_fits.csv")
    networks = list(static.network_names)
    groups = list(static.group_names)
    spatial_cols = [c for c in fits.columns if c.startswith("log_noise_gain_")]
    spatial_names = [c.removeprefix("log_noise_gain_") for c in spatial_cols]
    hemi = spatial_names == groups
    mapping = np.eye(len(groups)) if hemi else np.asarray(
        [[float(g.rsplit("_", 1)[0] == n) for n in networks] for g in groups])
    FSN.MAP = jnp.asarray(mapping)
    FSN.USE_COMMON = bool(args.common_drive)
    d = mapping.shape[1]
    freq = jnp.asarray(np.arange(cfg.spectral.frequency_min_hz, cfg.spectral.frequency_max_hz + 0.5,
                                 cfg.spectral.frequency_bin_hz))
    origin = jnp.asarray(args.common_origin)
    iu = FSN.IU

    def model(theta):
        u, z = theta[:10], theta[10:]
        phys = LJ.unit_to_physical(u)
        p, speed = LJ.build_state(phys, static)
        delays = LJ.delays_for_speed(static, speed)
        if FSN.USE_COMMON:
            C, common, psi = LJ.contributions_with_common_drive(p, static, lead, delays, origin, args.common_speed)
        else:
            C, psi = LJ.network_contributions(p, static, lead, delays)
            common = None
        csd = FSN.combine(C, z, freq, common)
        return csd, p, psi

    def objective(theta, target, T, u0):
        csd, p, psi = model(theta)
        r = weighted_features(csd, T) - target
        z = theta[10:]
        beta = z[:d] - jnp.mean(z[:d])
        absc = LJ.node_abscissa_per_s(p, psi)
        margin = LJ.fold_margin(p, psi)
        sg = LJ.small_gain(p, psi, jnp.asarray(static.fine_frequency_hz))
        penalty = (jnp.mean(jax.nn.relu(absc + 2.0) ** 2) * 1e-2
                   + jnp.mean(jax.nn.relu(0.4 - margin) ** 2) * 10.0
                   + jax.nn.relu(sg - 0.9) ** 2 * 10.0)
        loss = jnp.sum(r**2) + args.prior * jnp.mean(beta**2) + args.global_prior * jnp.mean((theta[:10] - u0) ** 2)
        certified = (jnp.max(absc) < -1.0) & (jnp.min(margin) > 0.3) & (sg < 1.0)
        return loss + penalty, (loss, certified)

    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    predict = jax.jit(lambda theta: model(theta)[0])

    emp = ROOT / args.emp_dir
    only = (set(Path(ROOT / args.subjects_file).read_text().split()) if args.subjects_file else None)
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    rows, preds = [], {}
    t0 = time.time()
    done = 0
    for fold, T, null_csds, _ in PL.plans(args, fit, folds=set(fits.outer_fold.unique())):
        blocks_train = [feature_blocks(jnp.asarray(c), T) for c in null_csds]
        pooled_blocks = [np.mean([np.asarray(b[k]) for b in blocks_train], 0) for k in range(3)]
        pooled_total = np.mean([np.asarray(weighted_features(jnp.asarray(c), T)) for c in null_csds], 0)
        for _, srow in fits[fits.outer_fold == fold].iterrows():
            if args.max_subjects and done >= args.max_subjects:
                break
            sid = srow.subject_id
            if only is not None and sid not in only:
                continue
            i = index[sid]
            target = weighted_features(jnp.asarray(fit.csd[i]), T)
            u0 = jnp.asarray(units[int(srow.state)])
            z0 = list(srow[spatial_cols].to_numpy(float))
            z0 += [logit(srow.observation_noise_fraction / 0.95), logit(srow.observation_noise_exponent / 2.5)]
            if args.source_background:
                z0 += [logit(srow.source_background_fraction / 0.95), logit(srow.source_background_exponent / 3.0)]
            if FSN.USE_COMMON:
                z0 += [logit(srow.common_drive_share / 0.95)]
            theta = jnp.concatenate([u0, jnp.asarray(z0)])
            m = jnp.zeros_like(theta)
            v = jnp.zeros_like(theta)
            best = (np.inf, theta, 0)
            for step in range(args.steps + 1):
                (total, (loss, cert)), g = vg(theta, target, T, u0)
                if bool(cert) and float(loss) < best[0] and np.isfinite(float(loss)):
                    best = (float(loss), theta, step)
                if step == args.steps or not np.all(np.isfinite(np.asarray(g))):
                    break
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                theta = theta - args.lr * (m / (1 - 0.9 ** (step + 1))) / (jnp.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
            theta = best[1]
            csd = predict(theta)
            preds[sid] = np.asarray(csd)
            pred_total = np.asarray(weighted_features(csd, T))
            val_total = np.asarray(weighted_features(jnp.asarray(val.csd[i]), T))
            pb = [np.asarray(b) for b in feature_blocks(csd, T)]
            vb = [np.asarray(b) for b in feature_blocks(jnp.asarray(val.csd[i]), T)]
            fb = [np.asarray(b) for b in feature_blocks(jnp.asarray(fit.csd[i]), T)]
            row = {"subject_id": sid, "group": srow.group, "outer_fold": int(fold),
                   "start_fit_cost": float(srow.fit_cost), "refined_step": int(best[2]),
                   "fit_cost": float(np.sum((pred_total - np.asarray(target)) ** 2)),
                   "validation_ratio": float(np.sum((pred_total - val_total) ** 2) / np.sum((pooled_total - val_total) ** 2)),
                   "fit_ratio": float(np.sum((pred_total - np.asarray(target)) ** 2) /
                                      np.sum((pooled_total - np.asarray(target)) ** 2))}
            for name, k in (("auto", 0), ("cross", 1), ("topo", 2)):
                row[f"validation_{name}_ratio"] = float(np.mean((pb[k] - vb[k]) ** 2) / np.mean((pooled_blocks[k] - vb[k]) ** 2))
                row[f"fit_{name}_ratio"] = float(np.mean((pb[k] - fb[k]) ** 2) / np.mean((pooled_blocks[k] - fb[k]) ** 2))
            phys = np.asarray(LJ.unit_to_physical(theta[:10]))
            for k, name in enumerate(LJ.GLOBAL_NAMES):
                row[name] = float(phys[k])
            zz = np.asarray(theta[10:])
            beta = zz[:d] - zz[:d].mean()
            for k, name in enumerate(spatial_names):
                row[f"log_noise_gain_{name}"] = float(beta[k])
            row["observation_noise_fraction"] = float(0.95 / (1 + np.exp(-zz[d])))
            if args.source_background:
                row["source_background_fraction"] = float(0.95 / (1 + np.exp(-zz[d + 2])))
            if FSN.USE_COMMON:
                row["common_drive_share"] = float(0.95 / (1 + np.exp(-zz[-1])))
            rows.append(row)
            done += 1
            if done % 10 == 0:
                t = pd.DataFrame(rows)
                print(f"{done} {time.time()-t0:.0f}s | unseen total {t.validation_ratio.median():.3f} auto "
                      f"{t.validation_auto_ratio.median():.3f} cross {t.validation_cross_ratio.median():.3f} "
                      f"topo {t.validation_topo_ratio.median():.3f} | fit {t.fit_ratio.median():.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_fits.csv", index=False)
    np.savez_compressed(out / "predictions.npz", subject_ids=np.asarray(list(preds)),
                        csd=np.stack(list(preds.values())))
    summary = {"subjects": len(table), "median_unseen_total_ratio": float(table.validation_ratio.median()),
               "median_unseen_auto_ratio": float(table.validation_auto_ratio.median()),
               "median_unseen_cross_ratio": float(table.validation_cross_ratio.median()),
               "median_unseen_topo_ratio": float(table.validation_topo_ratio.median()),
               "median_fit_total_ratio": float(table.fit_ratio.median()), "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
