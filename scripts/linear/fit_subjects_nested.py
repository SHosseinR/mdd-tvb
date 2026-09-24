"""Nested (M5.2 protocol) subject fitting with an analytic linear-regime bank.

For every development subject the outer fold in which it is a *validation*
subject provides the feature transformer and the pooled null (both learned
from that fold's training subjects only).  The subject's first half is fitted;
its second half is only used for scoring.  Diagnosis labels are never used.

Per subject:
1. screen all stable analytic global states with population spatial settings;
2. for the best ``--top`` states, continuously optimise 7 network input-noise
   log-gains and the diagonal observation-noise fraction/exponent (Adam);
3. keep the best penalised state (MAP) and score the unseen half.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax, jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_fit import load_transformer_arrays, feature_blocks, weighted_features  # noqa: E402

N_CH = 26
IU = np.triu_indices(N_CH)


def upper_to_full(u):
    full = jnp.zeros(u.shape[:-1] + (N_CH, N_CH), dtype=u.dtype)
    full = full.at[..., IU[0], IU[1]].set(u)
    diag = jnp.diagonal(full, axis1=-2, axis2=-1)
    full = full + jnp.conjugate(jnp.swapaxes(full, -1, -2))
    idx = jnp.arange(N_CH)
    return full.at[..., idx, idx].set(jnp.real(diag).astype(u.dtype))


MAP = None  # (groups, spatial parameters) matrix, set in main()
SOURCE_COV = None  # lead-field covariance of independent parcels (26, 26) or None
USE_COMMON = False  # shared delayed alpha drive (bank 'common_upper') fitted per subject


def combine(C, z, freq, common=None):
    """C: (groups, F, 26, 26) complex.

    z = [spatial (d), diag_frac_logit, diag_exp_logit(, src_frac_logit, src_exp_logit)].
    The optional source background is independent aperiodic parcel activity
    seen through the lead field (spatially correlated by volume conduction).
    """
    d = MAP.shape[1]
    beta = MAP @ (z[:d] - jnp.mean(z[:d]))
    frac = 0.95 * jax.nn.sigmoid(z[d])
    expo = 2.5 * jax.nn.sigmoid(z[d + 1])
    csd = jnp.einsum("k,kfcd->fcd", jnp.exp(beta), C)
    diag = jnp.real(jnp.diagonal(csd, axis1=1, axis2=2))
    level = jnp.mean(diag)
    if common is not None:
        share = 0.95 * jax.nn.sigmoid(z[-1])
        cdiag = jnp.mean(jnp.real(jnp.diagonal(common, axis1=1, axis2=2)))
        csd = csd + common * (level / cdiag) * share / (1.0 - share)
        level = jnp.mean(jnp.real(jnp.diagonal(csd, axis1=1, axis2=2)))
    shape = freq ** (-expo)
    shape = shape / jnp.mean(shape)
    out = csd + (level * frac / (1.0 - frac) * shape)[:, None, None] * jnp.eye(N_CH)[None]
    if SOURCE_COV is not None:
        sfrac = 0.95 * jax.nn.sigmoid(z[d + 2])
        sexp = 3.0 * jax.nn.sigmoid(z[d + 3])
        sshape = freq ** (-sexp)
        sshape = sshape / jnp.mean(sshape)
        out = out + (level * sfrac / (1.0 - sfrac) * sshape)[:, None, None] * SOURCE_COV[None]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", nargs="+", default=["outputs/linear_regime/analytic_bank_tvb.npz"])
    ap.add_argument("--population", default="outputs/linear_regime/population_fit_tvb.json")
    ap.add_argument("--out", default="outputs/linear_regime/nested_tvb")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prior", type=float, default=0.02)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--spatial", choices=["network", "network_hemisphere"], default="network")
    ap.add_argument("--max-small-gain", type=float, default=1.0)
    ap.add_argument("--source-background", action="store_true")
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--min-fold-margin", type=float, default=0.3)
    ap.add_argument("--max-residual", type=float, default=1e-6,
                    help="equilibrium residual bound (use ~1e-4 for float32 banks)")
    ap.add_argument("--common-drive", action="store_true",
                    help="fit the share of a shared delayed alpha drive (bank must contain common_upper)")
    args = ap.parse_args()
    global MAP, SOURCE_COV, USE_COMMON
    USE_COMMON = bool(args.common_drive)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    loaded = [np.load(ROOT / path) for path in args.bank]
    bank = {key: (np.concatenate([b[key] for b in loaded])
                  if loaded[0][key].ndim and key not in ("frequency_hz", "network_names", "group_names", "global_names")
                  else loaded[0][key])
            for key in loaded[0].files}
    bank["files"] = list(loaded[0].files)
    # Node stability (all local eigenvalues damped) plus the small-gain bound
    # G*||W||_2*max|h| < 1, a sufficient condition for stability of the
    # delayed network: every retained state is a certified stable equilibrium.
    valid = (
        (bank["node_abscissa_per_s"] < -1.0)
        & (bank["fixed_point_residual"] < args.max_residual)
        & (bank["small_gain"] < args.max_small_gain)
        & (bank["fold_margin"] > args.min_fold_margin)
    )
    states = np.flatnonzero(valid)
    print(f"bank states {len(valid)}, stable {valid.sum()}", flush=True)
    contrib = np.asarray(bank["contrib_upper"][states])  # (S,K,F,351) complex64 on host
    common = (np.asarray(bank["common_upper"][states]) if USE_COMMON
              else np.zeros((len(states), 1, 1), dtype=np.complex64))
    freq = jnp.asarray(bank["frequency_hz"])
    groups = contrib.shape[1]
    networks = list(bank["network_names"].astype(str))
    group_names = list(bank["group_names"].astype(str)) if "group_names" in bank["files"] else networks
    if args.spatial == "network":
        mapping = np.asarray([[float(g.rsplit("_", 1)[0] == n) for n in networks] for g in group_names])
        spatial_names = networks
    else:
        mapping = np.eye(groups)
        spatial_names = group_names
    MAP = jnp.asarray(mapping)
    K = mapping.shape[1]
    population = json.loads((ROOT / args.population).read_text())["best"]
    pop_beta = np.asarray(population["x"][10:10 + len(networks)])
    pop_obs = np.asarray(population["x"][10 + len(networks):12 + len(networks)])
    if args.spatial == "network":
        z_spatial = pop_beta
    else:
        z_spatial = np.asarray([pop_beta[networks.index(g.rsplit("_", 1)[0])] for g in group_names])
    z0 = np.concatenate([z_spatial, pop_obs])
    if args.source_background:
        if args.lead == "bem":
            from mdd_tvb.template_bem import load_template_bem_gain
            gain, _, _ = load_template_bem_gain(
                ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
        else:
            from mdd_tvb.spectral_config import load_spectral_m5_config
            from mdd_tvb.linear_spectral import CandidateLinearizer
            gain = CandidateLinearizer(load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")).gain
        lead = gain - gain.mean(0, keepdims=True)
        cov = lead @ lead.T
        SOURCE_COV = jnp.asarray(cov / np.mean(np.diag(cov)))
        z0 = np.concatenate([z0, [-1.0, 0.0]])  # start at ~26 % source background, exponent 1.5
    if USE_COMMON:
        z0 = np.concatenate([z0, [-1.5]])  # start at ~17 % shared-drive power
    z0 = jnp.asarray(z0)

    def full(u, uc):
        return upper_to_full(u), (upper_to_full(uc) if USE_COMMON else None)

    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    emp = ROOT / "outputs/m5_spectral_m51_production/empirical"
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}

    def features(csd, T):
        return weighted_features(csd, T)

    def model_csd(u, uc, z):
        C, cc = full(u, uc)
        return combine(C, z, freq, cc)

    def state_cost(u, uc, z, target, T):
        return jnp.sum((features(model_csd(u, uc, z), T) - target) ** 2)

    state_features = jax.jit(jax.vmap(
        lambda u, uc, z, T: features(model_csd(u, uc, z), T), in_axes=(0, 0, None, None)))

    def all_state_features(T, chunk=64):
        rows = []
        for start in range(0, contrib.shape[0], chunk):
            u = jnp.asarray(contrib[start:start + chunk].astype(np.complex128))
            uc = jnp.asarray(common[start:start + chunk].astype(np.complex128))
            rows.append(np.asarray(state_features(u, uc, z0, T)))
        return np.concatenate(rows)

    def penalised(z, u, uc, target, T):
        beta = z[:K] - jnp.mean(z[:K])
        return state_cost(u, uc, z, target, T) + args.prior * jnp.mean(beta**2)

    grad = jax.grad(penalised)

    def adam(u, uc, target, T, z_init):
        def step(carry, _):
            z, m, v, t = carry
            g = grad(z, u, uc, target, T)
            t = t + 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9**t)
            vh = v / (1 - 0.999**t)
            z = z - 0.08 * mh / (jnp.sqrt(vh) + 1e-8)
            return (z, m, v, t), None
        (z, _, _, _), _ = jax.lax.scan(step, (z_init, jnp.zeros_like(z_init), jnp.zeros_like(z_init), 0.0),
                                       None, length=args.steps)
        return z, penalised(z, u, uc, target, T)

    inner = jax.jit(jax.vmap(adam, in_axes=(0, 0, None, None, None)))
    predict = jax.jit(model_csd)

    rows = []
    predictions = {}
    top_store = {"subject_id": [], "loss": [], "csd_upper": []}
    t0 = time.time()
    done = 0
    for fold in sorted(splits.outer_fold.unique()):
        fr = splits[splits.outer_fold == fold]
        train = np.asarray([index[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str)])
        test = [s for s in fr.loc[fr.outer_role == "validation", "subject_id"].astype(str)]
        T = load_transformer_arrays(ROOT / f"outputs/m52_nested_baseline/outer_{fold}/spectral_transformer.npz")
        screen_features = all_state_features(T)
        print(f"fold {fold}: screened {len(screen_features)} states {time.time()-t0:.0f}s", flush=True)
        # pooled null from training fit halves (blocks and weighted total)
        blocks_train = [feature_blocks(jnp.asarray(fit.csd[i]), T) for i in train]
        pooled_blocks = [np.mean([np.asarray(b[k]) for b in blocks_train], 0) for k in range(3)]
        pooled_total = np.mean([np.asarray(weighted_features(jnp.asarray(fit.csd[i]), T)) for i in train], 0)
        for sid in test:
            if args.max_subjects and done >= args.max_subjects:
                break
            i = index[sid]
            target = weighted_features(jnp.asarray(fit.csd[i]), T)
            costs = np.sum((screen_features - np.asarray(target)[None]) ** 2, axis=1)
            top = np.argsort(costs)[: args.top]
            top_u = jnp.asarray(contrib[top].astype(np.complex128))
            top_c = jnp.asarray(common[top].astype(np.complex128))
            zs, losses = inner(top_u, top_c, target, T, z0)
            losses = np.asarray(losses)
            order = np.argsort(losses)[:12]
            top_csd = np.stack([np.asarray(predict(top_u[m], top_c[m], zs[m]))[:, IU[0], IU[1]] for m in order])
            top_store["subject_id"].append(sid)
            top_store["loss"].append(losses[order])
            top_store["csd_upper"].append(top_csd.astype(np.complex64))
            best = int(np.argmin(losses))
            z = zs[best]
            state = int(states[top[best]])
            csd = predict(top_u[best], top_c[best], z)
            predictions[sid] = np.asarray(csd)
            pred_total = np.asarray(weighted_features(csd, T))
            fit_total = np.asarray(target)
            val_total = np.asarray(weighted_features(jnp.asarray(val.csd[i]), T))
            pb = [np.asarray(b) for b in feature_blocks(csd, T)]
            vb = [np.asarray(b) for b in feature_blocks(jnp.asarray(val.csd[i]), T)]
            fbk = [np.asarray(b) for b in feature_blocks(jnp.asarray(fit.csd[i]), T)]
            row = {
                "subject_id": sid, "group": str(fit.groups[i]), "outer_fold": int(fold),
                "state": state, "screen_best_cost": float(costs[top[0]]),
                "fit_cost": float(np.sum((pred_total - fit_total) ** 2)),
                "fit_null": float(np.sum((pooled_total - fit_total) ** 2)),
                "validation_cost": float(np.sum((pred_total - val_total) ** 2)),
                "validation_null": float(np.sum((pooled_total - val_total) ** 2)),
                "persistence_cost": float(np.sum((fit_total - val_total) ** 2)),
            }
            for name, k in (("auto", 0), ("cross", 1), ("topo", 2)):
                row[f"validation_{name}_ratio"] = float(np.mean((pb[k] - vb[k]) ** 2) / np.mean((pooled_blocks[k] - vb[k]) ** 2))
                row[f"fit_{name}_ratio"] = float(np.mean((pb[k] - fbk[k]) ** 2) / np.mean((pooled_blocks[k] - fbk[k]) ** 2))
            zz = np.asarray(z)
            beta = zz[:K] - zz[:K].mean()
            for k, name in enumerate(spatial_names):
                row[f"log_noise_gain_{name}"] = float(beta[k])
            row["observation_noise_fraction"] = float(0.95 / (1 + np.exp(-zz[K])))
            row["observation_noise_exponent"] = float(2.5 / (1 + np.exp(-zz[K + 1])))
            if USE_COMMON:
                row["common_drive_share"] = float(0.95 / (1 + np.exp(-zz[-1])))
            if SOURCE_COV is not None:
                row["source_background_fraction"] = float(0.95 / (1 + np.exp(-zz[K + 2])))
                row["source_background_exponent"] = float(3.0 / (1 + np.exp(-zz[K + 3])))
            for k, name in enumerate(bank["global_names"].astype(str)):
                row[name] = float(bank["physical"][state, k])
            row["validation_ratio"] = row["validation_cost"] / row["validation_null"]
            row["fit_ratio"] = row["fit_cost"] / row["fit_null"]
            rows.append(row)
            done += 1
            if done % 10 == 0:
                d = pd.DataFrame(rows)
                print(f"{done} subjects {time.time()-t0:.0f}s | median unseen total {d.validation_ratio.median():.3f} "
                      f"auto {d.validation_auto_ratio.median():.3f} cross {d.validation_cross_ratio.median():.3f} "
                      f"topo {d.validation_topo_ratio.median():.3f} obsnoise {d.observation_noise_fraction.median():.2f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_fits.csv", index=False)
    np.savez_compressed(out / "top_states.npz", subject_ids=np.asarray(top_store["subject_id"]),
                        loss=np.stack(top_store["loss"]), csd_upper=np.stack(top_store["csd_upper"]))
    np.savez_compressed(out / "predictions.npz", subject_ids=np.asarray(list(predictions)),
                        csd=np.stack(list(predictions.values())))
    summary = {
        "subjects": len(table),
        "median_unseen_total_ratio": float(table.validation_ratio.median()),
        "fraction_beating_null": float((table.validation_ratio < 1).mean()),
        "median_unseen_auto_ratio": float(table.validation_auto_ratio.median()),
        "median_unseen_cross_ratio": float(table.validation_cross_ratio.median()),
        "median_unseen_topo_ratio": float(table.validation_topo_ratio.median()),
        "median_fit_total_ratio": float(table.fit_ratio.median()),
        "median_observation_noise_fraction": float(table.observation_noise_fraction.median()),
        "stable_states": int(valid.sum()),
        "args": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
