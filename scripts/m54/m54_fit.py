"""Per-subject M5.4 fits: bank screen -> spatial/nuisance Adam -> full refinement -> Laplace.

Scoring: the Wishart deviance of the unseen second half, relative to a pooled
population null (mean normalised CSD of training subjects; nested M5.2 folds on
development data, all development subjects for an external cohort) with the
null's scale also fitted on the first half.  ``--swap-halves`` fits the second
half instead (for split-half reliability and posterior calibration; no scoring).
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, W, LJ  # noqa: E402

N_CH = 26
IU = np.triu_indices(N_CH)


def upper_to_full(u):
    full = jnp.zeros(u.shape[:-1] + (N_CH, N_CH), dtype=u.dtype)
    full = full.at[..., IU[0], IU[1]].set(u)
    diag = jnp.diagonal(full, axis1=-2, axis2=-1)
    full = full + jnp.conjugate(jnp.swapaxes(full, -1, -2))
    idx = jnp.arange(N_CH)
    return full.at[..., idx, idx].set(jnp.real(diag).astype(u.dtype))


def adam(fun, x0, steps, lr):
    grad = jax.grad(fun)

    def step(carry, _):
        x, m, v, t = carry
        g = grad(x)
        t = t + 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x = x - lr * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-8)
        return (x, m, v, t), None

    (x, _, _, _), _ = jax.lax.scan(step, (x0, jnp.zeros_like(x0), jnp.zeros_like(x0), 0.0), None, length=steps)
    return x, fun(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--population", required=True)
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--emp-dir", required=True)
    ap.add_argument("--null-emp-dir", default=None, help="external mode: development collection for the null")
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--dev-subjects-file", default=None, help="restrict the development null to these subjects")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--bank-steps", type=int, default=120)
    ap.add_argument("--refine-steps", type=int, default=60)
    ap.add_argument("--refine-lr", type=float, default=0.03)
    ap.add_argument("--swap-halves", action="store_true")
    ap.add_argument("--no-emg-mask", action="store_true", help="score every channel at every frequency")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--max-residual", type=float, default=1e-4)
    args = ap.parse_args()
    out = M.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    population = json.loads((M.ROOT / args.population).read_text())
    setup = M.make_setup(args.lead, population)
    prior = M.default_prior(setup)
    k = len(M.FREE_INDEX)
    d = setup.d

    only = Path(M.ROOT / args.subjects_file).read_text().split() if args.subjects_file else None
    data = (M.load_subjects(args.emp_dir, only, use_emg_flags=False, fixed_emg=()) if args.no_emg_mask
            else M.load_subjects(args.emp_dir, only))
    fit_half, val_half = (data.val, data.fit) if args.swap_halves else (data.fit, data.val)
    raw_fit = data.raw_val if args.swap_halves else data.raw_fit
    raw_val = data.raw_fit if args.swap_halves else data.raw_val
    print(f"subjects {len(data.ids)}", flush=True)

    # --- null(s) ---------------------------------------------------------------
    splits = pd.read_csv(M.ROOT / "configs/m52_nested_splits.csv")
    if args.null_emp_dir:
        dev = M.load_subjects(args.null_emp_dir, None, use_emg_flags=False)
        dev_ids = set(splits.subject_id.astype(str))
        if args.dev_subjects_file:
            dev_ids &= set(Path(M.ROOT / args.dev_subjects_file).read_text().split())
        null_all = M.pooled_null([dev.raw_fit[i] for i, s in enumerate(dev.ids) if s in dev_ids])
        null_of = {s: null_all for s in data.ids}
        fold_of = {s: -1 for s in data.ids}
    else:
        pos = {s: i for i, s in enumerate(data.ids)}
        null_of, fold_of = {}, {}
        for fold in sorted(splits.outer_fold.unique()):
            fr = splits[splits.outer_fold == fold]
            train = [pos[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str) if s in pos]
            null = M.pooled_null([raw_fit[i] for i in train])
            for s in fr.loc[fr.outer_role == "validation", "subject_id"].astype(str):
                null_of[s], fold_of[s] = null, int(fold)

    # --- bank ------------------------------------------------------------------
    loaded = [np.load(M.ROOT / b) for b in args.bank]
    cat = lambda key: np.concatenate([b[key] for b in loaded])  # noqa: E731
    valid = ((cat("node_abscissa_per_s") < -1.0) & (cat("fixed_point_residual") < args.max_residual)
             & (cat("small_gain") < 1.0) & (cat("fold_margin") > 0.3))
    states = np.flatnonzero(valid)
    contrib = jnp.asarray(cat("contrib_upper")[states])  # kept on the device
    common = jnp.asarray(cat("common_upper")[states])
    units = cat("unit")[states]
    fixed_dev = np.abs(units[:, [LJ.GLOBAL_NAMES.index(n) for n in M.FIXED_GLOBALS]]
                       - setup.u_population[[LJ.GLOBAL_NAMES.index(n) for n in M.FIXED_GLOBALS]]).max()
    print(f"bank states {len(valid)}, certified {len(states)}, max fixed-dim deviation {fixed_dev:.2e}", flush=True)

    z_prior_mean = jnp.asarray(prior.mean[k:])
    z_prior_sd = jnp.asarray(prior.sd[k:])

    def screen_nll(u_c, u_cc, A, pad, Cc, nu):
        S = M.combine(setup, upper_to_full(u_c), upper_to_full(u_cc), jnp.asarray(setup.z_population))
        return W.profiled_nll(S, A, pad, Cc, nu)[0]

    screen = jax.jit(jax.vmap(screen_nll, in_axes=(0, 0, None, None, None, None)))

    def z_objective(z, C, cc, A, pad, Cc, nu):
        S = M.combine(setup, C, cc, z)
        return W.profiled_nll(S, A, pad, Cc, nu)[0] + 0.5 * jnp.sum(((z - z_prior_mean) / z_prior_sd) ** 2)

    def fit_z(u_c, u_cc, A, pad, Cc, nu):
        C, cc = upper_to_full(u_c), upper_to_full(u_cc)
        return adam(lambda z: z_objective(z, C, cc, A, pad, Cc, nu), jnp.asarray(setup.z_population),
                    args.bank_steps, 0.05)

    fit_z_batch = jax.jit(jax.vmap(fit_z, in_axes=(0, 0, None, None, None, None)))

    prior_mean, prior_sd = jnp.asarray(prior.mean), jnp.asarray(prior.sd)

    def full_objective(theta, A, pad, Cc, nu):
        S, p, psi = M.model_csd(setup, theta)
        nll, logs = W.profiled_nll(S, A, pad, Cc, nu)
        pen, cert = M.stability(setup, p, psi)
        post = nll + 0.5 * jnp.sum(((theta - prior_mean) / prior_sd) ** 2)
        return post + pen, (post, nll, logs, cert)

    vg = jax.jit(jax.value_and_grad(full_objective, has_aux=True))
    model_only = jax.jit(lambda theta: M.model_csd(setup, theta)[0])

    def observed(theta_ext, A, pad):
        S = M.model_csd(setup, theta_ext[:-1])[0] * jnp.exp(theta_ext[-1])
        return W.project_model(S, A, pad)

    jac = jax.jit(jax.jacfwd(lambda t, A, pad: observed(t, A, pad), holomorphic=False))

    rows, preds, covs = [], {}, {}
    t0 = time.time()
    for n, sid in enumerate(data.ids):
        if args.max_subjects and n >= args.max_subjects:
            break
        if sid not in null_of:
            continue
        A = jnp.asarray(data.A[n]); pad = jnp.asarray(data.pad[n])
        Cc = jnp.asarray(fit_half.Cc[n]); nu = float(fit_half.nu[n])
        # 1. screen certified bank states at population spatial/nuisance values
        costs = []
        for s0 in range(0, len(states), 128):
            costs.append(np.asarray(screen(contrib[s0:s0 + 128], common[s0:s0 + 128], A, pad, Cc, nu)))
        costs = np.concatenate(costs)
        top = np.argsort(np.where(np.isfinite(costs), costs, np.inf))[: args.top]
        # 2. spatial gains and nuisances on the best states
        zs, zl = fit_z_batch(contrib[top], common[top], A, pad, Cc, nu)
        zl = np.asarray(zl)
        b = int(np.nanargmin(zl))
        theta = jnp.concatenate([jnp.asarray(units[top[b]][M.FREE_INDEX]), zs[b]])
        # 3. continuous refinement of all free parameters (best certified iterate kept)
        m_ = jnp.zeros_like(theta); v_ = jnp.zeros_like(theta)
        best = (np.inf, theta, -1, None)
        for step in range(args.refine_steps + 1):
            (tot, (post, nll, logs, cert)), g = vg(theta, A, pad, Cc, nu)
            if bool(cert) and np.isfinite(float(post)) and float(post) < best[0]:
                best = (float(post), theta, step, float(logs))
            if step == args.refine_steps or not np.all(np.isfinite(np.asarray(g))):
                break
            m_ = 0.9 * m_ + 0.1 * g
            v_ = 0.999 * v_ + 0.001 * g * g
            theta = theta - args.refine_lr * (m_ / (1 - 0.9 ** (step + 1))) / (jnp.sqrt(v_ / (1 - 0.999 ** (step + 1))) + 1e-8)
        if best[3] is None:  # no certified iterate: keep the bank solution
            theta = jnp.concatenate([jnp.asarray(units[top[b]][M.FREE_INDEX]), zs[b]])
            (_, (post, nll, logs, cert)), _ = vg(theta, A, pad, Cc, nu)
            best = (float(post), theta, -1, float(logs))
        theta, logs = best[1], best[3]
        (_, (post, nll, _, cert)), _ = vg(theta, A, pad, Cc, nu)
        # 4. Laplace posterior: Fisher information of the Whittle likelihood + prior precision
        ext = jnp.concatenate([theta, jnp.asarray([logs])])
        Sc = observed(ext, A, pad)
        J = jac(ext, A, pad)  # (F, 25, 25, P)
        Sinv = jnp.linalg.inv(Sc)
        X = jnp.einsum("fij,fjkp->fikp", Sinv, J)
        fisher = nu * jnp.real(jnp.einsum("fijp,fjiq->pq", X, X))
        precision = np.array(fisher, dtype=float)
        precision[:-1, :-1] += np.diag(1.0 / np.asarray(prior.sd) ** 2)
        precision[-1, -1] += 1e-6
        try:
            cov = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            cov = np.full_like(precision, np.nan)
        # 5. predictions (raw units) and scoring on the other half
        S0 = np.asarray(model_only(theta), dtype=np.complex128)
        pred_raw = S0 * np.exp(logs) * fit_half.scale[n]
        preds[sid] = pred_raw.astype(np.complex64)
        covs[sid] = cov.astype(np.float32)
        Dn, An, padn = data.D[n], data.A[n], data.pad[n]
        null_raw = null_of[sid]
        null_c = np.einsum("fij,fjk,flk->fil", Dn, null_raw, Dn)
        _, null_logs = W.profiled_nll(jnp.asarray(null_raw), jnp.asarray(Dn), pad, Cc, nu)
        ratio = fit_half.scale[n] / val_half.scale[n]
        Cv = val_half.Cc[n]
        nu_v = float(val_half.nu[n])
        dev_model = W.deviance(S0, logs + np.log(ratio), An, data.pad[n], Cv, nu_v)
        dev_null = W.deviance(null_raw, float(null_logs) + np.log(ratio), Dn, data.pad[n], Cv, nu_v)
        dev_persist = W.deviance(raw_fit[n] / val_half.scale[n], 0.0, Dn, data.pad[n], Cv, nu_v)
        fit_dev_model = W.deviance(S0, logs, An, data.pad[n], np.asarray(Cc), nu)
        fit_dev_null = W.deviance(null_raw, float(null_logs), Dn, data.pad[n], np.asarray(Cc), nu)
        r_obs = float(np.sum(~data.pad[n]) )
        row = {"subject_id": sid, "group": data.groups[n], "outer_fold": fold_of[sid], "state": int(states[top[b]]),
               "refine_step": best[2], "certified": bool(cert), "nu_fit": nu, "nu_val": nu_v,
               "fit_nll_per_dof": float(nll) / (nu * r_obs), "log_scale": float(logs) + float(np.log(fit_half.scale[n])),
               "fit_deviance_ratio": fit_dev_model / fit_dev_null,
               "validation_deviance_ratio": dev_model / dev_null,
               "persistence_deviance_ratio": dev_persist / dev_null,
               "validation_deviance_per_dof": dev_model / (nu_v * r_obs),
               "null_deviance_per_dof": dev_null / (nu_v * r_obs)}
        names = setup.theta_names()
        sd = np.sqrt(np.clip(np.diag(cov)[:-1], 0, None))
        for j, name in enumerate(names):
            row[name] = float(theta[j])
            row[f"sd_{name}"] = float(sd[j])
        phys = np.asarray(LJ.unit_to_physical(M.full_u(setup, theta[:k])))
        for j, name in enumerate(LJ.GLOBAL_NAMES):
            row[f"phys_{name}"] = float(phys[j])
        z = np.asarray(theta[k:])
        beta = z[:d] - z[:d].mean()
        centre = np.eye(d) - 1.0 / d  # centred gains remove the unidentified common offset
        beta_cov = centre @ cov[k:k + d, k:k + d] @ centre.T
        beta_sd = np.sqrt(np.clip(np.diag(beta_cov), 0, None))
        for j, name in enumerate(setup.spatial):
            row[f"gain_{name}"] = float(beta[j])
            row[f"sd_gain_{name}"] = float(beta_sd[j])
        for j, name in enumerate(M.NUISANCE):
            scale_ = {"obs_fraction": 0.95, "obs_exponent": 2.5, "src_fraction": 0.95, "src_exponent": 3.0,
                      "common_share": 0.95}[name]
            row[f"value_{name}"] = float(scale_ / (1 + np.exp(-z[d + j])))
        rows.append(row)
        if len(rows) % 10 == 0:
            t = pd.DataFrame(rows)
            print(f"{len(rows)} {time.time()-t0:.0f}s | unseen deviance ratio median {t.validation_deviance_ratio.median():.3f} "
                  f"(ceiling {t.persistence_deviance_ratio.median():.3f}) fit {t.fit_deviance_ratio.median():.3f} "
                  f"certified {t.certified.mean():.2f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_fits.csv", index=False)
    np.savez_compressed(out / "predictions.npz", subject_ids=np.asarray(list(preds)), csd=np.stack(list(preds.values())))
    np.savez_compressed(out / "posterior_cov.npz", subject_ids=np.asarray(list(covs)), cov=np.stack(list(covs.values())),
                        names=np.asarray(setup.theta_names() + ["log_scale"]))
    summary = {"subjects": len(table), "median_validation_deviance_ratio": float(table.validation_deviance_ratio.median()),
               "median_persistence_deviance_ratio": float(table.persistence_deviance_ratio.median()),
               "fraction_beating_null": float((table.validation_deviance_ratio < 1).mean()),
               "median_fit_deviance_ratio": float(table.fit_deviance_ratio.median()),
               "certified_fraction": float(table.certified.mean()), "args": vars(args)}
    M.save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
