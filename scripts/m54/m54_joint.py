"""Joint eyes-closed + eyes-open M5.4 fits with a small set of condition changes.

Following Rowe et al. (2004) and Hartoyo et al. (2020): one parameter set explains
both recordings of a subject, and only the parameters named by ``--delta`` may
differ with eyes open (theta_EO = theta_EC + delta).  Recording nuisances
(observation / source noise level and colour, population-background share) are
separate per condition.  The two recordings share ONE overall scale: they come
from the same amplifier and head in one session, so the eyes-open/eyes-closed
power ratio must be explained by the parameters (``--delta level`` adds a free
eyes-open scale instead, i.e. separate scales, as a comparison).

Per subject: bank screen and spatial/nuisance fit on the eyes-closed first half
(as m54_fit), then joint refinement of theta, delta and the eyes-open nuisances on
both first halves, a Laplace posterior from the joint Fisher information, and
scoring on both second halves against each condition's population null.

Delta sets: none, level, mu, a_scale, b_scale, coupling, common, vis, all.
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
from m54_fit import adam, upper_to_full  # noqa: E402

DELTA_SETS = {"none": [], "level": [], "mu": ["mu"], "a_scale": ["a_scale"], "b_scale": ["b_scale"],
              "coupling": ["global_coupling"], "common": ["common_share"], "vis": ["gain_Vis_LH", "gain_Vis_RH"],
              "all": ["ALL"]}


def delta_index(setup, names):
    k, d = len(M.FREE_INDEX), setup.d
    idx, labels = [], []
    for n in names:
        if n == "ALL":
            idx += list(range(k)) + [k + j for j in range(d)] + [k + d + setup.nuisance.index("common_share")]
            labels += list(M.FREE_GLOBALS) + [f"gain_{s}" for s in setup.spatial] + ["common_share"]
        elif n in M.FREE_GLOBALS:
            idx.append(M.FREE_GLOBALS.index(n)); labels.append(n)
        elif n.startswith("gain_"):
            idx.append(k + setup.spatial.index(n[5:])); labels.append(n)
        elif n == "common_share":
            idx.append(k + d + setup.nuisance.index("common_share")); labels.append(n)
        else:
            raise ValueError(n)
    return np.asarray(idx, int), labels


def condition_nulls(data, raw_fit, splits, null_dir, dev_ids_file, swap):
    """Population null per subject (nested folds within the data, or the development cohort)."""
    if null_dir:
        dev = M.load_subjects(null_dir, None, use_emg_flags=False)
        ids = set(splits.subject_id.astype(str))
        if dev_ids_file:
            ids &= set(Path(M.ROOT / dev_ids_file).read_text().split())
        null = M.pooled_null([dev.raw_fit[i] for i, s in enumerate(dev.ids) if s in ids])  # as m54_fit
        return {s: null for s in data.ids}, {s: -1 for s in data.ids}
    pos = {s: i for i, s in enumerate(data.ids)}
    null_of, fold_of = {}, {}
    for fold in sorted(splits.outer_fold.unique()):
        fr = splits[splits.outer_fold == fold]
        train = [pos[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str) if s in pos]
        null = M.pooled_null([raw_fit[i] for i in train])
        for s in fr.loc[fr.outer_role == "validation", "subject_id"].astype(str):
            null_of[s], fold_of[s] = null, int(fold)
    return null_of, fold_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["tvb", "bem"], default="bem")
    ap.add_argument("--population", required=True)
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--emp-ec", required=True)
    ap.add_argument("--emp-eo", required=True)
    ap.add_argument("--subjects-file", default=None, help="subjects (only those with both recordings are fitted)")
    ap.add_argument("--eo-subjects-file", default=None, help="usable eyes-open recordings (QC list)")
    ap.add_argument("--splits", default="configs/m52_nested_splits.csv")
    ap.add_argument("--null-ec", default=None, help="external mode: development eyes-closed collection for the null")
    ap.add_argument("--null-eo", default=None, help="external mode: development eyes-open collection for the null")
    ap.add_argument("--dev-subjects-file", default=None)
    ap.add_argument("--delta", choices=sorted(DELTA_SETS), default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--bank-steps", type=int, default=120)
    ap.add_argument("--refine-steps", type=int, default=80)
    ap.add_argument("--refine-lr", type=float, default=0.03)
    ap.add_argument("--swap-halves", action="store_true")
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--pop-background", action="store_true")
    ap.add_argument("--thalamus", choices=["T1", "T2", "S1", "S2"], default=None)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--max-residual", type=float, default=1e-4)
    args = ap.parse_args()
    out = M.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    M.configure(thalamus=args.thalamus)
    setup = M.make_setup(args.lead, json.loads((M.ROOT / args.population).read_text()), use_pop=args.pop_background)
    prior = M.default_prior(setup)
    k, d, nn = len(M.FREE_INDEX), setup.d, len(setup.nuisance)
    ntheta = len(prior.mean)
    didx, dlabels = delta_index(setup, DELTA_SETS[args.delta])
    use_level = args.delta == "level"
    eo_nuis = np.asarray([k + d + j for j, n in enumerate(setup.nuisance) if n != "common_share"], int)
    m, ne = len(didx), len(eo_nuis)

    only = Path(M.ROOT / args.subjects_file).read_text().split() if args.subjects_file else None
    ec = M.load_subjects(args.emp_ec, only)
    eo_ok = set(Path(M.ROOT / args.eo_subjects_file).read_text().split()) if args.eo_subjects_file else None
    eo = M.load_subjects(args.emp_eo, [s for s in ec.ids if eo_ok is None or s in eo_ok])
    both = [s for s in ec.ids if s in set(eo.ids)]
    ec = M.load_subjects(args.emp_ec, both)
    eo = M.load_subjects(args.emp_eo, both)
    assert ec.ids == eo.ids
    halves = lambda D_: (D_.val, D_.fit) if args.swap_halves else (D_.fit, D_.val)  # noqa: E731
    fit1, val1 = halves(ec)
    fit2, val2 = halves(eo)
    raw1 = ec.raw_val if args.swap_halves else ec.raw_fit
    raw2 = eo.raw_val if args.swap_halves else eo.raw_fit
    print(f"subjects with both recordings {len(both)}; delta = {args.delta} {dlabels}", flush=True)
    splits = pd.read_csv(M.ROOT / args.splits)
    null1, fold_of = condition_nulls(ec, raw1, splits, args.null_ec, args.dev_subjects_file, args.swap_halves)
    null2, _ = condition_nulls(eo, raw2, splits, args.null_eo, args.dev_subjects_file, args.swap_halves)

    loaded = [np.load(M.ROOT / b) for b in args.bank]
    cat = lambda key: np.concatenate([b[key] for b in loaded])  # noqa: E731
    valid = ((cat("node_abscissa_per_s") < -1.0) & (cat("fixed_point_residual") < args.max_residual)
             & (cat("small_gain") < 1.0) & (cat("fold_margin") > 0.3))
    states = np.flatnonzero(valid)
    contrib, common = jnp.asarray(cat("contrib_upper")[states]), jnp.asarray(cat("common_upper")[states])
    units = cat("unit")[states]
    print(f"bank states {len(valid)}, certified {len(states)}", flush=True)

    z_pop = jnp.asarray(setup.z_population)
    zmean, zsd = jnp.asarray(prior.mean[k:k + len(setup.z_population)]), jnp.asarray(prior.sd[k:k + len(setup.z_population)])

    def screen_nll(u_c, u_cc, A, pad, Cc, nu, B):
        S = M.combine(setup, upper_to_full(u_c), upper_to_full(u_cc), z_pop, B)
        return W.profiled_nll(S, A, pad, Cc, nu)[0]

    screen = jax.jit(jax.vmap(screen_nll, in_axes=(0, 0, None, None, None, None, None)))

    def fit_z(u_c, u_cc, A, pad, Cc, nu, B):
        C, cc = upper_to_full(u_c), upper_to_full(u_cc)
        obj = lambda z: (W.profiled_nll(M.combine(setup, C, cc, z, B), A, pad, Cc, nu)[0]  # noqa: E731
                         + 0.5 * jnp.sum(((z - zmean) / zsd) ** 2))
        return adam(obj, z_pop, args.bank_steps, 0.05)

    fit_z_batch = jax.jit(jax.vmap(fit_z, in_axes=(0, 0, None, None, None, None, None)))

    pmean, psd = jnp.asarray(prior.mean), jnp.asarray(prior.sd)
    nuis_mean = jnp.asarray(prior.mean[eo_nuis])

    def unpack(phi):
        th = phi[:ntheta]
        dl = phi[ntheta:ntheta + m]
        ze = phi[ntheta + m:ntheta + m + ne]
        lev = phi[-1] if use_level else 0.0
        th2 = th.at[jnp.asarray(eo_nuis)].set(ze)
        if m:
            th2 = th2.at[jnp.asarray(didx)].add(dl)
        return th, th2, dl, ze, lev

    def models(phi, B1, B2):
        th, th2, dl, ze, lev = unpack(phi)
        S1, p1, psi1 = M.model_csd(setup, th, B1)
        S2, p2, psi2 = M.model_csd(setup, th2, B2)
        return S1, S2 * jnp.exp(lev), (p1, psi1), (p2, psi2)

    def objective(phi, A, pad, Cc, nu, B1, B2):
        th, th2, dl, ze, lev = unpack(phi)
        S1, S2, st1, st2 = models(phi, B1, B2)
        nll, logs = W.profiled_nll(jnp.concatenate([S1, S2]), A, pad, Cc, nu)
        pen1, c1 = M.stability(setup, *st1)
        pen2, c2 = M.stability(setup, *st2)
        pri = (0.5 * jnp.sum(((th - pmean) / psd) ** 2) + 0.5 * jnp.sum(dl ** 2)
               + 0.5 * jnp.sum(((ze - nuis_mean) / 1.5) ** 2) + 0.5 * (lev / 2.0) ** 2)
        return nll + pri + pen1 + pen2, (nll + pri, nll, logs, c1 & c2)

    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    model_pair = jax.jit(lambda phi, B1, B2: models(phi, B1, B2)[:2])

    def observed(phi_ext, A, pad, B1, B2):
        S1, S2, _, _ = models(phi_ext[:-1], B1, B2)
        return W.project_model(jnp.concatenate([S1, S2]) * jnp.exp(phi_ext[-1]), A, pad)

    jvp_one = jax.jit(jax.vmap(lambda e, v, A, pad, B1, B2: jax.jvp(lambda x: observed(x, A, pad, B1, B2), (e,), (v,))[1],
                               in_axes=(None, 0, None, None, None, None)))

    def jac(e, A, pad, B1, B2, chunk=8):
        """Jacobian (F, 25, 25, P) by forward-mode products in chunks (memory-light for many parameters)."""
        eye = jnp.eye(e.shape[0])
        cols = [jvp_one(e, eye[i:i + chunk], A, pad, B1, B2) for i in range(0, e.shape[0], chunk)]
        return jnp.moveaxis(jnp.concatenate(cols), 0, -1)
    freq_np = np.asarray(setup.freq)
    F = len(freq_np)
    rows, preds = [], {}
    t0 = time.time()
    for n, sid in enumerate(ec.ids):
        if args.max_subjects and n >= args.max_subjects:
            break
        if sid not in null1 or sid not in null2:
            continue
        per = []
        for D_, fh, vh, null in ((ec, fit1, val1, null1[sid]), (eo, fit2, val2, null2[sid])):
            Dn, An, padn, Cf, Cv = D_.D[n], D_.A[n], D_.pad[n], fh.Cc[n], vh.Cc[n]
            if args.modes:
                Dn, An, pad_new, E = M.mode_projection(Dn, An, padn, null, freq_np, args.modes)
                Cf, Cv = M.project_modes(Cf, E, padn, pad_new), M.project_modes(Cv, E, padn, pad_new)
                padn = pad_new
            per.append({"D": Dn, "A": An, "pad": padn, "Cf": Cf, "Cv": Cv, "nu": float(fh.nu[n]), "nu_v": float(vh.nu[n]),
                        "scale": float(fh.scale[n]), "scale_v": float(vh.scale[n]), "null": null})
        a, b = per
        # eyes-open data in eyes-closed first-half units (one shared scale)
        ratio = b["scale"] / a["scale"]
        eye = np.eye(W.N_OBS)[None] * b["pad"][:, :, None]
        Cf2 = (b["Cf"] - eye) * ratio + eye
        A = jnp.asarray(np.concatenate([a["A"], b["A"]])); pad = jnp.asarray(np.concatenate([a["pad"], b["pad"]]))
        Cc = jnp.asarray(np.concatenate([a["Cf"], Cf2]))
        nu = jnp.asarray(np.concatenate([np.full(F, a["nu"]), np.full(F, b["nu"])]))
        B1, B2 = jnp.asarray(a["null"]), jnp.asarray(b["null"])
        # 1-2. eyes-closed bank screen and spatial/nuisance fit (as m54_fit)
        A1, pad1, C1 = jnp.asarray(a["A"]), jnp.asarray(a["pad"]), jnp.asarray(a["Cf"])
        costs = np.concatenate([np.asarray(screen(contrib[s:s + 128], common[s:s + 128], A1, pad1, C1, a["nu"], B1))
                                for s in range(0, len(states), 128)])
        top = np.argsort(np.where(np.isfinite(costs), costs, np.inf))[: args.top]
        zs, zl = fit_z_batch(contrib[top], common[top], A1, pad1, C1, a["nu"], B1)
        bi = int(np.nanargmin(np.asarray(zl)))
        theta0 = jnp.concatenate([jnp.asarray(units[top[bi]][M.FREE_INDEX]), zs[bi], jnp.asarray(M.extra_init(setup))])
        phi = jnp.concatenate([theta0, jnp.zeros(m), theta0[jnp.asarray(eo_nuis)], jnp.zeros(1 if use_level else 0)])
        # 3. joint refinement (best certified iterate kept)
        m_, v_ = jnp.zeros_like(phi), jnp.zeros_like(phi)
        best = (np.inf, phi, -1, None)
        for step in range(args.refine_steps + 1):
            (tot, (post, nll, logs, cert)), g = vg(phi, A, pad, Cc, nu, B1, B2)
            if bool(cert) and np.isfinite(float(post)) and float(post) < best[0]:
                best = (float(post), phi, step, float(logs))
            if step == args.refine_steps or not np.all(np.isfinite(np.asarray(g))):
                break
            m_ = 0.9 * m_ + 0.1 * g
            v_ = 0.999 * v_ + 0.001 * g * g
            phi = phi - args.refine_lr * (m_ / (1 - 0.9 ** (step + 1))) / (jnp.sqrt(v_ / (1 - 0.999 ** (step + 1))) + 1e-8)
        certified = best[3] is not None
        if not certified:
            (_, (post, nll, logs, _)), _ = vg(phi, A, pad, Cc, nu, B1, B2)
            best = (float(post), phi, -1, float(logs))
        phi, logs = best[1], best[3]
        # 4. Laplace posterior from the joint Fisher information
        ext = jnp.concatenate([phi, jnp.asarray([logs])])
        Sc = observed(ext, A, pad, B1, B2)
        J = jac(ext, A, pad, B1, B2)
        X = jnp.einsum("fij,fjkp->fikp", jnp.linalg.inv(Sc), J)
        fisher = np.array(jnp.real(jnp.einsum("f,fijp,fjiq->pq", nu, X, X)), dtype=float)
        prec_prior = np.concatenate([1.0 / np.asarray(prior.sd) ** 2, np.ones(m), np.full(ne, 1 / 1.5 ** 2),
                                     np.full(1 if use_level else 0, 0.25), [1e-6]])
        try:
            cov = np.linalg.inv(fisher + np.diag(prec_prior))
        except np.linalg.LinAlgError:
            cov = np.full_like(fisher, np.nan)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        # 5. scoring on both second halves (each condition's own null keeps its own fitted scale)
        S1, S2 = (np.asarray(x, np.complex128) for x in model_pair(phi, B1, B2))
        row = {"subject_id": sid, "group": ec.groups[n], "outer_fold": fold_of.get(sid, -1), "certified": certified,
               "refine_step": best[2], "delta_set": args.delta}
        dev_tot, null_tot = 0.0, 0.0
        for tag, c, S, shift in (("ec", a, S1, np.log(a["scale"] / a["scale_v"])),
                                 ("eo", b, S2, np.log(a["scale"] / b["scale_v"]))):
            _, nl = W.profiled_nll(jnp.asarray(c["null"]), jnp.asarray(c["D"]), jnp.asarray(c["pad"]),
                                   jnp.asarray(c["Cf"]), c["nu"])
            dm = W.deviance(S, logs + shift, c["A"], c["pad"], c["Cv"], c["nu_v"])
            dn = W.deviance(c["null"], float(nl) + np.log(c["scale"] / c["scale_v"]), c["D"], c["pad"], c["Cv"], c["nu_v"])
            row[f"validation_deviance_ratio_{tag}"] = dm / dn
            dev_tot += dm
            null_tot += dn
            preds.setdefault(tag, {})[sid] = (S * np.exp(logs) * a["scale"]).astype(np.complex64)
        row["validation_deviance_ratio_joint"] = dev_tot / null_tot
        row["validation_deviance_joint"] = dev_tot
        row["log_eo_over_ec_power_data"] = float(np.log(b["scale"] / a["scale"]))
        # the same quantity for the model: mean observed-coordinate power, eyes open over eyes closed
        obs_power = lambda S, c: np.mean(np.real(np.einsum("fij,fjk,fik->fi", c["A"], S, c["A"]))[~c["pad"]])  # noqa: E731
        row["log_eo_over_ec_power_model"] = float(np.log(obs_power(S2, b) / obs_power(S1, a)))
        names = setup.theta_names()
        th = np.asarray(phi[:ntheta])
        for j, nm in enumerate(names):
            row[nm], row[f"sd_{nm}"] = float(th[j]), float(sd[j])
        z = th[k:k + d]
        beta = z - z.mean()
        centre = np.eye(d) - 1.0 / d
        bsd = np.sqrt(np.clip(np.diag(centre @ cov[k:k + d, k:k + d] @ centre.T), 0, None))
        for j, s_ in enumerate(setup.spatial):
            row[f"gain_{s_}"], row[f"sd_gain_{s_}"] = float(beta[j]), float(bsd[j])
        for j, lab in enumerate(dlabels):
            row[f"delta_{lab}"], row[f"sd_delta_{lab}"] = float(phi[ntheta + j]), float(sd[ntheta + j])
        if use_level:
            row["delta_level"], row["sd_delta_level"] = float(phi[-1]), float(sd[len(phi) - 1])
        row["log_scale"] = float(logs) + float(np.log(a["scale"]))
        rows.append(row)
        if len(rows) % 10 == 0:
            t = pd.DataFrame(rows)
            print(f"{len(rows)} {time.time()-t0:.0f}s | joint {t.validation_deviance_ratio_joint.median():.3f} "
                  f"EC {t.validation_deviance_ratio_ec.median():.3f} EO {t.validation_deviance_ratio_eo.median():.3f} "
                  f"certified {t.certified.mean():.2f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_fits.csv", index=False)
    for tag, p in preds.items():
        np.savez_compressed(out / f"predictions_{tag}.npz", subject_ids=np.asarray(list(p)), csd=np.stack(list(p.values())))
    summary = {"subjects": len(table), "delta": args.delta, "delta_parameters": dlabels,
               **{f"median_{c}": float(table[c].median()) for c in table.columns if c.startswith("validation_deviance_ratio")},
               "total_validation_deviance": float(table.validation_deviance_joint.sum()),
               "r_eo_ec_power_ratio_model_vs_data": float(np.corrcoef(table.log_eo_over_ec_power_model,
                                                                      table.log_eo_over_ec_power_data)[0, 1]),
               "certified_fraction": float(table.certified.mean()), "args": vars(args)}
    M.save_json(out / "summary.json", summary)
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "args"}, indent=1))


if __name__ == "__main__":
    main()
