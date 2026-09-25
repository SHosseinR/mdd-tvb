"""Label-blind Whittle population fit (all 10 globals, spatial gains, nuisances).

Maximises the summed profiled Whittle likelihood of the development subjects'
*first halves* (each subject keeps its own overall scale).  The fitted values of
the four non-identifiable globals are then held fixed for every subject fit.
The output JSON also carries ``best.x`` so build_analytic_bank.py can centre a
bank on it.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, W, LJ  # noqa: E402


def m53_start(path, setup):
    x = np.asarray(json.loads(Path(path).read_text())["best"]["x"], float)
    networks = list(setup.static.network_names)
    net_gain = dict(zip(networks, x[10:10 + len(networks)]))
    z = [net_gain[name.rsplit("_", 1)[0]] for name in setup.spatial]
    z += list(x[10 + len(networks):12 + len(networks)]) + [-1.0, 0.0, -1.5]
    return x[:10], np.asarray(z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--emp-dir", required=True)
    ap.add_argument("--subjects-file", default=None, help="subjects whose first halves are used")
    ap.add_argument("--start", default="configs/m53_frozen/population_fit_tvb.json")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup = M.make_setup(args.lead, None)
    u0, z0 = m53_start(M.ROOT / args.start, setup)
    ids = Path(M.ROOT / args.subjects_file).read_text().split() if args.subjects_file else None
    if ids is None:
        ids = sorted(pd.read_csv(M.ROOT / "configs/m52_nested_splits.csv").subject_id.astype(str).unique())
    data = M.load_subjects(args.emp_dir, ids)
    A = jnp.asarray(data.A); pad = jnp.asarray(data.pad)
    Cc = jnp.asarray(data.fit.Cc); nu = jnp.asarray(data.fit.nu)
    dof = float(np.sum(data.fit.nu[:, None] * np.sum(~data.pad, axis=-1)))
    print(f"subjects {len(data.ids)}, total dof {dof:.3g}", flush=True)
    nz = len(z0)

    def objective(x):
        u, z = x[:10], x[10:]
        C, common, p, psi = M.contributions(setup, u)
        S = M.combine(setup, C, common, z)
        nll = jax.vmap(lambda a, pd_, c, n: W.profiled_nll(S, a, pd_, c, n)[0])(A, pad, Cc, nu)
        pen, cert = M.stability(setup, p, psi)
        # tiny ridge on the spatial log-gains removes their free common offset
        return jnp.sum(nll) / dof + pen + 1e-4 * jnp.sum(z[:setup.d] ** 2), cert

    x0 = jnp.asarray(np.concatenate([u0, z0]))
    vg = jax.jit(jax.value_and_grad(lambda x: objective(x)[0]))
    cert_fn = jax.jit(lambda x: objective(x)[1])
    history = []
    t0 = time.time()

    def fun(x):
        v, g = vg(jnp.asarray(x))
        v, g = float(v), np.asarray(g, float)
        if not np.isfinite(v) or not np.all(np.isfinite(g)):
            return 1e3, np.zeros_like(g)
        history.append(v)
        if len(history) % 5 == 0:
            print(f"eval {len(history)} {time.time()-t0:.0f}s nll/dof {v:.5f}", flush=True)
        return v, g

    bounds = [(-6, 6)] * 10 + [(-4, 4)] * setup.d + [(-6, 6)] * 5
    res = minimize(fun, np.asarray(x0), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter})
    x = res.x
    phys = np.asarray(LJ.unit_to_physical(jnp.asarray(x[:10])))
    out = {"best": {"x": x[:10].tolist(), "fun": float(res.fun), "certified": bool(cert_fn(jnp.asarray(x)))},
           "u": x[:10].tolist(), "z": x[10:].tolist(), "start_fun": float(history[0]) if history else None,
           "physical": dict(zip(LJ.GLOBAL_NAMES, phys.tolist())), "spatial_names": setup.spatial,
           "nuisance": dict(zip(M.NUISANCE, x[10 + setup.d:].tolist())), "subjects": len(data.ids),
           "emp_dir": args.emp_dir, "lead": args.lead, "message": str(res.message), "nit": int(res.nit)}
    Path(M.ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    M.save_json(M.ROOT / args.out, out)
    print(json.dumps({k: out[k] for k in ("physical", "nuisance", "start_fun")}, indent=1), "final", res.fun)


if __name__ == "__main__":
    main()
