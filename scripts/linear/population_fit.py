"""Population (pooled, label-blind) fit of the linear-regime dual-JR model.

Uses only the first (fitting) halves of the 262 M5.2 development subjects.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax, jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_spectral import CandidateLinearizer  # noqa: E402
from mdd_tvb import linear_jax as LJ  # noqa: E402
from mdd_tvb.linear_fit import load_transformer_arrays, weighted_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--transformer", default="outputs/m52_nested_baseline/outer_0/spectral_transformer.npz")
    ap.add_argument("--out", default="outputs/linear_regime/population_fit_tvb.json")
    ap.add_argument("--maxiter", type=int, default=60)
    args = ap.parse_args()
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    if args.lead == "bem":
        from mdd_tvb.template_bem import load_template_bem_gain
        gain, _, _ = load_template_bem_gain(ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    else:
        gain = lin.gain
    lead = LJ.average_reference(gain)
    T = load_transformer_arrays(ROOT / args.transformer)
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
    dev = np.isin(fit.subject_ids.astype(str), splits.subject_id.astype(str))
    feats = np.stack([np.asarray(weighted_features(jnp.asarray(c), T)) for c in fit.csd[dev]])
    target = jnp.asarray(feats.mean(0))
    freq = jnp.asarray(fit.frequency_hz)
    K = static.n_networks
    group_to_network = jnp.asarray(
        [[float(static.group_names[g].rsplit("_", 1)[0] == static.network_names[k]) for k in range(K)]
         for g in range(static.n_groups)])

    def unpack(x):
        phys = LJ.unit_to_physical(x[:10])
        beta = x[10 : 10 + K]
        beta = beta - beta.mean()
        frac = 0.95 * jax.nn.sigmoid(x[10 + K])
        expo = 2.5 * jax.nn.sigmoid(x[11 + K])
        return phys, beta, frac, expo

    def model(x):
        phys, beta, frac, expo = unpack(x)
        p, speed = LJ.build_state(phys, static)
        C, psi = LJ.network_contributions(p, static, lead, LJ.delays_for_speed(static, speed))
        csd = LJ.combine(C, group_to_network @ beta, frac, expo, freq)
        return csd, p, psi

    def loss(x):
        csd, p, psi = model(x)
        r = weighted_features(csd, T) - target
        absc = LJ.node_abscissa_per_s(p, psi)
        stab = jnp.mean(jax.nn.relu(absc + 5.0) ** 2) * 1e-2
        # Stay out of the Jansen-Rit bistable (fold) window.
        stab = stab + jnp.mean(jax.nn.relu(0.6 - LJ.fold_margin(p, psi)) ** 2)
        prior = 1e-3 * jnp.sum(x[:10] ** 2) + 1e-3 * jnp.sum(x[10:] ** 2)
        return jnp.sum(r**2) + stab + prior

    vg = jax.jit(jax.value_and_grad(loss))
    diag = jax.jit(lambda x: (LJ.node_abscissa_per_s(model(x)[1], model(x)[2]).max(),
                              LJ.small_gain(model(x)[1], model(x)[2], jnp.asarray(static.fine_frequency_hz)),
                              LJ.fold_margin(model(x)[1], model(x)[2]).min(),
                              jnp.max(jnp.abs(LJ._residual(model(x)[2], model(x)[1])))))
    starts = [
        [4.0, 6.0, 0.27, 1.15, 1.0, 2.2, 0.25, 1.0, 0.0, 0.0],
        [2.0, 5.0, 0.30, 1.2, 1.3, 2.0, 0.30, 2.0, 0.0, 0.0],
    ]
    best = None
    history = []
    for start in starts:
        x0 = np.concatenate([LJ.physical_to_unit(np.asarray(start)), np.zeros(K), [0.0, 0.0]])
        t0 = time.time()
        def f(x):
            v, g = vg(jnp.asarray(x))
            v = float(v)
            g = np.asarray(g, dtype=float)
            if not np.isfinite(v) or not np.all(np.isfinite(g)):
                return 1e3, np.zeros_like(g)
            return v, g
        v0 = f(x0)[0]
        bounds = [(-5.0, 5.0)] * 10 + [(-4.0, 4.0)] * K + [(-5.0, 5.0)] * 2
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": args.maxiter, "maxls": 30})
        absc, sg, margin, resid = diag(jnp.asarray(res.x))
        phys, beta, frac, expo = unpack(jnp.asarray(res.x))
        row = {
            "start": start, "initial_loss": v0, "loss": float(res.fun), "iterations": int(res.nit),
            "seconds": time.time() - t0,
            "physical": dict(zip(LJ.GLOBAL_NAMES, np.asarray(phys).tolist())),
            "log_network_noise_gain": dict(zip(static.network_names, np.asarray(beta).tolist())),
            "observation_noise_fraction": float(frac), "observation_noise_exponent": float(expo),
            "node_max_abscissa_per_s": float(absc), "small_gain_bound": float(sg),
            "min_fold_margin": float(margin), "equilibrium_residual": float(resid), "x": res.x.tolist(),
        }
        print(json.dumps({k: v for k, v in row.items() if k != "x"}, indent=1), flush=True)
        history.append(row)
        if best is None or row["loss"] < best["loss"]:
            best = row
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "history": history, "lead": args.lead}, indent=1))


if __name__ == "__main__":
    main()
