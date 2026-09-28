"""Scan the corticothalamic loop on top of a population M5.4 state.

For a grid of loop gain, reticular fraction, delay, intrathalamic gain and a
rescaling of the cortical coupling G, report: whether each region's own loop is
stable (thalamic_certificate), the network small-gain value (< 1 certifies the
whole network), the model's mean scalp spectrum shape (alpha peak frequency,
beta peak frequency and "beta excess" = maximum over 16-26 Hz of the spectrum
divided by a power law through 13 and 30 Hz; > 1 is a beta peak), and the
exact network winding number (argument principle on det(I - diag(h) K)) for
settings that fail the conservative small-gain test.
"""
from __future__ import annotations

import argparse, itertools, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, LJ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="outputs/m54/population_bem_m10.json")
    ap.add_argument("--lead", default="bem")
    ap.add_argument("--out", default="outputs/m54/thal_scan.csv")
    args = ap.parse_args()
    pop = json.loads((M.ROOT / args.population).read_text())
    setup = M.make_setup(args.lead, pop, use_pop=False)
    u = jnp.asarray(setup.u_population)
    z = jnp.asarray(setup.z_population)
    phys = LJ.unit_to_physical(u)
    p0, speed = LJ.build_state(phys, setup.static)
    delays = LJ.delays_for_speed(setup.static, speed)
    freq = np.asarray(setup.freq)
    fine = jnp.asarray(setup.static.fine_frequency_hz)
    n = p0["a"].shape[0]
    dense = jnp.arange(0.0, 150.0, 0.05)
    om_dense = 2 * jnp.pi * dense / 1000.0

    @jax.jit
    def run(g_scale, thal):
        p = LJ.with_thalamus(dict(p0, G=p0["G"] * g_scale), thal)
        C, common, psi = LJ.contributions_with_common_drive(p, setup.static, setup.lead, delays,
                                                           jnp.asarray(M.COMMON_ORIGIN), M.COMMON_SPEED)
        ok, dist = LJ.thalamic_certificate(p, psi)
        sg = LJ.small_gain(p, psi, fine)
        beta = setup.mapping @ (z[:setup.d] - jnp.mean(z[:setup.d]))
        pw = jnp.real(jnp.einsum("fcc->fc", jnp.einsum("k,kfcd->fcd", jnp.exp(beta), C)))
        # exact network test: winding of det(I - diag(h) K) along the imaginary axis
        h = LJ.long_range_response(p, psi, 1j * om_dense)
        K = p["G"] * p["W"][None] * jnp.exp(-1j * om_dense[:, None, None] * delays[None])
        K = K + LJ.thalamic_loop(p, om_dense, n)[:, :, None] * jnp.eye(n)[None]
        sign, _ = jnp.linalg.slogdet(jnp.eye(n)[None] - h[:, :, None] * K)
        ang = jnp.angle(sign)
        d = (jnp.diff(ang) + jnp.pi) % (2 * jnp.pi) - jnp.pi
        winding = jnp.round(2 * jnp.sum(d) / (2 * jnp.pi))
        return pw, ok, dist, sg, winding, jnp.real(sign[0])

    def shape(pw):
        m = pw.mean(1)
        m = m / m.mean()
        sel_a = (freq >= 6) & (freq <= 14)
        sel_b = (freq >= 16) & (freq <= 26)
        line = np.exp(np.interp(np.log(freq), np.log([13, 30]), np.log([m[freq == 13][0], m[freq == 30][0]])))
        return (float(freq[sel_a][np.argmax(m[sel_a])]), float(freq[sel_b][np.argmax((m / line)[sel_b])]),
                float((m / line)[sel_b].max()), float(m[sel_a].max() / m[(freq >= 2) & (freq <= 40)].mean()))

    rows, t0 = [], time.time()
    grid = itertools.product([2, 10, 20, 40, 80, 160, 300], [0.0, 0.5, 1.0, 1.4], [60, 75, 90, 110], [0.0, 0.5, 1.5],
                             [1.0, 0.7, 0.4])
    for gain, gamma, t0_ms, kappa, gsc in grid:
        pw, ok, dist, sg, wind, det0 = run(gsc, jnp.asarray([gain, gamma, t0_ms, kappa], float))
        af, bf, bex, apk = shape(np.asarray(pw))
        rows.append({"gain": gain, "gamma": gamma, "t0_ms": t0_ms, "kappa": kappa, "G_scale": gsc, "loop_ok": bool(ok),
                     "min_dist": float(dist), "small_gain": float(sg), "network_winding": float(wind),
                     "det0_positive": bool(det0 > 0), "alpha_peak_hz": af, "beta_peak_hz": bf, "beta_excess": bex,
                     "alpha_peakiness": apk})
        if len(rows) % 100 == 0:
            print(len(rows), f"{time.time()-t0:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    t["exactly_stable"] = t.loop_ok & (t.network_winding == 0) & t.det0_positive
    t["certified"] = t.loop_ok & (t.min_dist > 0.05) & (t.small_gain < 1.0)
    out = M.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    base = t[(t.gain == 2) & (t.gamma == 0) & (t.t0_ms == 75) & (t.kappa == 0) & (t.G_scale == 1.0)]
    print("near-off reference:", base[["small_gain", "alpha_peak_hz", "beta_excess", "alpha_peakiness"]].to_dict("records"))
    for label, sub in (("certified", t[t.certified]), ("exactly stable", t[t.exactly_stable])):
        print(f"{label}: {len(sub)}/{len(t)}; beta excess > 1.05: {int((sub.beta_excess > 1.05).sum())}")
        print(sub.sort_values("beta_excess", ascending=False).head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
