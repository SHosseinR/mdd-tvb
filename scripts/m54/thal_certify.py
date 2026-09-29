"""Re-certify fitted states on a wide frequency grid (0-150 Hz).

The refinement certifies the network with the small-gain bound on the fitted
grid (2-40 Hz).  A corticothalamic loop is strongest near 0 Hz, so here every
subject's MAP state is re-checked with max_w |h~(iw)| G ||W|| over 0-150 Hz
(h~ = h / (1 - h theta), the node with its own loop) plus the per-region loop
certificate.  Both sufficient: pass = stable.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, LJ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--thalamus", choices=["T1", "T2"], default=None)
    ap.add_argument("--lead", default="bem")
    args = ap.parse_args()
    M.configure(thalamus=args.thalamus)
    setup = M.make_setup(args.lead, json.loads((M.ROOT / args.population).read_text()), use_pop=True)
    k = len(M.FREE_INDEX)
    wide = jnp.arange(0.0, 150.0, 0.05)

    @jax.jit
    def check(theta_globals, extra):
        C_unused = None  # noqa: F841
        p, speed = LJ.build_state(LJ.unit_to_physical(M.full_u(setup, theta_globals)), setup.static)
        if M.THALAMUS:
            p = LJ.with_thalamus(p, LJ.thal_unit_to_physical(M.thal_unit(setup, extra)))
        psi = LJ.equilibrium(p)
        ok, dist = LJ.thalamic_certificate(p, psi)
        return LJ.small_gain(p, psi, wide), LJ.small_gain(p, psi, jnp.asarray(setup.static.fine_frequency_hz)), ok, dist

    t = pd.read_csv(M.ROOT / args.fits)
    rows = []
    for _, r in t.iterrows():
        g = jnp.asarray([r[n] for n in M.FREE_GLOBALS])
        e = jnp.asarray([r[n] for n in M.EXTRA_NAMES]) if M.EXTRA_NAMES else None
        sg_w, sg_f, ok, dist = check(g, e)
        rows.append({"subject_id": r.subject_id, "small_gain_wide": float(sg_w), "small_gain_fitted_grid": float(sg_f),
                     "loop_ok": bool(ok), "loop_min_dist": float(dist)})
    c = pd.DataFrame(rows)
    c["certified_wide"] = (c.small_gain_wide < 1.0) & c.loop_ok & (c.loop_min_dist > 0.05)
    out = Path(M.ROOT / args.fits).with_name("certify_wide.csv")
    c.to_csv(out, index=False)
    print(json.dumps({"n": len(c), "certified_wide": float(c.certified_wide.mean()),
                      "median_small_gain_wide": float(c.small_gain_wide.median()),
                      "max_small_gain_wide": float(c.small_gain_wide.max()),
                      "median_small_gain_fitted": float(c.small_gain_fitted_grid.median())}, indent=1))


if __name__ == "__main__":
    main()
