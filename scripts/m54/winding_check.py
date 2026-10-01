"""Cross-check the exact network winding test against the factorised certificate.

For private loops (T variants) every region's loop can be certified separately
(scalar argument principle) and the rest by small gain on the node-plus-loop
response.  The exact test (linear_jax.network_winding: argument principle on the
full delayed-network determinant) must agree on states that pass the factorised
test; it is the only certificate for shared nuclei.  Grids of 0.02 and 0.01 Hz (0.1 Hz aliased).
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, LJ  # noqa: E402


def find(pattern):
    hits = glob.glob(pattern, recursive=True) + glob.glob(str(M.ROOT / pattern), recursive=True)
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="/kaggle/input/**/dev_bem_m10popthal2/subject_fits.csv")
    ap.add_argument("--population", default="/kaggle/input/**/population_bem_m10thal.json")
    ap.add_argument("--thalamus", default="T2")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="outputs/m54/winding_check.json")
    args = ap.parse_args()
    M.configure(thalamus=args.thalamus)
    setup = M.make_setup("bem", json.loads(Path(find(args.population)).read_text()), use_pop=True)
    fits = pd.read_csv(find(args.fits)).head(args.n)
    delays = M.delays_of(setup)

    @jax.jit
    def check(g, e):
        p, _ = LJ.build_state(LJ.unit_to_physical(M.full_u(setup, g)), setup.static)
        p = LJ.with_thalamus(p, LJ.thal_unit_to_physical(M.thal_unit(setup, e)))
        psi = LJ.equilibrium(p)
        ok_f, dist = LJ.thalamic_certificate(p, psi, 150.0, 0.1)
        sg = LJ.small_gain(p, psi, M.WIDE_GRID)
        return ok_f & (dist > 0.05) & (sg < 1.0), LJ.network_winding(p, psi, delays, 150.0, 0.02), \
            LJ.network_winding(p, psi, delays, 150.0, 0.01)

    rows = []
    for _, r in fits.iterrows():
        fact, (ok1, w1), (ok2, w2) = check(jnp.asarray([r[n] for n in M.FREE_GLOBALS]),
                                           jnp.asarray([r[n] for n in M.EXTRA_NAMES]))
        rows.append({"subject_id": r.subject_id, "factorised_certified": bool(fact), "exact_ok_0p02": bool(ok1),
                     "winding_0p02": float(w1), "exact_ok_0p01": bool(ok2), "winding_0p01": float(w2)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    out = {"n": len(t), "agree_0p02": float((t.factorised_certified == t.exact_ok_0p02).mean()),
           "agree_0p01": float((t.factorised_certified == t.exact_ok_0p01).mean()),
           "grids_agree": float((t.exact_ok_0p02 == t.exact_ok_0p01).mean()), "rows": rows}
    Path(M.ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (M.ROOT / args.out).write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
