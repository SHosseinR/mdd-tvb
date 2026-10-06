"""Spectrum of each Jansen-Rit generator at the fitted population states.

The fast generator is the alpha column with both synaptic rates multiplied by
fast_ratio.  Its PSP response to its own noise input (with and without the
within-region cross-coupling) shows whether it resonates anywhere.
Output: outputs/m54s_shared/fast_generator_spectrum.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jnp, LJ  # noqa: E402

POPS = {"M5.4": "outputs/m54b_kaggle_bem/results/population_bem_m10.json",
        "shared thalamus": "outputs/m54s_shared/results/population_bem_m10thalS.json"}


def main():
    f = np.arange(0.5, 80.0, 0.1)
    s = jnp.asarray(2j * np.pi * f / 1000.0)
    out = {}
    for label, path in POPS.items():
        pop = json.loads((M.ROOT / path).read_text())
        setup = M.make_setup("bem", pop, use_pop=True)
        phys = LJ.unit_to_physical(jnp.asarray(setup.u_population))
        p, _ = LJ.build_state(phys, setup.static)
        rec = {"fast_ratio": float(phys[LJ.GLOBAL_NAMES.index("fast_ratio")]),
               "fast_fraction": float(phys[LJ.GLOBAL_NAMES.index("fast_fraction")])}
        for tag, pp in (("coupled", p), ("isolated", dict(p, cross=0.0))):
            psi = LJ.equilibrium(pp)
            r, _, bn = LJ.node_responses(pp, psi, s)
            r, bn = np.asarray(r), np.asarray(bn)
            spec = {"alpha": np.median(np.abs(r[:, :, 0, 0] * bn[:, :, 0]) ** 2, axis=1),
                    "fast": np.median(np.abs(r[:, :, 1, 1] * bn[:, :, 1]) ** 2, axis=1)}
            for gen, v in spec.items():
                v = v / v.max()
                rec[f"{gen}_{tag}_peak_hz"] = float(f[np.argmax(v)])
                rec[f"{gen}_{tag}_relative_power"] = {str(x): float(v[f.searchsorted(x)]) for x in (1, 5, 10, 15, 20, 25,
                                                                                                30, 35, 40)}
        out[label] = rec
        print(label, json.dumps({k: v for k, v in rec.items() if "coupled" in k or "ratio" in k or "fraction" in k}))
    dest = M.ROOT / "outputs/m54s_shared/fast_generator_spectrum.json"
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
