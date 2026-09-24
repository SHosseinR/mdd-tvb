"""Build an analytic (noise-free, exact-expectation) bank of global states.

Each state stores per-network CSD contributions (unit network noise gain), so
network noise gains and observation nuisances can later be fitted per
subject continuously.  Only the Hermitian upper triangle is stored
(complex64) to keep the bank in memory.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax, jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.linear_spectral import CandidateLinearizer  # noqa: E402
from mdd_tvb import linear_jax as LJ  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=["tvb", "bem"], default="tvb")
    ap.add_argument("--population", default="outputs/linear_regime/population_fit_tvb.json")
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--out", default="outputs/linear_regime/analytic_bank_tvb.npz")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--half-width", type=float, default=1.6,
                    help="half-width of the sampling box in unconstrained (logit) units")
    ap.add_argument("--common-origin", type=float, nargs=3, default=[0.0, -15.0, 60.0])
    ap.add_argument("--common-speed", type=float, default=3.0)
    ap.add_argument("--adaptive-from", nargs="*", default=None,
                    help="subject_fits.csv files; sample around the global states subjects selected")
    ap.add_argument("--adaptive-bank", nargs="*", default=None,
                    help="bank(s) whose 'unit' vectors the subject_fits 'state' column indexes")
    args = ap.parse_args()
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    if args.lead == "bem":
        from mdd_tvb.template_bem import load_template_bem_gain
        gain, _, _ = load_template_bem_gain(
            ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    else:
        gain = lin.gain
    lead = LJ.average_reference(gain)
    population = json.loads((ROOT / args.population).read_text())["best"]
    center = np.asarray(population["x"][:10])
    # Sobol box in logit space around the population state, clipped to
    # the declared physical bounds by construction of the sigmoid map.
    unit = qmc.Sobol(d=10, scramble=True, seed=args.seed).random(args.samples)
    if args.adaptive_from:
        import pandas as pd
        anchors = []
        for fits_path, bank_path in zip(args.adaptive_from, args.adaptive_bank, strict=True):
            fits = pd.read_csv(ROOT / fits_path)
            units = np.load(ROOT / bank_path)["unit"]
            anchors.append(units[fits["state"].to_numpy(int)])
        anchors = np.concatenate(anchors)
        # Local Sobol perturbations around subject-selected states (label-blind:
        # only first-half fits chose them); anchors reused in proportion to use.
        rng = np.random.default_rng(args.seed)
        picks = anchors[rng.integers(0, len(anchors), args.samples)]
        u = picks + (2.0 * unit - 1.0) * args.half_width
    else:
        u = center[None, :] + (2.0 * unit - 1.0) * args.half_width
        u[0] = center

    origin = np.asarray(args.common_origin, dtype=float)

    @jax.jit
    def evaluate(uu):
        phys = LJ.unit_to_physical(uu)
        p, speed = LJ.build_state(phys, static)
        C, common, psi = LJ.contributions_with_common_drive(
            p, static, lead, LJ.delays_for_speed(static, speed), origin, args.common_speed)
        absc = LJ.node_abscissa_per_s(p, psi)
        sg = LJ.small_gain(p, psi, jnp.asarray(static.fine_frequency_hz))
        resid = jnp.max(jnp.abs(psi - LJ._psi_map(psi, p)))
        margin = jnp.min(LJ.fold_margin(p, psi))
        return C, common, phys, jnp.max(absc), sg, resid, margin

    iu = np.triu_indices(26)
    K = static.n_groups
    F = len(static.bin_groups)
    contrib = np.zeros((args.samples, K, F, iu[0].size), dtype=np.complex64)
    phys_all = np.zeros((args.samples, 10))
    absc_all = np.zeros(args.samples)
    sg_all = np.zeros(args.samples)
    res_all = np.zeros(args.samples)
    margin_all = np.zeros(args.samples)
    common_all = np.zeros((args.samples, F, iu[0].size), dtype=np.complex64)
    t0 = time.time()
    for i in range(args.samples):
        C, common, phys, absc, sg, resid, margin = evaluate(jnp.asarray(u[i]))
        C = np.asarray(C)
        contrib[i] = C[:, :, iu[0], iu[1]].astype(np.complex64)
        common_all[i] = np.asarray(common)[:, iu[0], iu[1]].astype(np.complex64)
        phys_all[i] = np.asarray(phys)
        absc_all[i], sg_all[i], res_all[i] = float(absc), float(sg), float(resid)
        margin_all[i] = float(margin)
        if i % 32 == 0:
            print(f"{i}/{args.samples} {time.time()-t0:.0f}s absc={absc_all[i]:.1f} sg={sg_all[i]:.2f} "
                  f"res={res_all[i]:.1e} margin={margin_all[i]:.2f}", flush=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, contrib_upper=contrib, common_upper=common_all,
             common_origin_mm=origin, common_speed_mm_per_ms=args.common_speed, physical=phys_all, unit=u, node_abscissa_per_s=absc_all,
             small_gain=sg_all, fixed_point_residual=res_all, fold_margin=margin_all, frequency_hz=np.arange(cfg.spectral.frequency_min_hz,
                                         cfg.spectral.frequency_max_hz + 0.5, cfg.spectral.frequency_bin_hz),
             network_names=np.asarray(static.network_names), group_names=np.asarray(static.group_names), lead=args.lead,
             global_names=np.asarray(LJ.GLOBAL_NAMES))
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
