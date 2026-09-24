"""Can slow short-range coupling generate the empirical lagged alpha coherency?

Starting from the label-blind population fit, add a distance-dependent local
kernel between parcel centroids (length scale ``lambda``) with slow
(intracortical/U-fibre) conduction and compare the model's group-level alpha
lagged-coherency pattern with the empirical group mean (first halves).
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_spectral import (  # noqa: E402
    CandidateLinearizer, LinearNetworkParameters, linear_regional_transfer,
    sensor_csd_from_transfer, solve_fixed_point, stability_report,
)
from mdd_tvb import linear_jax as LJ  # noqa: E402


def lagged(csd):
    d = np.sqrt(np.real(np.einsum("...ii->...i", csd)))
    coh = csd / (d[..., :, None] * d[..., None, :])
    return np.imag(coh) / np.sqrt(np.maximum(1 - np.real(coh) ** 2, 1e-8)), np.real(coh)


def main() -> None:
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    pop = json.loads((ROOT / "outputs/linear_regime/population_fit_tvb.json").read_text())["best"]
    phys = jnp.asarray(list(pop["physical"].values()))
    state, speed = LJ.build_state(phys, static)
    beta = np.asarray(list(pop["log_network_noise_gain"].values()))
    noise = np.asarray(state["noise_base"]) * np.exp(beta[static.network_index])
    base = LinearNetworkParameters(
        weights=np.asarray(state["W"]), delays_ms=static.lengths_mm / float(speed),
        a=np.asarray(state["a"]), b=np.asarray(state["b"]), mu=np.asarray(state["mu"]),
        noise_nsig=noise, global_coupling=float(state["G"]), fast_ratio=float(state["fast_ratio"]),
        fast_fraction=float(state["fast_fraction"]), noise_tau_ms=float(state["noise_tau"]),
    )
    centres = lin.connectome.centres
    dist = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    hemi = np.char.find(lin.connectome.region_labels.astype(str), "_RH_") >= 0
    same_hemi = hemi[:, None] == hemi[None]
    fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
    freq = fit.frequency_hz
    alpha = (freq >= 8) & (freq < 13)
    emp_lag, emp_re = lagged(fit.csd[:, alpha].mean(1))
    emp_lag = emp_lag.mean(0)
    iu = np.triu_indices(26, 1)
    fine = np.arange(8.0, 13.0, 0.25)
    rows = []
    for lam in (12.0, 20.0):
        kernel = np.exp(-dist / lam) * (dist < 3 * lam) * same_hemi
        np.fill_diagonal(kernel, 0.0)
        kernel /= kernel.sum(1).max()
        for v_local in (0.3, 1.0):
            for g_local in (0.0, 2.0, 5.0, 10.0, 20.0):
                p = LinearNetworkParameters(**{**base.__dict__, "local_weights": kernel,
                                               "local_delays_ms": dist / v_local, "local_coupling": g_local})
                pt = solve_fixed_point(p)
                T, S, _ = linear_regional_transfer(p, fine, pt)
                csd = sensor_csd_from_transfer(lin.gain, T, S).mean(0)
                frac = pop["observation_noise_fraction"]
                csd = csd + np.eye(26) * np.real(np.trace(csd)) / 26 * frac / (1 - frac)
                mlag, mre = lagged(csd)
                st = stability_report(p, pt, frequency_max_hz=60.0, frequency_step_hz=0.05)
                row = {"lambda_mm": lam, "v_local_mm_per_ms": v_local, "g_local": g_local,
                       "pattern_r_lagged": float(np.corrcoef(mlag[iu], emp_lag[iu])[0, 1]),
                       "mean_abs_lagged_model": float(np.abs(mlag[iu]).mean()),
                       "mean_abs_lagged_empirical_group_mean": float(np.abs(emp_lag[iu]).mean()),
                       "pattern_r_zero_lag": float(np.corrcoef(mre[iu], emp_re.mean(0)[iu])[0, 1]),
                       "stable": st["stable"], "node_max_eig": st["node_max_real_eigenvalue_per_s"],
                       "winding": st["network_winding"], "fp_residual": pt.residual}
                rows.append(row)
                print(json.dumps(row), flush=True)
                if g_local == 0.0 and (lam, v_local) != (12.0, 0.3):
                    continue
    out = ROOT / "outputs/linear_regime/travelling_wave_experiment.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
