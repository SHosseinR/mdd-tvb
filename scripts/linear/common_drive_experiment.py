"""Can a shared, delayed (thalamic-like) input explain lagged alpha coherency?

Linear-regime population state + an additional common noise source that
reaches every cortical parcel's excitatory input with a delay proportional to
its distance from a thalamic seed.  Its CSD contribution is rank one per
frequency: S_c(f) v(f) v(f)^H with v = sum_i T_i c_i exp(-i w d_i).
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
    sensor_csd_from_transfer, solve_fixed_point,
)
from mdd_tvb import linear_jax as LJ  # noqa: E402

THALAMUS_MNI = {"LH": np.array([-12.0, -18.0, 8.0]), "RH": np.array([12.0, -18.0, 8.0])}


def lagged(csd):
    d = np.sqrt(np.real(np.einsum("...ii->...i", csd)))
    coh = csd / (d[..., :, None] * d[..., None, :])
    return np.imag(coh) / np.sqrt(np.maximum(1 - np.real(coh) ** 2, 1e-8)), np.real(coh)


def main() -> None:
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    static = LJ.static_from_linearizer(lin, cfg)
    pop = json.loads((ROOT / "outputs/linear_regime/population_fit_tvb.json").read_text())["best"]
    state, speed = LJ.build_state(jnp.asarray(list(pop["physical"].values())), static)
    beta = np.asarray(list(pop["log_network_noise_gain"].values()))
    noise = np.asarray(state["noise_base"]) * np.exp(beta[static.network_index])
    params = LinearNetworkParameters(
        weights=np.asarray(state["W"]), delays_ms=static.lengths_mm / float(speed),
        a=np.asarray(state["a"]), b=np.asarray(state["b"]), mu=np.asarray(state["mu"]),
        noise_nsig=noise, global_coupling=float(state["G"]), fast_ratio=float(state["fast_ratio"]),
        fast_fraction=float(state["fast_fraction"]), noise_tau_ms=float(state["noise_tau"]),
    )
    fine = np.arange(8.0, 13.0, 0.25)
    point = solve_fixed_point(params)
    transfer, noise_psd, _ = linear_regional_transfer(params, fine, point)
    lead = lin.gain - lin.gain.mean(0, keepdims=True)
    sensor_t = np.einsum("cr,frk->fck", lead, transfer)  # (F, C, 2n)
    alpha_in = sensor_t[:, :, 0::2]  # response to each region's alpha-generator input
    private = sensor_csd_from_transfer(lin.gain, transfer, noise_psd)
    frac = pop["observation_noise_fraction"]

    fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
    freq = fit.frequency_hz
    band = (freq >= 8) & (freq < 13)
    emp_lag, emp_re = lagged(fit.csd[:, band].mean(1))
    emp_lag = emp_lag.mean(0)
    iu = np.triu_indices(26, 1)
    labels = lin.connectome.region_labels.astype(str)
    hemi = np.where(np.char.find(labels, "_RH_") >= 0, "RH", "LH")
    centres = lin.connectome.centres
    dist = np.asarray([np.linalg.norm(centres[i] - THALAMUS_MNI[h]) for i, h in enumerate(hemi)])
    omega = 2 * np.pi * fine / 1000.0
    rows = []
    for v_thal in (1.0, 2.0, 5.0):
        delay = dist / v_thal
        for share in (0.1, 0.3, 0.6):
            v = np.einsum("fcr,fr->fc", alpha_in, np.exp(-1j * omega[:, None] * delay[None]))
            common = np.einsum("fc,fd->fcd", np.conjugate(v), v)
            # scale the common term to `share` of the total neural power
            p_priv = np.real(np.einsum("fcc->", private))
            p_comm = np.real(np.einsum("fcc->", common))
            neural = private + common * (share / (1 - share)) * p_priv / p_comm
            csd = neural.mean(0)
            csd = csd + np.eye(26) * np.real(np.trace(csd)) / 26 * frac / (1 - frac)
            mlag, mre = lagged(csd)
            row = {"thalamocortical_speed_mm_per_ms": v_thal, "common_share": share,
                   "pattern_r_lagged": float(np.corrcoef(mlag[iu], emp_lag[iu])[0, 1]),
                   "mean_abs_lagged_model": float(np.abs(mlag[iu]).mean()),
                   "mean_abs_lagged_empirical": float(np.abs(emp_lag[iu]).mean()),
                   "pattern_r_zero_lag": float(np.corrcoef(mre[iu], emp_re.mean(0)[iu])[0, 1]),
                   "mean_zero_lag_model": float(mre[iu].mean()), "mean_zero_lag_empirical": float(emp_re.mean(0)[iu].mean())}
            rows.append(row)
            print(json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in row.items()}), flush=True)
    (ROOT / "outputs/linear_regime/common_drive_experiment.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
