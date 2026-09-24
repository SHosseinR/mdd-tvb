"""Scan the origin of a shared delayed alpha drive (population level, fit halves only)."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/linear"))
import jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_spectral import (CandidateLinearizer, LinearNetworkParameters,  # noqa: E402
                                     linear_regional_transfer, sensor_csd_from_transfer, solve_fixed_point)
from mdd_tvb import linear_jax as LJ  # noqa: E402
from common_drive_experiment import lagged  # noqa: E402

cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
lin = CandidateLinearizer(cfg)
static = LJ.static_from_linearizer(lin, cfg)
pop = json.loads((ROOT / "outputs/linear_regime/population_fit_tvb.json").read_text())["best"]
state, speed = LJ.build_state(jnp.asarray(list(pop["physical"].values())), static)
beta = np.asarray(list(pop["log_network_noise_gain"].values()))
params = LinearNetworkParameters(
    weights=np.asarray(state["W"]), delays_ms=static.lengths_mm / float(speed), a=np.asarray(state["a"]),
    b=np.asarray(state["b"]), mu=np.asarray(state["mu"]),
    noise_nsig=np.asarray(state["noise_base"]) * np.exp(beta[static.network_index]),
    global_coupling=float(state["G"]), fast_ratio=float(state["fast_ratio"]),
    fast_fraction=float(state["fast_fraction"]), noise_tau_ms=float(state["noise_tau"]))
fine = np.arange(8.0, 13.0, 0.25)
omega = 2 * np.pi * fine / 1000.0
transfer, noise_psd, _ = linear_regional_transfer(params, fine, solve_fixed_point(params))
lead = lin.gain - lin.gain.mean(0, keepdims=True)
alpha_in = np.einsum("cr,frk->fck", lead, transfer)[:, :, 0::2]
private = sensor_csd_from_transfer(lin.gain, transfer, noise_psd)
frac = pop["observation_noise_fraction"]
fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
import pandas as pd
dev = np.isin(fit.subject_ids.astype(str), pd.read_csv(ROOT / "configs/m52_nested_splits.csv").subject_id.astype(str))
band = (fit.frequency_hz >= 8) & (fit.frequency_hz < 13)
emp_lag, emp_re = lagged(fit.csd[dev][:, band].mean(1))
emp_lag = emp_lag.mean(0)
iu = np.triu_indices(26, 1)
centres = lin.connectome.centres
rows = []
for y in (-45.0, -35.0, -25.0, -15.0, -5.0, 5.0):
    for z in (40.0, 50.0, 60.0, 70.0):
        origin = np.array([0.0, y, z])
        dist = np.linalg.norm(centres - origin, axis=1)
        for v_wave in (2.0, 3.0, 4.0, 6.0):
            v = np.einsum("fcr,fr->fc", alpha_in, np.exp(-1j * omega[:, None] * (dist / v_wave)[None]))
            common = np.einsum("fc,fd->fcd", np.conjugate(v), v)
            share = 0.4
            neural = private + common * (share / (1 - share)) * np.real(np.einsum("fcc->", private)) / np.real(np.einsum("fcc->", common))
            csd = neural.mean(0)
            csd = csd + np.eye(26) * np.real(np.trace(csd)) / 26 * frac / (1 - frac)
            mlag, mre = lagged(csd)
            rows.append({"origin_y": y, "origin_z": z, "speed": v_wave,
                         "pattern_r_lagged": float(np.corrcoef(mlag[iu], emp_lag[iu])[0, 1]),
                         "mean_abs_lagged": float(np.abs(mlag[iu]).mean()),
                         "pattern_r_zero_lag": float(np.corrcoef(mre[iu], emp_re.mean(0)[iu])[0, 1])})
rows.sort(key=lambda r: -r["pattern_r_lagged"])
for r in rows[:6] + rows[-3:]:
    print({k: round(v, 3) for k, v in r.items()})
(ROOT / "outputs/linear_regime/common_drive_origin_scan_dev.json").write_text(json.dumps(rows, indent=1))
