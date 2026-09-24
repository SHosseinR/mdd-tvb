"""Reproduce the M5.1/M5.2 failure-mode audit (no new simulations needed).

Outputs ``outputs/audit_claude/failure_mode_audit.json`` with:

1. per-block noise ceilings (same-subject temporal persistence) vs the model;
2. replicability of Healthy-MDD group effects across subject samples;
3. posterior observation-noise fractions (how much of the signal is nuisance);
4. which frequencies the reliability filter kept in the objective;
5. empirical vs simulated zero-lag and lagged coherency magnitudes;
6. lead-field-implied volume conduction vs empirical zero-lag coherence;
7. EMG-like high-frequency spectral slopes per channel;
8. age/sex confounding of the group effects.

Only first/second halves already extracted by M5.1 are used; the bank array
is read through a memory map (``--bank-npy``) when available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdd_tvb.spectral_features import (  # noqa: E402
    load_cross_spectral_collection,
    load_spectral_transformer,
)
from mdd_tvb.spectral_fit import (  # noqa: E402
    _channel_log_power,
    _connectivity_metric,
    _group_difference,
    _safe_correlation,
)

CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
    "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
)


def recoh(csd: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.real(np.einsum("...ii->...i", csd)))
    return np.real(csd) / (d[..., :, None] * d[..., None, :])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", type=Path, default=ROOT / "outputs/m5_spectral_m51_production")
    ap.add_argument("--bank-npy", type=Path, default=ROOT / "outputs/audit_claude/m51bank_csd_replicates.npy")
    ap.add_argument("--participants", type=Path, default=Path(
        "D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1/TDBRAIN_participants_V3.xlsx"))
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/audit_claude/failure_mode_audit.json")
    args = ap.parse_args()
    emp = args.production / "empirical"
    fitdir = args.production / "fit"
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    tr = load_spectral_transformer(fitdir / "spectral_transformer.npz")
    split = pd.read_csv(fitdir / "data_split.csv")
    hold = np.flatnonzero(split.subject_split.values == "holdout")
    train = np.flatnonzero(split.subject_split.values == "train")
    pred = np.load(fitdir / "posterior_predictive_csd.npz")["csd"]
    post = pd.read_csv(fitdir / "subject_posteriors.csv")
    freq = fit.frequency_hz
    report: dict = {}

    # 1. noise ceilings
    fb, vb, pb = tr.transform_blocks(fit.csd), tr.transform_blocks(val.csd), tr.transform_blocks(pred)
    ceilings = {}
    for k, name in enumerate(("autospectrum", "lagged_coherency", "alpha_topography")):
        pooled = fb[k][train].mean(0)
        null = np.mean((pooled - vb[k]) ** 2, 1)
        x = fb[k] - fb[k].mean(0)
        y = vb[k] - vb[k].mean(0)
        rel = (x * y).sum(0) / np.sqrt((x**2).sum(0) * (y**2).sum(0))
        ceilings[name] = {
            "holdout_median_persistence_ratio": float(np.median((np.mean((fb[k] - vb[k]) ** 2, 1) / null)[hold])),
            "holdout_median_model_ratio": float(np.median((np.mean((pb[k] - vb[k]) ** 2, 1) / null)[hold])),
            "median_coordinate_test_retest_r": float(np.median(rel)),
        }
    report["noise_ceilings"] = ceilings

    # 2. group-effect replicability
    g = val.groups
    rng = np.random.default_rng(0)
    alpha = (freq >= 8) & (freq < 13)

    def atopo(c):
        p = _channel_log_power(c)[:, alpha].mean(1)
        return p - p.mean(1, keepdims=True)

    rep = {}
    for name, vv, ff in (
        ("channel_frequency_power", _channel_log_power(val.csd), _channel_log_power(fit.csd)),
        ("alpha_topography", atopo(val.csd), atopo(fit.csd)),
        ("lagged_coherency", _connectivity_metric(val.csd, "lagged_coherency"),
         _connectivity_metric(fit.csd, "lagged_coherency")),
    ):
        def eff(values, idx):
            return _group_difference(values[idx], g[idx], "Healthy", "MDD")
        splits = []
        for _ in range(200):
            idx = rng.permutation(len(g))
            splits.append(_safe_correlation(eff(vv, idx[:65]), eff(vv, idx[65:130])))
        rep[name] = {
            "holdout_first_half_vs_unseen_r": _safe_correlation(eff(vv, hold), eff(ff, hold)),
            "development_vs_holdout_effect_r": _safe_correlation(eff(vv, train), eff(vv, hold)),
            "random_65_vs_65_subject_split_median_r": float(np.median(splits)),
        }
    report["group_effect_replicability"] = rep

    # 3. observation-noise fraction
    frac = post["observation_noise_fraction_posterior_mean"]
    report["posterior_observation_noise_fraction"] = {
        "median": float(frac.median()), "q25": float(frac.quantile(0.25)),
        "q75": float(frac.quantile(0.75)), "fraction_above_0p75": float((frac > 0.75).mean()),
    }

    # 4. frequencies kept by the reliability filter
    m = tr.sensor_basis.shape[1]
    npairs = m * (m - 1) // 2
    cross_sel = np.argmax(tr.cross_components, 1)
    auto_sel = np.argmax(tr.auto_components, 1)
    nres = freq.size * m
    report["objective_coordinates"] = {
        "lagged_coherency_frequencies_hz": sorted(float(freq[s // npairs]) for s in cross_sel),
        "autospectrum_frequencies_hz": sorted(float(freq[s // m]) for s in auto_sel if s < nres),
        "autospectrum_aperiodic_exponents": int(np.sum(auto_sel >= nres)),
        "alpha_topography_channels": [CHANNELS[i] for i in np.argmax(tr.topography_components, 1)],
    }

    # 5. coherency magnitudes
    bands = {"theta": (freq >= 4) & (freq < 8), "alpha": alpha, "beta": (freq >= 13) & (freq < 31)}
    lag_e = np.abs(_connectivity_metric(fit.csd, "lagged_coherency"))
    re_e = np.abs(recoh(fit.csd)[..., *np.triu_indices(26, 1)])
    coh = {"empirical": {b: {"lagged": float(lag_e[:, s].mean()), "zero_lag": float(re_e[:, s].mean())}
                         for b, s in bands.items()}}
    if args.bank_npy.is_file():
        bank = np.load(args.bank_npy, mmap_mode="r")
        ks = np.random.default_rng(0).choice(bank.shape[0], 40, replace=False)
        sim = np.stack([np.asarray(bank[k, 0]) for k in ks])
        lag_s = np.abs(_connectivity_metric(sim, "lagged_coherency"))
        re_s = np.abs(recoh(sim)[..., *np.triu_indices(26, 1)])
        coh["m51_bank_40_random_candidates"] = {
            b: {"lagged": float(lag_s[:, s].mean()), "zero_lag": float(re_s[:, s].mean())}
            for b, s in bands.items()}
    report["coherency_magnitudes"] = coh

    # 6. lead-field volume conduction
    from mdd_tvb.config import load_config
    from mdd_tvb.connectome import load_connectome
    from mdd_tvb.eeg import build_eeg_monitor, regularize_analytic_eeg_gain
    from mdd_tvb.template_bem import load_template_bem_gain
    base = load_config(ROOT / "configs/baseline.toml")
    conn = load_connectome(base.paths, base.connectivity)
    mon, _ = build_eeg_monitor(base.monitor, 200, base.simulation.monitor_period_ms)
    mon.configure()
    regularize_analytic_eeg_gain(mon, conn.centres, conn.connectivity.orientations,
                                 base.monitor.minimum_source_sensor_distance_mm)
    bem, _, _ = load_template_bem_gain(ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    xyz = pd.read_csv(ROOT / "data/sensors/TDBRAIN_Table3_electrode_coordinates.csv").set_index("label").loc[
        list(CHANNELS), ["x_mm", "y_mm", "z_mm"]].to_numpy()
    iu = np.triu_indices(26, 1)
    near = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)[iu] < 60
    E = recoh(fit.csd).mean(0)
    avg = np.eye(26) - 1 / 26
    lf = {}
    for name, gain in (("tvb_analytic", np.asarray(mon.gain)), ("fsaverage_bem_corrected", bem)):
        L = avg @ gain
        c = recoh(L @ L.T)[iu]
        lf[name] = {"neighbour_coherence": float(c[near].mean()),
                    **{f"pattern_r_{b}": float(np.corrcoef(c, E[s].mean(0)[iu])[0, 1]) for b, s in bands.items()}}
        p = np.log((L**2).sum(1))
        p -= p.mean()
        lf[name]["equal_source_power_Fp1_Fp2_O1_Oz_O2"] = [float(p[i]) for i in (0, 1, 23, 24, 25)]
    lf["empirical_neighbour_coherence"] = {b: float(E[s].mean(0)[iu][near].mean()) for b, s in bands.items()}
    report["lead_field_volume_conduction"] = lf

    # 7. EMG-like slopes
    lp = np.log(np.real(np.einsum("sfii->sfi", fit.csd)))
    hi = (freq >= 20) & (freq <= 40)
    X = np.c_[np.ones(hi.sum()), np.log(freq[hi])]
    slope = np.linalg.lstsq(X, lp[:, hi, :].transpose(1, 0, 2).reshape(hi.sum(), -1), rcond=None)[0][1]
    slope = slope.reshape(len(lp), 26)
    report["high_frequency_slope_20_40hz"] = {
        c: {"median": float(np.median(slope[:, i])), "fraction_flat_or_rising": float(np.mean(slope[:, i] > -0.5))}
        for i, c in enumerate(CHANNELS)}

    # 8. age/sex confound
    if args.participants.is_file():
        T = pd.read_excel(args.participants)
        T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
        ids = val.subject_ids.astype(str)
        age = T.loc[ids, "age"].to_numpy(float)
        sex = T.loc[ids, "gender"].to_numpy(float)
        grp = (g == "MDD").astype(float)
        conf = {"age_mean": {"Healthy": float(age[grp == 0].mean()), "MDD": float(age[grp == 1].mean())},
                "formal_status_counts_MDD": T.loc[ids[grp == 1], "formal_status"].value_counts().to_dict()}
        for name, Y in (("channel_frequency_power", _channel_log_power(val.csd).reshape(len(ids), -1)),
                        ("alpha_topography", atopo(val.csd))):
            b0 = np.linalg.lstsq(np.c_[np.ones(len(grp)), grp], Y, rcond=None)[0][1]
            b1 = np.linalg.lstsq(np.c_[np.ones(len(grp)), grp, age, sex], Y, rcond=None)[0][1]
            conf[name] = {"raw_vs_adjusted_r": float(np.corrcoef(b0, b1)[0, 1]),
                          "adjusted_norm_over_raw": float(np.linalg.norm(b1) / np.linalg.norm(b0))}
        report["demographic_confounding"] = conf

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
