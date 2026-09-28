"""Evoked responses of ds003474 predicted from each person's resting-state fit (ds003478).

In the linear regime the network that shapes the resting spectrum also shapes the
response to a brief input: the evoked potential is the impulse response
T(w) U(w) of the same transfer function (linear_jax.network_sensor_transfer; an
input enters a region like the mean drive, alpha and fast generators 1 : 0.85).
Inputs are alpha-function pulses u(t) = ((t - t0)/tau) exp(1 - (t - t0)/tau):
  * stimulus-locked:  visual input to all visual-network regions (amplitude, t0, tau);
  * feedback-locked:  visual input (shared by correct / incorrect feedback) plus a
    medial-frontal input (cingulate / medial prefrontal parcels) with its own
    amplitude for correct and for incorrect feedback (the reward positivity).
Latency and width by grid search, amplitudes by least squares (variable
projection), all on the odd-trial averages in noise-whitened sensor space; the fit
is scored on the even-trial averages (0-600 ms).

Key test: does a person's OWN resting network predict their held-out evoked
responses better than the population network or OTHER people's networks (with
the inputs refitted each time)?  If yes, the resting-state parameters carry
individual dynamics that transfer to a different measurement.
Group test: reward input (correct - incorrect medial-frontal amplitude), high vs low BDI.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m54_core as M  # noqa: E402
from m54_core import jax, jnp, LJ  # noqa: E402

T_WIN, DT = 2.0, 0.01          # FFT window (s) and ERP sampling step (s)
FIT_WINDOW = (0.0, 0.6)
LOWPASS_HZ = 30.0
FAST_INPUT = 0.85              # like the mean drive (linear_jax FAST_DRIVE_RATIO)
V_GRID = [(t0, tau) for t0 in np.arange(0.03, 0.16, 0.01) for tau in (0.01, 0.02, 0.04, 0.07)]
F_GRID = [(t0, tau) for t0 in np.arange(0.10, 0.36, 0.02) for tau in (0.02, 0.04, 0.07, 0.1)]


def target_sets(static):
    atlas = pd.read_csv(M.ROOT / "data/atlas/Schaefer2018_200Parcels_7Networks_centroids.csv").set_index("ROI Name")
    names = [str(n) for n in np.asarray(LJ_labels(static))]
    xyz = atlas.loc[names, ["R", "A", "S"]].to_numpy(float)
    vis = np.asarray(["_Vis_" in n for n in names])
    medfront = (np.abs(xyz[:, 0]) <= 16) & (xyz[:, 1] >= -5) & (xyz[:, 1] <= 45) & (xyz[:, 2] >= 15) & (xyz[:, 2] <= 60)
    return vis.astype(float), medfront.astype(float), names, xyz


def LJ_labels(static):
    from mdd_tvb.spectral_config import load_spectral_m5_config
    from mdd_tvb.linear_spectral import CandidateLinearizer
    lin = CandidateLinearizer(load_spectral_m5_config(M.ROOT / "configs/m5_spectral.toml"))
    return lin.connectome.region_labels


def alpha_pulse_spectrum(omega_s, t0, tau):
    """Fourier transform of ((t - t0)/tau) exp(1 - (t - t0)/tau) H(t - t0); omega in rad/s."""
    return np.e * tau / (1.0 + 1j * omega_s * tau) ** 2 * np.exp(-1j * omega_s * t0)


def whitener(cov, keep_rank):
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:keep_rank]
    w, V = w[order], V[:, order]
    w = np.maximum(w, w[0] * 1e-4)
    return (V / np.sqrt(w)).T  # (rank, 26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest-fits", default="outputs/m54_ds003478/results/ds003478_bem_m10pop/subject_fits.csv")
    ap.add_argument("--population", default="outputs/m54_ds003478/results/population_bem_m10.json")
    ap.add_argument("--erps", default="outputs/erp/ds003474/erps.npz")
    ap.add_argument("--qc", default="outputs/erp/ds003474/qc.csv")
    ap.add_argument("--lead", default="bem")
    ap.add_argument("--others", type=int, default=3, help="other people's networks per subject")
    ap.add_argument("--min-trials", type=int, default=16, help="per feedback condition and half")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--out", default="outputs/m54_evoked")
    args = ap.parse_args()
    out = M.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    M.configure()
    setup = M.make_setup(args.lead, json.loads((M.ROOT / args.population).read_text()), use_pop=True)
    k = len(M.FREE_INDEX)
    if args.lead == "bem":
        from mdd_tvb.template_bem import load_template_bem_gain
        gain, _, _ = load_template_bem_gain(M.ROOT / "data/forward/template_bem_corrected/template_bem_schaefer200_gain.npz")
    else:
        from mdd_tvb.spectral_config import load_spectral_m5_config
        from mdd_tvb.linear_spectral import CandidateLinearizer
        gain = CandidateLinearizer(load_spectral_m5_config(M.ROOT / "configs/m5_spectral.toml")).gain
    gain = np.asarray(gain, float)
    vis, med, names, _ = target_sets(setup.static)
    print(f"visual targets {int(vis.sum())}, medial-frontal targets {int(med.sum())}: "
          f"{[names[i] for i in np.flatnonzero(med)]}", flush=True)
    fits = pd.read_csv(M.ROOT / args.rest_fits).set_index("subject_id")
    E = np.load(M.ROOT / args.erps)
    ids = E["subject_ids"].astype(str)
    conds = list(E["conditions"].astype(str))
    times = E["times"]
    counts = E["counts"]
    use = [i for i, s in enumerate(ids) if s in fits.index
           and counts[i, conds.index("fb_correct"), 1:].min() >= args.min_trials
           and counts[i, conds.index("fb_incorrect"), 1:].min() >= args.min_trials]
    if args.max_subjects:
        use = use[: args.max_subjects]
    print(f"subjects with rest fit and enough trials: {len(use)}", flush=True)
    freqs = np.arange(0.0, 0.5 / DT + 1e-9, 1.0 / T_WIN)          # 0 .. 50 Hz, 0.5 Hz steps
    omega_s = 2 * np.pi * freqs
    lp = 1.0 / (1.0 + (freqs / LOWPASS_HZ) ** 8)                  # |H|^2 of a 4th-order Butterworth (filtfilt)
    n_t = int(round(T_WIN / DT))
    tsel = (times >= FIT_WINDOW[0] - 1e-9) & (times <= FIT_WINDOW[1] + 1e-9)
    t_idx = np.round(times[tsel] / DT).astype(int)                 # model samples at t >= 0
    R = np.eye(26) - 1.0 / 26
    static = setup.static
    delays_cache = {}

    @jax.jit
    def transfer_for(u10, lead_obs):
        p, speed = LJ.build_state(LJ.unit_to_physical(u10), static)
        T, _ = LJ.network_sensor_transfer(p, lead_obs, LJ.delays_for_speed(static, speed), jnp.asarray(freqs))
        return T[..., 0] + FAST_INPUT * T[..., 1]                   # (F, C, n)

    def unit_of(sid):
        row = fits.loc[sid]
        return np.asarray(M.full_u(setup, jnp.asarray([row[n] for n in M.FREE_GLOBALS])))

    def basis(Tn, weights, grid):
        """Sensor waveforms (len(grid), 26, n_t_fit) for unit-amplitude input into the weighted regions."""
        tr = np.einsum("fcn,n->fc", Tn, weights / weights.sum())
        out = []
        for t0, tau in grid:
            spec = tr * (alpha_pulse_spectrum(omega_s, t0, tau) * lp)[:, None]
            wave = np.fft.irfft(spec, n=n_t, axis=0) / DT          # continuous-time inverse transform
            out.append(wave[t_idx].T)
        return np.stack(out)

    def fit_condition_set(Tn, Wh, data_odd, data_even, which):
        """Best grid inputs on the odd averages; returns (sse_even, ss_even, params)."""
        if which == "stim":
            Bv = basis(Tn, vis, V_GRID)
            best = None
            for g, b in enumerate(Bv):
                X = (Wh @ b).ravel()[:, None]
                y = (Wh @ data_odd[0]).ravel()
                a, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = np.sum((y - X @ a) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, g, a)
            _, g, a = best
            pred = a[0] * Bv[g]
            ye = Wh @ data_even[0]
            return float(np.sum((ye - Wh @ pred) ** 2)), float(np.sum(ye ** 2)), \
                {"stim_amp": float(a[0]), "stim_t0": V_GRID[g][0], "stim_tau": V_GRID[g][1]}
        # feedback: shared visual input + condition-specific medial-frontal amplitudes
        Bv, Bf = basis(Tn, vis, V_GRID), basis(Tn, med, F_GRID)
        y = np.concatenate([(Wh @ d).ravel() for d in data_odd])
        ye = np.concatenate([(Wh @ d).ravel() for d in data_even])
        best = None
        for gv, bv in enumerate(Bv):
            v = (Wh @ bv).ravel()
            for gf, bf in enumerate(Bf):
                f = (Wh @ bf).ravel()
                z = np.zeros_like(f)
                X = np.column_stack([np.concatenate([v, v]), np.concatenate([f, z]), np.concatenate([z, f])])
                a, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = np.sum((y - X @ a) ** 2)
                if best is None or sse < best[0]:
                    best = (sse, gv, gf, a, X)
        _, gv, gf, a, X = best
        return float(np.sum((ye - X @ a) ** 2)), float(np.sum(ye ** 2)), \
            {"fb_visual_amp": float(a[0]), "fb_frontal_amp_correct": float(a[1]), "fb_frontal_amp_incorrect": float(a[2]),
             "fb_visual_t0": V_GRID[gv][0], "fb_visual_tau": V_GRID[gv][1], "fb_frontal_t0": F_GRID[gf][0],
             "fb_frontal_tau": F_GRID[gf][1]}

    rng = np.random.default_rng(7)
    fit_ids = [ids[i] for i in use]
    rows, t0 = [], time.time()
    for j, i in enumerate(use):
        sid = ids[i]
        P = E["repair"][i]
        lead_obs = jnp.asarray(R @ P @ gain)
        cov = E["noise_cov"][i]
        Wh = whitener(R @ cov @ R, 25)
        erp = E["erp"][i][:, :, :, tsel]                          # (cond, all/odd/even, 26, T)
        sets = {"stim": ([erp[conds.index("stim"), 1]], [erp[conds.index("stim"), 2]]),
                "feedback": ([erp[conds.index("fb_correct"), 1], erp[conds.index("fb_incorrect"), 1]],
                             [erp[conds.index("fb_correct"), 2], erp[conds.index("fb_incorrect"), 2]])}
        others = [s for s in rng.choice([s for s in fit_ids if s != sid], args.others, replace=False)]
        sources = [("own", unit_of(sid)), ("population", np.asarray(setup.u_population))] + \
                  [(f"other{q}", unit_of(s)) for q, s in enumerate(others)]
        row = {"subject_id": sid, "group": str(E["groups"][i])}
        for label, u in sources:
            Tn = np.asarray(transfer_for(jnp.asarray(u), lead_obs), np.complex128)
            for which, (odd, even) in sets.items():
                sse, ss, par = fit_condition_set(Tn, Wh, odd, even, which)
                row[f"{which}_r2_{label}"] = 1.0 - sse / ss
                if label == "own":
                    row.update(par)
        for which, (odd, even) in sets.items():  # noise ceiling: the odd average itself predicting the even one
            yo = np.concatenate([(Wh @ d).ravel() for d in odd]); ye = np.concatenate([(Wh @ d).ravel() for d in even])
            row[f"{which}_r2_odd_as_prediction"] = 1.0 - np.sum((ye - yo) ** 2) / np.sum(ye ** 2)
            row[f"{which}_r2_other_mean"] = float(np.mean([row[f"{which}_r2_other{q}"] for q in range(args.others)]))
        rows.append(row)
        if (j + 1) % 5 == 0:
            t = pd.DataFrame(rows)
            print(f"{j+1}/{len(use)} {time.time()-t0:.0f}s | feedback own {t.feedback_r2_own.median():.3f} "
                  f"other {t.feedback_r2_other_mean.median():.3f} pop {t.feedback_r2_population.median():.3f}; "
                  f"stim own {t.stim_r2_own.median():.3f} other {t.stim_r2_other_mean.median():.3f}", flush=True)
    t = pd.DataFrame(rows)
    t["reward_input"] = t.fb_frontal_amp_correct - t.fb_frontal_amp_incorrect
    t.to_csv(out / "evoked_fits.csv", index=False)
    import scipy.stats as st
    summ = {"n": int(len(t))}
    for which in ("stim", "feedback"):
        d_other = t[f"{which}_r2_own"] - t[f"{which}_r2_other_mean"]
        d_pop = t[f"{which}_r2_own"] - t[f"{which}_r2_population"]
        summ[which] = {"median_r2_own": float(t[f"{which}_r2_own"].median()),
                       "median_r2_population": float(t[f"{which}_r2_population"].median()),
                       "median_r2_other": float(t[f"{which}_r2_other_mean"].median()),
                       "median_r2_ceiling_odd_as_prediction": float(t[f"{which}_r2_odd_as_prediction"].median()),
                       "own_beats_other_fraction": float((d_other > 0).mean()),
                       "own_minus_other_median": float(d_other.median()),
                       "wilcoxon_p_own_vs_other": float(st.wilcoxon(d_other).pvalue),
                       "own_beats_population_fraction": float((d_pop > 0).mean()),
                       "wilcoxon_p_own_vs_population": float(st.wilcoxon(d_pop).pvalue)}
    hi, lo = t.loc[t.group == "HighBDI", "reward_input"], t.loc[t.group == "LowBDI", "reward_input"]
    summ["reward_input"] = {"high_bdi_mean": float(hi.mean()), "low_bdi_mean": float(lo.mean()),
                            "cohen_d": float((hi.mean() - lo.mean()) / np.sqrt((hi.var() + lo.var()) / 2)),
                            "p": float(st.ttest_ind(hi, lo).pvalue), "n_high": int(len(hi)), "n_low": int(len(lo)),
                            "mean_all": float(t.reward_input.mean()), "t_vs_zero": float(st.ttest_1samp(t.reward_input, 0).statistic)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
