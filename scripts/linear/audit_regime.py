"""Classify M5.1 bank candidates as fixed-point or limit-cycle regimes.

Runs the repository's JAX simulator with the neural noise switched off and
measures whether the 26-channel EEG decays (stable fixed point) or keeps a
self-sustained oscillation (limit cycle).  The deterministic 2-40 Hz variance
is compared with the stochastic bank variance of the same candidate.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.linear_spectral import CandidateLinearizer  # noqa: E402
from mdd_tvb.jax_backend import JaxDualBatch, run_dual_jansen_rit_jax  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-npy", default="outputs/audit_claude/m51bank_csd_replicates.npy")
    ap.add_argument("--params-npy", default="outputs/audit_claude/m51bank_parameters.npy")
    ap.add_argument("--posteriors", default="outputs/m5_spectral_m51_production/fit/subject_posteriors.csv")
    ap.add_argument("--random", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="outputs/audit_claude/regime_audit.csv")
    args = ap.parse_args()
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    lin = CandidateLinearizer(cfg)
    P = np.load(ROOT / args.params_npy)
    C = np.load(ROOT / args.bank_npy, mmap_mode="r")
    post = pd.read_csv(ROOT / args.posteriors)
    counts = post.map_candidate_index.value_counts()
    rng = np.random.default_rng(11)
    random = rng.choice(len(P), args.random, replace=False)
    candidates = list(dict.fromkeys(counts.index.tolist() + random.tolist()))
    rows = []
    t0 = time.time()
    for start in range(0, len(candidates), args.batch):
        idx = candidates[start : start + args.batch]
        ps = [lin.parameters(P[k]) for k in idx]
        batch = JaxDualBatch(
            weights=np.stack([p.weights for p in ps]),
            tract_lengths_ms_at_unit_speed=lin.connectome.tract_lengths,
            gain_matrix=lin.gain,
            regional_a=np.stack([p.a for p in ps]), regional_b=np.stack([p.b for p in ps]),
            regional_mu=np.stack([p.mu for p in ps]),
            regional_noise_nsig=np.zeros((len(ps), 200)),
            global_coupling=np.array([p.global_coupling for p in ps]),
            speed_mm_per_ms=P[idx, 1], fast_ratio=np.array([p.fast_ratio for p in ps]),
            fast_fraction=np.array([p.fast_fraction for p in ps]),
            noise_tau_ms=np.array([p.noise_tau_ms for p in ps]),
            seeds=np.arange(len(ps), dtype=np.int32))
        res = run_dual_jansen_rit_jax(batch, duration_ms=6000.0, transient_ms=0.0, dt_ms=0.5,
                                      monitor_period_ms=2.0, history_steps=int(250 / 2 / 0.5) + 2,
                                      precision="float64")
        for j, k in enumerate(idx):
            x = res.eeg[j]
            early = x[1000:1500].std(0).mean()
            late = x[-500:].std(0).mean()
            seg = x[-1500:] - x[-1500:].mean(0)
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], axis=0)) ** 2
            f = np.fft.rfftfreq(len(seg), 0.002)
            band = (f >= 1.5) & (f <= 40.5)
            det_var = (spec[band].sum(0) * 2 / (500 * np.sum(np.hanning(len(seg)) ** 2))).mean()
            sto_var = np.real(np.einsum("fii->fi", np.asarray(C[k]).mean(0))).sum(0).mean()
            rows.append({
                "candidate_index": int(k), "subjects_with_this_map": int(counts.get(k, 0)),
                "std_2_3s": float(early), "std_5_6s": float(late),
                "decay_ratio": float(late / max(early, 1e-300)),
                "deterministic_peak_hz": float(f[1:][np.argmax(spec.mean(1)[1:])]),
                "deterministic_to_stochastic_band_variance": float(det_var / sto_var),
                "regime": "limit_cycle" if late / max(early, 1e-300) > 0.5 and det_var / sto_var > 0.05 else "fixed_point",
            })
        print(f"{min(start + args.batch, len(candidates))}/{len(candidates)} {time.time()-t0:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    map_rows = table[table.subjects_with_this_map > 0]
    summary = {
        "candidates_simulated": len(table),
        "random_candidates_limit_cycle_fraction": float((table[table.candidate_index.isin(random)].regime == "limit_cycle").mean()),
        "subjects_whose_map_is_limit_cycle": int(map_rows.loc[map_rows.regime == "limit_cycle", "subjects_with_this_map"].sum()),
        "subjects_total": int(counts.sum()),
        "median_deterministic_to_stochastic_variance_for_map_states": float(map_rows.deterministic_to_stochastic_band_variance.median()),
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
