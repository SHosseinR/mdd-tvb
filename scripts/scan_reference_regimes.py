"""Small, non-fitting scan used to locate stable reference JR regimes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from mdd_tvb.config import load_config
from mdd_tvb.connectome import load_connectome
from mdd_tvb.simulation import run_baseline


def peak_and_alpha_ratio(data: np.ndarray, sfreq: float) -> tuple[float, float]:
    frequencies, power = welch(
        data, fs=sfreq, axis=0, nperseg=min(data.shape[0], int(2 * sfreq))
    )
    mean_power = power.mean(axis=1)
    analysis = (frequencies >= 1.0) & (frequencies <= 45.0)
    alpha = (frequencies >= 8.0) & (frequencies < 13.0)
    peak = float(frequencies[analysis][np.argmax(mean_power[analysis])])
    ratio = float(np.trapezoid(mean_power[alpha], frequencies[alpha]) / np.trapezoid(
        mean_power[analysis], frequencies[analysis]
    ))
    return peak, ratio


def compact_realism_metrics(
    data: np.ndarray,
    sfreq: float,
    channel_names: tuple[str, ...],
) -> tuple[float, float, float]:
    frequencies, power = welch(
        data, fs=sfreq, axis=0, nperseg=min(data.shape[0], int(2 * sfreq))
    )
    mean_power = power.mean(axis=1)
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    spectrum = mean_power[mask]
    probability = spectrum / spectrum.sum()
    entropy = float(
        -(probability * np.log(probability + np.finfo(float).tiny)).sum()
        / np.log(probability.size)
    )
    centered = data - data.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    pc5 = float(np.sum(singular[:5] ** 2) / np.sum(singular**2))
    name_to_index = {name: index for index, name in enumerate(channel_names)}
    posterior = [name_to_index[name] for name in ("P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")]
    anterior = [name_to_index[name] for name in ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")]
    alpha = (frequencies >= 8.0) & (frequencies < 13.0)
    alpha_power = np.trapezoid(power[alpha], frequencies[alpha], axis=0)
    posterior_ratio = float(alpha_power[posterior].mean() / alpha_power[anterior].mean())
    return entropy, pc5, posterior_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    parser.add_argument("--gains", type=float, nargs="+", default=[0, 2, 5, 10, 15])
    parser.add_argument("--mus", type=float, nargs="+", default=[0.18, 0.22])
    parser.add_argument(
        "--time-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Common multiplier applied to JR a and b inverse time constants.",
    )
    parser.add_argument("--duration-ms", type=float, default=2500.0)
    parser.add_argument("--transient-ms", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=None)
    parser.add_argument("--noise-nsigs", type=float, nargs="+", default=None)
    parser.add_argument("--noise-taus", type=float, nargs="+", default=None)
    parser.add_argument(
        "--visual-noise-multipliers", type=float, nargs="+", default=None
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_regime_scan.csv"))
    args = parser.parse_args()

    base = load_config(args.config)
    base = replace(
        base,
        simulation=replace(
            base.simulation,
            duration_ms=args.duration_ms,
            transient_ms=args.transient_ms,
            dt_ms=base.simulation.dt_ms if args.dt_ms is None else args.dt_ms,
        ),
        monitor=replace(base.monitor, surface_laplacian=False),
    )
    connectome = load_connectome(base.paths, base.connectivity)
    rows: list[dict[str, float]] = []
    sfreq = 1000.0 / base.simulation.monitor_period_ms
    noise_nsigs = args.noise_nsigs or [base.simulation.noise_nsig]
    noise_taus = args.noise_taus or [base.simulation.noise_tau_ms]
    visual_noise_multipliers = args.visual_noise_multipliers or [
        base.heterogeneity.visual_noise_multiplier
    ]
    for visual_noise_multiplier in visual_noise_multipliers:
        for noise_nsig in noise_nsigs:
            for noise_tau in noise_taus:
                for time_scale in args.time_scales:
                    for mu in args.mus:
                        for gain in args.gains:
                            model = replace(
                                base.model,
                                mu=mu,
                                a=base.model.a * time_scale,
                                b=base.model.b * time_scale,
                            )
                            config = replace(
                                base,
                                model=model,
                                coupling=replace(base.coupling, global_gain=gain),
                                simulation=replace(
                                    base.simulation,
                                    noise_nsig=noise_nsig,
                                    noise_tau_ms=noise_tau,
                                ),
                                heterogeneity=replace(
                                    base.heterogeneity,
                                    visual_noise_multiplier=visual_noise_multiplier,
                                ),
                            )
                            result = run_baseline(config, connectome)
                            region_peak, region_alpha = peak_and_alpha_ratio(result.region_psp, sfreq)
                            eeg_peak, eeg_alpha = peak_and_alpha_ratio(result.eeg, sfreq)
                            eeg_entropy, eeg_pc5, posterior_ratio = compact_realism_metrics(
                                result.eeg, sfreq, result.channel_names
                            )
                            row = {
                                "noise_nsig": noise_nsig,
                                "noise_tau_ms": noise_tau,
                                "visual_noise_multiplier": visual_noise_multiplier,
                                "time_scale": time_scale,
                                "a": model.a,
                                "b": model.b,
                                "mu": mu,
                                "global_gain": gain,
                                "region_peak_hz": region_peak,
                                "region_alpha_power_fraction_1_45": region_alpha,
                                "eeg_peak_hz": eeg_peak,
                                "eeg_alpha_power_fraction_1_45": eeg_alpha,
                                "eeg_spectral_entropy_1_45": eeg_entropy,
                                "eeg_pc1_to_pc5_variance_fraction": eeg_pc5,
                                "posterior_to_anterior_alpha_power_ratio": posterior_ratio,
                                "region_std": result.metadata["region_psp_standard_deviation"],
                                "eeg_std": result.metadata["eeg_standard_deviation"],
                            }
                            rows.append(row)
                            print(row, flush=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
