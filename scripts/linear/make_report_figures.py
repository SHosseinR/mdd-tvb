"""Figures for docs/INDEPENDENT_AUDIT_2026-09-24.md (reads saved outputs only)."""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402

OUT = ROOT / "docs/figures/audit_2026-09-24"
CH = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T7", "C3", "Cz",
      "C4", "T8", "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2")


def lagged(csd):
    d = np.sqrt(np.real(np.einsum("...ii->...i", csd)))
    coh = csd / (d[..., :, None] * d[..., None, :])
    return np.imag(coh) / np.sqrt(np.maximum(1 - np.real(coh) ** 2, 1e-8)), np.real(coh)


def fig_regime() -> None:
    det = ROOT / "outputs/audit_claude/det_eeg.npy"
    table = ROOT / "outputs/audit_claude/regime_audit.csv"
    if not det.is_file():
        return
    eeg = np.load(det)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    t = np.arange(eeg.shape[1]) * 2.0 / 1000
    for j, (idx, label) in enumerate(((2, "M5.1 candidate 1535 (most-selected MAP): noise OFF"),
                                      (4, "M5.1 candidate 500 (fixed-point corner): noise OFF"))):
        axes[j].plot(t, eeg[idx][:, 23] - eeg[idx][:, 23].mean(), lw=0.6)
        axes[j].set_title(label)
        axes[j].set_xlabel("time (s)")
        axes[j].set_ylabel("O1 (model units)")
    fig.suptitle("Without any noise most fitted M5.1 states keep oscillating: a deterministic limit cycle")
    fig.savefig(OUT / "regime_noise_off_traces.png", dpi=140)
    plt.close(fig)
    if table.is_file():
        tab = pd.read_csv(table)
        fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
        colors = np.where(tab.regime == "limit_cycle", "C3", "C0")
        ax.scatter(tab.decay_ratio.clip(1e-6, 10), tab.deterministic_to_stochastic_band_variance.clip(1e-8, 10),
                   c=colors, s=12 + 3 * tab.subjects_with_this_map, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("noise-free EEG std (5-6 s) / (2-3 s)")
        ax.set_ylabel("noise-free / stochastic 2-40 Hz variance")
        ax.set_title("M5.1 candidates (size = subjects whose MAP it is)\nred = self-sustained limit cycle")
        fig.savefig(OUT / "regime_classification.png", dpi=140)
        plt.close(fig)


def fig_lagged_pattern() -> None:
    fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
    dev = np.isin(fit.subject_ids.astype(str), pd.read_csv(ROOT / "configs/m52_nested_splits.csv").subject_id.astype(str))
    band = (fit.frequency_hz >= 8) & (fit.frequency_hz < 13)
    emp, _ = lagged(fit.csd[dev][:, band].mean(1))
    emp = emp.mean(0)
    panels = [("Empirical group mean (262 dev subjects, first halves)", emp)]
    base = []
    for k in range(5):
        split = pd.read_csv(ROOT / f"outputs/m52_nested_baseline/outer_{k}/data_split.csv")
        with np.load(ROOT / f"outputs/m52_nested_baseline/outer_{k}/posterior_predictive_csd.npz") as p:
            base.append(p["csd"][split.subject_split.values == "holdout"][:, band].mean(1))
    model, _ = lagged(np.concatenate(base))
    panels.append(("M5.1 bank posterior (out-of-fold predictions, mean)", model.mean(0)))
    for name, path in (("Final model (TVB lead, refined; out-of-fold mean)",
                        ROOT / "outputs/linear_regime/kaggle/final/final_tvb_refined/predictions.npz"),):
        if path.is_file():
            csd = np.load(path)["csd"]
            model, _ = lagged(csd[:, band].mean(1))
            panels.append((name, model.mean(0)))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), constrained_layout=True)
    vmax = np.abs(emp).max()
    iu = np.triu_indices(26, 1)
    for ax, (name, mat) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        r = np.corrcoef(mat[iu], emp[iu])[0, 1] if mat is not emp else 1.0
        ax.set_title(f"{name}\nr with empirical = {r:.2f}, mean|lag| = {np.abs(mat[iu]).mean():.3f}", fontsize=9)
        ax.set_xticks(range(26)); ax.set_xticklabels(CH, rotation=90, fontsize=6)
        ax.set_yticks(range(26)); ax.set_yticklabels(CH, fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.7, label="alpha (8-13 Hz) lagged coherency")
    fig.savefig(OUT / "lagged_coherency_patterns.png", dpi=140)
    plt.close(fig)


def fig_volume_conduction() -> None:
    audit = ROOT / "outputs/audit_claude/failure_mode_audit.json"
    if not audit.is_file():
        return
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
        list(CH), ["x_mm", "y_mm", "z_mm"]].to_numpy()
    iu = np.triu_indices(26, 1)
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)[iu]
    fit = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz")
    band = (fit.frequency_hz >= 13) & (fit.frequency_hz < 31)
    _, re = lagged(fit.csd[:, band].mean(1))
    emp = re.mean(0)[iu]
    avg = np.eye(26) - 1 / 26
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.scatter(dist, emp, s=8, c="k", label="empirical (beta, 327 subjects)")
    for name, gain, color in (("TVB analytic gain", np.asarray(mon.gain), "C3"), ("fsaverage BEM", bem, "C0")):
        L = avg @ gain
        c = L @ L.T
        d = np.sqrt(np.diag(c))
        c = (c / np.outer(d, d))[iu]
        ax.scatter(dist, c, s=8, c=color, alpha=0.6,
                   label=f"{name}: independent equal sources (r={np.corrcoef(c, emp)[0, 1]:.2f})")
    ax.set_xlabel("electrode distance (mm)")
    ax.set_ylabel("zero-lag coherence (average reference)")
    ax.legend(fontsize=8)
    ax.set_title("Volume conduction implied by each lead field vs empirical EEG")
    fig.savefig(OUT / "lead_field_volume_conduction.png", dpi=140)
    plt.close(fig)


def fig_spectra() -> None:
    val = load_cross_spectral_collection(ROOT / "outputs/m5_spectral_m51_production/empirical/cross_spectra_validation.npz")
    f = val.frequency_hz
    ids = val.subject_ids.astype(str)
    runs = {"M5.1 nested posterior": None,
            "new (analytic, linear regime)": ROOT / "outputs/linear_regime/kaggle/final/final_tvb_refined/predictions.npz"}
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    emp = np.log(np.real(np.einsum("sfii->sf", val.csd)))
    emp -= emp.mean(1, keepdims=True)
    ax.plot(f, emp.mean(0), "k", lw=2, label="empirical unseen halves")
    base = []
    for k in range(5):
        with np.load(ROOT / f"outputs/m52_nested_baseline/outer_{k}/posterior_predictive_csd.npz") as p:
            split = pd.read_csv(ROOT / f"outputs/m52_nested_baseline/outer_{k}/data_split.csv")
            hold = split.subject_split.values == "holdout"
            base.append(np.log(np.real(np.einsum("sfii->sf", p["csd"][hold]))))
    base = np.concatenate(base)
    base -= base.mean(1, keepdims=True)
    ax.plot(f, base.mean(0), "C3", label="M5.1 bank posterior (out-of-fold)")
    path = runs["new (analytic, linear regime)"]
    if path.is_file():
        p = np.load(path)
        new = np.log(np.real(np.einsum("sfii->sf", p["csd"])))
        new -= new.mean(1, keepdims=True)
        ax.plot(f, new.mean(0), "C0", label="final model, TVB lead, refined (out-of-fold)")
    ax.set_xlabel("Hz")
    ax.set_ylabel("centred log channel-summed power")
    ax.legend(fontsize=8)
    ax.set_title("Group-mean spectra, development subjects")
    fig.savefig(OUT / "group_mean_spectra.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_regime, fig_lagged_pattern, fig_volume_conduction, fig_spectra):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as error:  # keep other figures
            print("failed", fn.__name__, repr(error))
