"""Figures for docs/NEXT_STEPS_REPORT_2026-09-25.md (reads saved outputs only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures/next_steps_2026-09-25"
OUT.mkdir(parents=True, exist_ok=True)
BLOCKS = ["spectrum", "topo_theta", "topo_alpha", "topo_beta", "zerolag_theta", "zerolag_alpha", "zerolag_beta",
          "lagged_theta", "lagged_alpha", "lagged_beta", "iaf"]
LABELS = {"spectrum": "log spectrum", "topo_theta": "θ topography", "topo_alpha": "α topography", "topo_beta": "β topography",
          "zerolag_theta": "θ zero-lag coh.", "zerolag_alpha": "α zero-lag coh.", "zerolag_beta": "β zero-lag coh.",
          "lagged_theta": "θ lagged coh.", "lagged_alpha": "α lagged coh.", "lagged_beta": "β lagged coh.", "iaf": "alpha peak freq."}


def medians(path, rename):
    e = json.loads((ROOT / path).read_text())
    return {rename.get(s["label"], s["label"]): s["median_ratio"] for s in e["summaries"] if s["label"] in rename}


def fig_blocks():
    dev = medians("outputs/m54_eval/v2_dev_m54b_bem/evaluation.json",
                  {"persist": "own first half (ceiling)", "M53v2": "M5.3 (frozen)", "M54full_bem": "M5.4 full likelihood",
                   "M54b_bem": "M5.4b, 10 modes", "M54b_bem_pop": "M5.4b final"})
    v1 = medians("outputs/m54_eval/v1_m51_vs_m53/evaluation.json", {"M51": "M5.1 (original data)"})
    order = ["M5.1 (original data)", "M5.3 (frozen)", "M5.4 full likelihood", "M5.4b, 10 modes", "M5.4b final",
             "own first half (ceiling)"]
    table = {**v1, **dev}
    M = np.array([[table[m][b] for b in BLOCKS] for m in order])
    fig, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    im = ax.imshow(np.log2(M), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(np.log2(M[i, j])) > 2 else "black")
    ax.set_xticks(range(len(BLOCKS)), [LABELS[b] for b in BLOCKS], rotation=35, ha="right")
    ax.set_yticks(range(len(order)), order)
    fig.colorbar(im, ax=ax, label="log2(unseen error / population null)")
    ax.set_title("Development subjects, unseen halves: error relative to the population average (blue = better than average)")
    fig.savefig(OUT / "blocks_heatmap.png", dpi=160)
    plt.close(fig)


def fig_external():
    d = json.loads((ROOT / "outputs/external/m53/paired_vs_m51.json").read_text())["ext_tvb_refined"]
    names = ["total", "auto", "cross", "topo"]
    lab = ["total", "autospectrum", "lagged coherency", "alpha topography"]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x - 0.18, [d[n][1] for n in names], 0.36, label="M5.1 bank", color="0.6")
    ax.bar(x + 0.18, [d[n][0] for n in names], 0.36, label="M5.3 frozen (TVB)", color="C0")
    ax.axhline(1, color="k", ls=":")
    for i, n in enumerate(names):
        ax.text(i + 0.18, d[n][0] + 0.02, f"{d[n][2]:+.3f}\n[{d[n][3]:+.3f}, {d[n][4]:+.3f}]", ha="center", fontsize=7)
    ax.set_xticks(x, lab)
    ax.set_ylabel("median unseen cost / pooled null")
    ax.set_title("Pre-specified external test: 164 rTMS patients (paired change, 95 % CI)")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "external_test.png", dpi=160)
    plt.close(fig)


def fig_quality():
    c = json.loads((ROOT / "outputs/preproc_v2/compare_v1_v2.json").read_text())
    keys = [k for k in BLOCKS if k != "iaf"] + ["iaf"]
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [c["v1"]["split_half_reliability"][k] for k in keys], 0.4, label="original cleaning (v1)", color="0.6")
    ax.bar(x + 0.2, [c["v2"]["split_half_reliability"][k] for k in keys], 0.4, label="artefact-aware cleaning (v2)", color="C2")
    ax.set_xticks(x, [LABELS[k] for k in keys], rotation=35, ha="right")
    ax.set_ylabel("split-half reliability (r)")
    ax.set_ylim(0, 1)
    ax.set_title("Same 252 subjects: test-retest reliability of EEG features between recording halves")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "data_quality.png", dpi=160)
    plt.close(fig)


def fig_group():
    g = pd.read_csv(ROOT / "outputs/m54_final/group_bem_m10pop/group_effects.csv")
    g = g.iloc[::-1]
    se_sd = g.se / (g.mdd_minus_healthy / g.effect_sd_units)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    y = np.arange(len(g))
    colors = ["C3" if p < 0.05 else "0.4" for p in g.p_holm]
    ax.errorbar(g.effect_sd_units, y, xerr=1.96 * se_sd, fmt="o", color="k", ecolor="0.6", ms=0)
    ax.scatter(g.effect_sd_units, y, c=colors, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, g.parameter)
    ax.set_xlabel("MDD-indication − Healthy (SD units, age/sex adjusted, 95 % CI)")
    ax.set_title("Final model (M5.4b, BEM): parameter group effects\nred = Holm-corrected p < 0.05")
    fig.savefig(OUT / "group_effects.png", dpi=160)
    plt.close(fig)


def fig_dorsattn(extra=None):
    p = "gain_DorsAttn_LH"
    d = pd.read_csv(ROOT / "outputs/m54b_kaggle_bem/results/dev_bem_m10pop/subject_fits.csv")[["group", p]]
    e = pd.read_csv(ROOT / "outputs/m54b_kaggle_bem/results/ext_bem_m10pop/subject_fits.csv")[[p]].assign(group="rTMS (external)")
    frames = [d, e]
    for ctrl in (ROOT / "outputs/m54_ctrl/results/ctrl_bem_m10pop/subject_fits.csv",
                 ROOT / "outputs/m54_smc/results/ctrl_smc_bem_m10pop/subject_fits.csv"):
        if ctrl.is_file():
            frames.append(pd.read_csv(ctrl)[["group", p]])
    x = pd.concat(frames)
    order = [g for g in ["Healthy", "SMC", "MDD", "rTMS (external)", "ADHD", "OCD", "INSOMNIA", "TINNITUS"] if g in set(x.group)]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    data = [x[x.group == g][p].to_numpy() for g in order]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), [f"{g}\n(n={len(v)})" for g, v in zip(order, data)])
    for i, v in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(v)), v, s=5, alpha=0.4)
    ax.set_ylabel("left dorsal-attention input log-gain")
    ax.axvline(2.5, color="k", ls="--", lw=0.8)
    ax.text(1.5, ax.get_ylim()[1], "early batch", ha="center", va="top", fontsize=8)
    ax.text(5.0, ax.get_ylim()[1], "late batch (all clinical groups)", ha="center", va="top", fontsize=8)
    ax.set_title("Left dorsal-attention input gain by cohort: it follows acquisition batch, not diagnosis")
    fig.savefig(OUT / "dorsattn_left_by_group.png", dpi=160)
    plt.close(fig)


def fig_rtms():
    s = json.loads((ROOT / "outputs/rtms_prediction/prediction_summary.json").read_text())["results"]
    names = [k for k in ["C", "E", "C+E", "M3", "M4", "C+E+M4"] if k in s]
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    x = np.arange(len(names))
    ax.bar(x, [s[n]["auc"] for n in names], color=["C0" if s[n]["perm_p"] < 0.05 else "0.6" for n in names])
    for i, n in enumerate(names):
        ax.text(i, s[n]["auc"] + 0.01, f"p={s[n]['perm_p']:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="k", ls=":")
    ax.set_xticks(x, names)
    ax.set_ylim(0.3, 0.8)
    ax.set_ylabel("cross-validated AUC (responder)")
    ax.set_title("rTMS response (163 patients): C clinical, E EEG markers, M3/M4 model parameters")
    fig.savefig(OUT / "rtms_prediction.png", dpi=160)
    plt.close(fig)


def fig_reliability():
    cal = json.loads((ROOT / "outputs/m54_final/group_bem_m10pop/calibration.json").read_text())
    rec = pd.read_csv(ROOT / "outputs/m54_synth/results/synthetic_bem_m10pop/recovery.csv", index_col=0)
    names = [n for n in cal if n in rec.index]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    y = np.arange(len(names))
    ax[0].barh(y - 0.2, [cal[n]["icc_test_retest"] for n in names], 0.4, label="test-retest ICC (real halves)")
    ax[0].barh(y + 0.2, [rec.loc[n, "r"] for n in names], 0.4, label="synthetic recovery r")
    ax[0].set_yticks(y, names)
    ax[0].axvline(0.5, color="k", ls=":")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("Reliability and identifiability")
    ax[1].barh(y, [rec.loc[n, "coverage95"] for n in names], color="C2")
    ax[1].axvline(0.95, color="k", ls=":")
    ax[1].set_yticks(y, [])
    ax[1].set_xlim(0, 1)
    ax[1].set_title("Coverage of 95 % Laplace intervals (synthetic)")
    fig.savefig(OUT / "reliability_recovery.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_blocks, fig_external, fig_quality, fig_group, fig_dorsattn, fig_rtms, fig_reliability):
        try:
            f()
            print("ok", f.__name__)
        except Exception as error:  # noqa: BLE001
            print("skip", f.__name__, repr(error))
