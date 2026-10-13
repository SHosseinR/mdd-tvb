"""M5.4 across TDBRAIN disorders: the MDD analyses repeated for every clinical group.

Groups (eyes closed, v2 cleaning, final M5.4 fits): Healthy, MDD (development),
rTMS-treated MDD, adult ADHD, OCD, insomnia, tinnitus, subjective memory
complaints (SMC).  All comparisons are age- and sex-adjusted.  The acquisition-
era question (patients recorded later than most healthy volunteers) is set aside
here and noted in the report.

1. Fit quality per group: unseen deviance ratio and interpretable blocks.
2. Parameters: each disorder vs Healthy, and each disorder vs all other patients
   (calibrated random-effects REML, Holm within disorder); similarity of the
   disorders' effect profiles; distributions of the most affected parameters.
3. Scalp maps (alpha / beta / theta): disorder - Healthy, measured on the unseen
   half and predicted by M5.4.
4. Spectra and coherence differences, measured vs predicted.
5. Classification (cross-validated AUC): disorder vs Healthy and vs other patients,
   from model parameters, plain EEG features, or both.

    .conda/python.exe scripts/m54/disorders_analysis.py
Output: outputs/m54_disorders/*.json|csv and docs/figures/disorders_2026-10/*.png
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_ind

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/m54"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from m54_evaluate import evaluate, features  # noqa: E402
from m54_group import holm, parameter_columns, reml_meta  # noqa: E402
from batch_confound import demographics  # noqa: E402

OUT = ROOT / "outputs/m54_disorders"
FIG = ROOT / "docs/figures/disorders_2026-10"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#17354D", "#536674"
plt.rcParams.update({"font.family": "Arial", "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 180})

COHORTS = {
    "dev": ("outputs/m54b_kaggle_bem/results/dev_bem_m10pop", None, "configs/m54_lists/dev_restEC_v2.txt"),
    "rtms": ("outputs/m54b_kaggle_bem/results/ext_bem_m10pop", "outputs/preproc_v2/dev_restEC/empirical",
             "configs/m54_lists/rtms_restEC_v2.txt"),
    "controls": ("outputs/m54_ctrl/results/ctrl_bem_m10pop", "outputs/preproc_v2/dev_restEC/empirical",
                 "configs/m54_lists/controls_restEC_v2.txt"),
    "smc": ("outputs/m54_smc/results/ctrl_smc_bem_m10pop", "outputs/preproc_v2/dev_restEC/empirical",
            "configs/m54_lists/smc_restEC_v2.txt"),
}
RENAME = {"Healthy": "Healthy", "MDD": "MDD", "rTMS": "rTMS-MDD", "ADHD": "ADHD", "OCD": "OCD", "INSOMNIA": "Insomnia",
          "TINNITUS": "Tinnitus", "SMC": "SMC"}
ORDER = ["Healthy", "MDD", "rTMS-MDD", "ADHD", "OCD", "Insomnia", "Tinnitus", "SMC"]
DISORDERS = ORDER[1:]
COLORS = dict(zip(ORDER, ["#536674", "#1B6AA3", "#4F94C9", "#DE7637", "#8E5BA8", "#00897B", "#C9A227", "#B04A5A"]))
REGIONS = {"occipital": ("O1", "Oz", "O2"), "parietal": ("P3", "Pz", "P4"), "central": ("C3", "Cz", "C4"),
           "frontal": ("F3", "Fz", "F4")}
BANDS = {"theta": "topo_theta", "alpha": "topo_alpha", "beta": "topo_beta"}
NUMBERS: dict = {}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    T = demographics()
    frames, blocks, csd_val, csd_pred = [], [], {}, {}
    for cohort, (run, null, subj) in COHORTS.items():
        t = pd.read_csv(ROOT / run / "subject_fits.csv").set_index("subject_id")
        t["group"] = (t["group"] if "group" in t.columns else "rTMS").map(lambda g: RENAME.get(str(g), str(g)))
        if cohort == "rtms":
            t["group"] = "rTMS-MDD"
        t["cohort"] = cohort
        frames.append(t)
        emp = f"outputs/preproc_v2/{cohort}_restEC/empirical"
        b, _ = evaluate(run, emp, null, set(Path(ROOT / subj).read_text().split()), cohort)
        blocks.append(b.set_index("subject_id").drop(columns="group"))
        val = load_cross_spectral_collection(ROOT / emp / "cross_spectra_validation.npz")
        idx = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
        with np.load(ROOT / run / "predictions.npz") as p:
            pred = dict(zip(p["subject_ids"].astype(str), p["csd"]))
        for s in t.index:
            if s in idx and s in pred:
                csd_val[s] = val.csd[idx[s]]
                csd_pred[s] = np.asarray(pred[s], np.complex128)
        freq = np.asarray(val.frequency_hz, float)
        labels = list(val.channel_names.astype(str))
    x = pd.concat(frames)
    x["age"] = T.reindex(x.index)["age"].astype(float).to_numpy()
    x["sex"] = T.reindex(x.index)["gender"].astype(float).to_numpy()
    x = x[x.group.isin(ORDER)]
    bl = pd.concat(blocks)
    return x, bl, csd_val, csd_pred, freq, labels


def residualise(A, cov):
    """Remove linear age/sex effects (fitted on all subjects) from rows of A."""
    X = np.column_stack([np.ones(len(cov)), cov])
    b, *_ = np.linalg.lstsq(X, A, rcond=None)
    return A - X[:, 1:] @ b[1:]


# ---------------------------------------------------------------------------
# 1. fit quality
# ---------------------------------------------------------------------------
def fig_fit_quality(x, bl):
    cols = {"validation_deviance_ratio": "whole cross-spectrum"}
    block_cols = {"spectrum_ratio": "spectrum", "topo_alpha_ratio": "α map", "topo_beta_ratio": "β map",
                  "topo_theta_ratio": "θ map", "zerolag_alpha_ratio": "α coherence", "zerolag_beta_ratio": "β coherence",
                  "lagged_alpha_ratio": "lagged α", "iaf_ratio": "alpha freq."}
    j = x.join(bl, how="left")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.5]}, layout="constrained")
    data = [j.loc[j.group == g, "validation_deviance_ratio"].dropna() for g in ORDER]
    bp = a1.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, g in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[g]); patch.set_alpha(0.35)
    for i, (g, v) in enumerate(zip(ORDER, data), start=1):
        a1.scatter(np.random.default_rng(i).normal(i, 0.07, len(v)), v, s=5, color=COLORS[g], alpha=0.6)
    a1.axhline(1, color=INK, lw=0.8, ls="--")
    a1.set_xticks(range(1, len(ORDER) + 1), [f"{g}\n(n={len(v)})" for g, v in zip(ORDER, data)], fontsize=9)
    a1.set_ylabel("unseen deviance / population average")
    a1.set_title("Fit to the unseen half (below 1 beats the average)")
    med = pd.DataFrame({g: [j.loc[j.group == g, c].median() for c in block_cols] for g in ORDER}, index=list(block_cols.values()))
    im = a2.imshow(med.to_numpy(), cmap="RdYlGn_r", vmin=0.5, vmax=1.5, aspect="auto")
    a2.set_xticks(range(len(ORDER)), ORDER, rotation=30, ha="right")
    a2.set_yticks(range(len(block_cols)), list(block_cols.values()))
    for r in range(med.shape[0]):
        for c in range(med.shape[1]):
            a2.text(c, r, f"{med.iat[r, c]:.2f}", ha="center", va="center", fontsize=9, color="black")
    a2.set_title("Median error / population average, by block (green < 1 beats the average)")
    fig.colorbar(im, ax=a2, shrink=0.8)
    fig.savefig(FIG / "fit_quality_by_group.png")
    plt.close(fig)
    NUMBERS["fit_quality"] = {g: {"n": int((j.group == g).sum()),
                                  "median_deviance_ratio": float(j.loc[j.group == g, "validation_deviance_ratio"].median()),
                                  "beat_null": float((j.loc[j.group == g, "validation_deviance_ratio"] < 1).mean()),
                                  **{block_cols[c]: float(j.loc[j.group == g, c].median()) for c in block_cols}}
                              for g in ORDER}


# ---------------------------------------------------------------------------
# 2. parameters
# ---------------------------------------------------------------------------
def nice(p):
    return (p.replace("gain_", "gain ").replace("log_gain_", "").replace("_time_contrast", " time contrast")
            .replace("global_coupling", "coupling G").replace("_", " "))


def group_test(x, case_mask, ref_mask, cols, kappa):
    d = x[case_mask | ref_mask].dropna(subset=["age", "sex"])
    case = case_mask[d.index] if isinstance(case_mask, pd.Series) else None
    y_case = case_mask.reindex(d.index).astype(float).to_numpy()
    X = np.column_stack([np.ones(len(d)), y_case, (d.age - d.age.mean()) / d.age.std(), d.sex - d.sex.mean()])
    rows = []
    for c, s in cols:
        v = kappa.get(c, 1.0) * d[s].to_numpy(float) ** 2
        beta, cov, tau2 = reml_meta(d[c].to_numpy(float), v, X)
        sd_between = np.sqrt(tau2 + np.mean(v))
        se = np.sqrt(cov[1, 1])
        rows.append({"parameter": c, "effect_sd": beta[1] / sd_between, "se_sd": se / sd_between,
                     "p": 2 * norm.sf(abs(beta[1] / se))})
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p.to_numpy())
    return r


def parameters(x):
    cols = parameter_columns(x)
    cal = json.loads((ROOT / "outputs/m54_final/group_bem_m10pop/calibration.json").read_text())
    kappa = {c: max(1.0, v.get("robust_var_z", 1.0)) for c, v in cal.items()}
    patients = x.group != "Healthy"
    vs_h, vs_p = {}, {}
    for g in DISORDERS:
        vs_h[g] = group_test(x, x.group == g, x.group == "Healthy", cols, kappa).set_index("parameter")
        mdd_family = {"MDD", "rTMS-MDD"}
        others = patients & ~x.group.isin(mdd_family if g in mdd_family else {g})
        vs_p[g] = group_test(x, x.group == g, others, cols, kappa).set_index("parameter")
    names = [c for c, _ in cols]
    for title, res, fname in (("vs Healthy", vs_h, "parameter_effects_vs_healthy.png"),
                              ("vs all other patients", vs_p, "parameter_effects_vs_other_patients.png")):
        E = pd.DataFrame({g: res[g].effect_sd for g in DISORDERS}).loc[names]
        Ph = pd.DataFrame({g: res[g].p_holm for g in DISORDERS}).loc[names]
        P = pd.DataFrame({g: res[g].p for g in DISORDERS}).loc[names]
        fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(names) + 1.6), layout="constrained")
        im = ax.imshow(E.to_numpy(), cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="auto")
        ax.set_xticks(range(len(DISORDERS)), DISORDERS, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), [nice(n) for n in names])
        for r in range(E.shape[0]):
            for c in range(E.shape[1]):
                mark = "**" if Ph.iat[r, c] < 0.05 else ("·" if P.iat[r, c] < 0.05 else "")
                ax.text(c, r, f"{E.iat[r, c]:+.2f}{mark}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Parameter differences, each disorder {title}\n(SD units, age/sex adjusted; ** Holm p < 0.05, · p < 0.05)")
        fig.colorbar(im, ax=ax, shrink=0.6, label="effect (SD)")
        fig.savefig(FIG / fname)
        plt.close(fig)
        E.to_csv(OUT / fname.replace(".png", "_effect.csv"))
        Ph.to_csv(OUT / fname.replace(".png", "_pholm.csv"))
        P.to_csv(OUT / fname.replace(".png", "_p.csv"))
    # similarity of the disorders' effect profiles (vs Healthy)
    E = pd.DataFrame({g: vs_h[g].effect_sd for g in DISORDERS}).loc[names]
    S = E.corr()
    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
    im = ax.imshow(S.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(DISORDERS)), DISORDERS, rotation=30, ha="right")
    ax.set_yticks(range(len(DISORDERS)), DISORDERS)
    for r in range(len(DISORDERS)):
        for c in range(len(DISORDERS)):
            ax.text(c, r, f"{S.iat[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("How similar are the disorders' parameter profiles?\n(correlation of effects vs Healthy across parameters)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "parameter_profile_similarity.png")
    plt.close(fig)
    # distributions of the most affected parameters
    strength = (E.abs()).max(axis=1).sort_values(ascending=False)
    top = list(strength.index[:9])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), layout="constrained")
    ref = x.loc[x.group == "Healthy"]
    for ax, p in zip(axes.ravel(), top):
        z = (x[p] - ref[p].mean()) / ref[p].std()
        data = [z[x.group == g].dropna() for g in ORDER]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        for patch, g in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COLORS[g]); patch.set_alpha(0.35)
        for i, (g, v) in enumerate(zip(ORDER, data), start=1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.07, len(v)), v, s=4, color=COLORS[g], alpha=0.5)
        ax.axhline(0, color=INK, lw=0.7, ls="--")
        ax.set_xticks(range(1, len(ORDER) + 1), ORDER, rotation=35, ha="right", fontsize=8.5)
        ax.set_title(nice(p), fontsize=10.5)
        ax.set_ylabel("z vs Healthy")
    fig.suptitle("The nine parameters with the largest differences (any disorder vs Healthy)", fontweight="bold", color=INK)
    fig.savefig(FIG / "parameter_distributions_top9.png")
    plt.close(fig)
    NUMBERS["parameters"] = {
        "vs_healthy_holm_significant": {g: vs_h[g].index[vs_h[g].p_holm < 0.05].tolist() for g in DISORDERS},
        "vs_other_patients_holm_significant": {g: vs_p[g].index[vs_p[g].p_holm < 0.05].tolist() for g in DISORDERS},
        "profile_similarity": S.round(3).to_dict(), "top9": top}
    return vs_h, vs_p


# ---------------------------------------------------------------------------
# 3-4. maps, spectra, coherence: measured (unseen half) vs predicted
# ---------------------------------------------------------------------------
def feature_matrix(x, csd, key, freq, labels):
    ids = [s for s in x.index if s in csd]
    A = np.stack([features(csd[s], freq, labels)[key] for s in ids])
    return ids, A


def fig_maps(x, csd_val, csd_pred, freq, labels):
    import mne
    info = mne.create_info(labels, 500.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    stats = {}
    for band, key in BANDS.items():
        ids, V = feature_matrix(x, csd_val, key, freq, labels)
        _, P = feature_matrix(x, csd_pred, key, freq, labels)
        d = x.loc[ids]
        cov = d[["age", "sex"]].fillna(d[["age", "sex"]].mean()).to_numpy()
        V, P = residualise(V, cov), residualise(P, cov)
        g = d.group.to_numpy()
        fig, axes = plt.subplots(len(DISORDERS), 3, figsize=(8.6, 2.45 * len(DISORDERS)), layout="constrained")
        lim = 0
        maps = {}
        for dis in DISORDERS:
            mv = V[g == dis].mean(0) - V[g == "Healthy"].mean(0)
            mp = P[g == dis].mean(0) - P[g == "Healthy"].mean(0)
            t, pv = ttest_ind(V[g == dis], V[g == "Healthy"], axis=0)
            maps[dis] = (mv, mp, t, pv)
            lim = max(lim, np.abs(mv).max(), np.abs(mp).max())
        for r, dis in enumerate(DISORDERS):
            mv, mp, t, pv = maps[dis]
            rr = float(np.corrcoef(mv, mp)[0, 1])
            q = np.sort(pv)
            n_fdr = int(np.sum(q <= 0.05 * np.arange(1, len(q) + 1) / len(q)))  # Benjamini-Hochberg count
            mask = pv < 0.05
            for c, (m, title) in enumerate(((mv, "measured (unseen half)"), (mp, "predicted by M5.4"))):
                mne.viz.plot_topomap(m, info, axes=axes[r, c], show=False, cmap="RdBu_r", vlim=(-lim, lim), contours=0,
                                     sensors=True, mask=mask if c == 0 else None,
                                     mask_params=dict(marker="o", markerfacecolor="k", markersize=3))
                if r == 0:
                    axes[r, c].set_title(title, fontsize=10)
            axes[r, 0].text(-0.3, 0.5, dis, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=11,
                            fontweight="bold", color=COLORS[dis])
            axes[r, 2].axis("off")
            axes[r, 2].text(0.0, 0.5, f"r(measured, predicted) = {rr:+.2f}\nchannels p < 0.05: {int(mask.sum())}/26\n"
                            f"FDR-significant: {n_fdr}\nmax |t| = {np.abs(t).max():.1f}", va="center", fontsize=9.5,
                            color=INK)
            stats[f"{band}|{dis}"] = {"r_measured_predicted": rr, "channels_p05": int(mask.sum()), "fdr": n_fdr,
                                      "max_abs_t": float(np.abs(t).max())}
        fig.suptitle(f"{band.capitalize()} topography: disorder − Healthy (centred log power, age/sex adjusted;\n"
                     f"dots = channels with p < 0.05)", fontweight="bold", color=INK, fontsize=11)
        fig.savefig(FIG / f"maps_{band}.png")
        plt.close(fig)
    NUMBERS["maps"] = stats


def fig_maps_wide(x, csd_val, csd_pred, freq, labels, band="alpha"):
    """Slide version: disorders as columns, measured / predicted as rows."""
    import mne
    info = mne.create_info(labels, 500.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
    key = BANDS[band]
    ids, V = feature_matrix(x, csd_val, key, freq, labels)
    _, P = feature_matrix(x, csd_pred, key, freq, labels)
    d = x.loc[ids]
    cov = d[["age", "sex"]].fillna(d[["age", "sex"]].mean()).to_numpy()
    V, P = residualise(V, cov), residualise(P, cov)
    g = d.group.to_numpy()
    maps = {dis: (V[g == dis].mean(0) - V[g == "Healthy"].mean(0), P[g == dis].mean(0) - P[g == "Healthy"].mean(0),
                  ttest_ind(V[g == dis], V[g == "Healthy"], axis=0)[1]) for dis in DISORDERS}
    lim = max(max(np.abs(a).max(), np.abs(b).max()) for a, b, _ in maps.values())
    fig, axes = plt.subplots(2, len(DISORDERS), figsize=(2.15 * len(DISORDERS), 5.0), layout="constrained")
    for c, dis in enumerate(DISORDERS):
        mv, mp, pv = maps[dis]
        for r, m in enumerate((mv, mp)):
            im, _ = mne.viz.plot_topomap(m, info, axes=axes[r, c], show=False, cmap="RdBu_r", vlim=(-lim, lim),
                                         contours=0, sensors=True, mask=(pv < 0.05) if r == 0 else None,
                                         mask_params=dict(marker="o", markerfacecolor="k", markersize=3))
        axes[0, c].set_title(dis, color=COLORS[dis], fontsize=12)
        axes[1, c].set_xlabel(f"r = {np.corrcoef(mv, mp)[0, 1]:+.2f}", fontsize=10.5)
    axes[0, 0].set_ylabel("measured", fontsize=12)
    axes[1, 0].set_ylabel("M5.4", fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Δ centred log power")
    fig.savefig(FIG / f"maps_{band}_wide.png")
    plt.close(fig)


def fig_spectra_wide(x, csd_val, csd_pred, freq, labels, regions=("occipital", "central")):
    """Slide version: disorders as columns, two channel regions as rows."""
    ids = [s for s in x.index if s in csd_val]
    d = x.loc[ids]
    g = d.group.to_numpy()
    cov = d[["age", "sex"]].fillna(d[["age", "sex"]].mean()).to_numpy()
    fig, axes = plt.subplots(len(regions), len(DISORDERS), figsize=(2.25 * len(DISORDERS), 2.6 * len(regions)),
                             sharex=True, sharey="row", layout="constrained")
    for r, rname in enumerate(regions):
        chans = REGIONS[rname]
        V = residualise(np.stack([rel_log_spectrum(csd_val[s], labels, chans) for s in ids]), cov)
        P = residualise(np.stack([rel_log_spectrum(csd_pred[s], labels, chans) for s in ids]), cov)
        for c, dis in enumerate(DISORDERS):
            ax = axes[r, c]
            a, b = V[g == dis], V[g == "Healthy"]
            m = a.mean(0) - b.mean(0)
            se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
            ax.fill_between(freq, m - 1.96 * se, m + 1.96 * se, color=COLORS[dis], alpha=0.18, lw=0)
            ax.plot(freq, m, color=COLORS[dis], lw=2)
            ax.plot(freq, P[g == dis].mean(0) - P[g == "Healthy"].mean(0), color=INK, lw=1.4, ls="--")
            ax.axhline(0, color=MUTED, lw=0.7)
            if r == 0:
                ax.set_title(dis, color=COLORS[dis], fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{rname}\nΔ log10 rel. power", fontsize=10.5)
            if r == len(regions) - 1:
                ax.set_xlabel("Hz")
    fig.savefig(FIG / "spectra_differences_wide.png")
    plt.close(fig)


def rel_log_spectrum(csd, labels, chans):
    pw = np.maximum(np.real(np.einsum("fii->fi", csd)), 1e-30)
    pw = pw / pw.sum(0, keepdims=True)
    return np.log10(pw[:, [labels.index(c) for c in chans]].mean(1))


def fig_spectra(x, csd_val, csd_pred, freq, labels):
    ids = [s for s in x.index if s in csd_val]
    d = x.loc[ids]
    g = d.group.to_numpy()
    cov = d[["age", "sex"]].fillna(d[["age", "sex"]].mean()).to_numpy()
    fig, axes = plt.subplots(len(DISORDERS), len(REGIONS), figsize=(14, 2.3 * len(DISORDERS)), sharex=True,
                             layout="constrained")
    stats = {}
    for c, (rname, chans) in enumerate(REGIONS.items()):
        V = residualise(np.stack([rel_log_spectrum(csd_val[s], labels, chans) for s in ids]), cov)
        P = residualise(np.stack([rel_log_spectrum(csd_pred[s], labels, chans) for s in ids]), cov)
        for r, dis in enumerate(DISORDERS):
            ax = axes[r, c]
            a, b = V[g == dis], V[g == "Healthy"]
            m = a.mean(0) - b.mean(0)
            se = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
            mp = P[g == dis].mean(0) - P[g == "Healthy"].mean(0)
            ax.fill_between(freq, m - 1.96 * se, m + 1.96 * se, color=COLORS[dis], alpha=0.18, lw=0)
            ax.plot(freq, m, color=COLORS[dis], lw=2, label="measured (unseen half)")
            ax.plot(freq, mp, color=INK, lw=1.6, ls="--", label="predicted by M5.4")
            ax.axhline(0, color=MUTED, lw=0.7)
            if r == 0:
                ax.set_title(f"{rname} ({', '.join(chans)})", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{dis}\nΔ log10 rel. power", fontsize=9.5, color=COLORS[dis])
            stats[f"{dis}|{rname}"] = float(np.corrcoef(m, mp)[0, 1])
    axes[0, -1].legend(fontsize=8.5, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("frequency (Hz)")
    fig.suptitle("Relative spectrum, disorder − Healthy (age/sex adjusted; band = 95 % CI)", fontweight="bold", color=INK)
    fig.savefig(FIG / "spectra_differences.png")
    plt.close(fig)
    NUMBERS["spectra_r_measured_predicted"] = stats


def fig_coherence(x, csd_val, csd_pred, freq, labels):
    n = len(labels)
    iu = np.triu_indices(n, 1)
    out = {}
    for key, title in (("zerolag_alpha", "zero-lag alpha coherence"), ("lagged_alpha", "lagged alpha coherency")):
        ids, V = feature_matrix(x, csd_val, key, freq, labels)
        _, P = feature_matrix(x, csd_pred, key, freq, labels)
        d = x.loc[ids]
        cov = d[["age", "sex"]].fillna(d[["age", "sex"]].mean()).to_numpy()
        V, P = residualise(V, cov), residualise(P, cov)
        g = d.group.to_numpy()
        fig, axes = plt.subplots(2, len(DISORDERS), figsize=(2.3 * len(DISORDERS), 5.2), layout="constrained")
        diffs = {}
        lim = 0
        for dis in DISORDERS:
            mv = V[g == dis].mean(0) - V[g == "Healthy"].mean(0)
            mp = P[g == dis].mean(0) - P[g == "Healthy"].mean(0)
            diffs[dis] = (mv, mp)
            lim = max(lim, np.abs(mv).max())
        for c, dis in enumerate(DISORDERS):
            for r, m in enumerate(diffs[dis]):
                M = np.zeros((n, n))
                M[iu] = m
                M = M + M.T
                im = axes[r, c].imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            rr = float(np.corrcoef(*diffs[dis])[0, 1])
            axes[0, c].set_title(dis, color=COLORS[dis], fontsize=10)
            axes[1, c].set_xlabel(f"r = {rr:+.2f}", fontsize=9)
            out[f"{key}|{dis}"] = {"r_measured_predicted": rr, "mean_measured_diff": float(diffs[dis][0].mean())}
        axes[0, 0].set_ylabel("measured", fontsize=10)
        axes[1, 0].set_ylabel("predicted", fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.7)
        fig.suptitle(f"{title}: disorder − Healthy (26 × 26 channels, age/sex adjusted)", fontweight="bold", color=INK)
        fig.savefig(FIG / f"{key}_differences.png")
        plt.close(fig)
        # similarity of measured difference patterns across disorders
        S = pd.DataFrame({dis: diffs[dis][0] for dis in DISORDERS}).corr()
        out[f"{key}|pattern_similarity"] = S.round(3).to_dict()
    NUMBERS["coherence"] = out


# ---------------------------------------------------------------------------
# 5. classification
# ---------------------------------------------------------------------------
def descriptive(csd, freq, labels):
    ft = features(csd, freq, labels)
    pw = np.maximum(np.real(np.einsum("fii->fi", csd)), 1e-30)
    tot = pw.sum(0)
    rel = []
    for lo, hi in ((2, 3), (4, 7), (8, 12), (13, 25), (26, 40)):
        sel = (freq >= lo) & (freq <= hi)
        for chans in REGIONS.values():
            idx = [labels.index(c) for c in chans]
            rel.append(np.log(pw[sel][:, idx].sum(0) / tot[idx]).mean())
    return np.concatenate([rel, ft["iaf"], ft["topo_alpha"], [np.log(tot.mean())]])


def cv_auc(X, y, cov, reps=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    aucs = []
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed + rep)
        pred = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            C = np.column_stack([np.ones(len(tr)), cov[tr]])
            b, *_ = np.linalg.lstsq(C, X[tr], rcond=None)            # age/sex removed with training-fold fits
            Xtr = X[tr] - cov[tr] @ b[1:]
            Xte = X[te] - cov[te] @ b[1:]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
            clf.fit((Xtr - mu) / sd, y[tr])
            pred[te] = clf.predict_proba((Xte - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return float(np.mean(aucs)), float(np.std(aucs))


def fig_classification(x, csd_val, freq, labels):
    pcols = [c for c, _ in parameter_columns(x)]
    ids = [s for s in x.index if s in csd_val and np.isfinite(x.loc[s, ["age", "sex"]].astype(float)).all()]
    d = x.loc[ids]
    D = np.stack([descriptive(csd_val[s], freq, labels) for s in ids])
    Mp = d[pcols].to_numpy(float)
    cov = d[["age", "sex"]].to_numpy(float)
    sets = {"model parameters": Mp, "plain EEG features": D, "both": np.column_stack([Mp, D])}
    res = {}
    for contrast in ("vs Healthy", "vs other patients"):
        for dis in DISORDERS:
            mdd_family = {"MDD", "rTMS-MDD"}
            if contrast == "vs Healthy":
                keep = d.group.isin([dis, "Healthy"]).to_numpy()
            else:
                keep = ((d.group != "Healthy") & ~d.group.isin((mdd_family - {dis}) if dis in mdd_family else set())).to_numpy()
            y = (d.group.to_numpy()[keep] == dis).astype(int)
            for name, X in sets.items():
                res[(contrast, dis, name)] = cv_auc(X[keep], y, cov[keep])
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True, layout="constrained")
    w = 0.26
    for ax, contrast in zip(axes, ("vs Healthy", "vs other patients")):
        for k, name in enumerate(sets):
            m = [res[(contrast, dis, name)][0] for dis in DISORDERS]
            s = [res[(contrast, dis, name)][1] for dis in DISORDERS]
            ax.bar(np.arange(len(DISORDERS)) + (k - 1) * w, m, w, yerr=s, label=name,
                   color=["#1B6AA3", "#9AA9B5", "#DE7637"][k], capsize=2)
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(DISORDERS)), DISORDERS, rotation=30, ha="right")
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"Each disorder {contrast} (5-fold CV × 5, age/sex removed)")
    axes[0].set_ylabel("cross-validated AUC")
    axes[1].legend(loc="upper right")
    fig.savefig(FIG / "classification_auc.png")
    plt.close(fig)
    NUMBERS["classification_auc"] = {f"{c}|{dname}|{n}": {"mean": v[0], "sd": v[1]} for (c, dname, n), v in res.items()}


def main():
    x, bl, csd_val, csd_pred, freq, labels = load()
    if "--slides" in sys.argv:  # only the wide slide figures
        for band in ("alpha", "beta"):
            fig_maps_wide(x, csd_val, csd_pred, freq, labels, band)
        fig_spectra_wide(x, csd_val, csd_pred, freq, labels)
        return
    NUMBERS["groups"] = {g: {"n": int((x.group == g).sum()), "age_mean": float(x.loc[x.group == g, "age"].mean()),
                             "sex_mean_code": float(x.loc[x.group == g, "sex"].mean())} for g in ORDER}
    print(json.dumps(NUMBERS["groups"], indent=1))
    fig_fit_quality(x, bl)
    parameters(x)
    fig_maps(x, csd_val, csd_pred, freq, labels)
    fig_spectra(x, csd_val, csd_pred, freq, labels)
    fig_coherence(x, csd_val, csd_pred, freq, labels)
    fig_classification(x, csd_val, freq, labels)
    (OUT / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float))
    print(json.dumps({k: v for k, v in NUMBERS.items() if k in ("fit_quality", "parameters")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
