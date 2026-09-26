"""Figures that show what the fits look like: spectra, scalp maps, coherence, EEG traces.

All comparisons use each subject's *unseen* second half, except where the first
(fitted) half is shown explicitly.  Models:
    M5.1  bank posterior (original preprocessing, v1)
    M5.3  frozen audit model (v1 and re-cleaned v2 data)
    M5.4  final likelihood model (BEM, 10 modes, population background; v2)
Model EEG traces are exact samples of the fitted model: in the stable linear
regime the model's scalp EEG is a stationary Gaussian process whose
cross-spectrum is the predicted CSD, so drawing Fourier coefficients with that
covariance *is* simulating the fitted model.

    .conda/python.exe scripts/m54/make_fit_figures.py
Output: docs/figures/fit_examples_2026-09-26/*.png and a JSON of the numbers shown.
"""
from __future__ import annotations

import json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("MDD_TVB_JAX_X64", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/m54"))
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402

OUT = ROOT / "docs/figures/fit_examples_2026-09-26"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#17354D"
MUTED = "#536674"
GRID = "#DDE3E9"
C_M51 = "#9AA9B5"
C_M53 = "#DE7637"
C_M54 = "#1B6AA3"
C_POP = "#00897B"
C_FIT = "#9AA9B5"
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200})

V1 = "outputs/m5_spectral_m51_production/empirical"
V2 = "outputs/preproc_v2/dev_restEC/empirical"
V2X = "outputs/preproc_v2/rtms_restEC/empirical"
PRED = {
    "M5.1 (v1)": "outputs/m54_eval/m51_v1/predictions.npz",
    "M5.3 (v1)": "outputs/linear_regime/kaggle/final/final_tvb_refined/predictions.npz",
    "M5.3": "outputs/m54_kaggle_bem/results/m53v2_dev_refined/predictions.npz",
    "M5.4": "outputs/m54b_kaggle_bem/results/dev_bem_m10pop/predictions.npz",
    "M5.3 ext": "outputs/m54_kaggle_bem/results/m53v2_ext_refined/predictions.npz",
    "M5.4 ext": "outputs/m54b_kaggle_bem/results/ext_bem_m10pop/predictions.npz",
}
FITS_M54 = "outputs/m54b_kaggle_bem/results/dev_bem_m10pop/subject_fits.csv"
POP_M54 = "outputs/m54b_kaggle_bem/results/population_bem_m10.json"
GROUPS = {"occipital (O1, Oz, O2)": ("O1", "Oz", "O2"), "parietal (P3, Pz, P4)": ("P3", "Pz", "P4"),
          "frontal (F3, Fz, F4)": ("F3", "Fz", "F4"), "temporal (T7, T8)": ("T7", "T8")}
TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
NUMBERS: dict = {}
CACHE: dict = {}


def coll(emp, half):
    return load_cross_spectral_collection(ROOT / emp / f"cross_spectra_{half}.npz")


def preds(key):
    if key not in CACHE:
        with np.load(ROOT / PRED[key]) as p:
            CACHE[key] = dict(zip(p["subject_ids"].astype(str), p["csd"].astype(np.complex128)))
    return CACHE[key]


def power(csd):
    return np.real(np.einsum("...fii->...fi", csd))


def rel_power(csd):
    """Power normalised by the subject's mean power over channels and 2-40 Hz."""
    p = power(csd)
    return p / p.mean(axis=(-1, -2), keepdims=True)


def group_curve(csd, labels, chans):
    idx = [labels.index(c) for c in chans]
    return rel_power(csd)[..., idx].mean(-1)


def dev_ids():
    return Path(ROOT / "configs/m54_lists/dev_restEC_v2.txt").read_text().split()


def style_spectrum_axis(ax, ylabel=True):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(2, 40)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlabel("frequency (Hz)")
    if ylabel:
        ax.set_ylabel("relative power")


# ---------------------------------------------------------------------------
# 1. group-average spectra
# ---------------------------------------------------------------------------
def fig_group_spectra():
    rows = [("Original preprocessing, 262 development subjects", V1, [("M5.1 (v1)", C_M51, "--"), ("M5.3 (v1)", C_M53, "-")], None),
            ("Re-cleaned EEG, 252 development subjects", V2, [("M5.3", C_M53, "-"), ("M5.4", C_M54, "-")], set(dev_ids())),
            ("Re-cleaned EEG, 162 external rTMS patients", V2X, [("M5.3 ext", C_M53, "-"), ("M5.4 ext", C_M54, "-")],
             set(Path(ROOT / "configs/m54_lists/rtms_restEC_v2.txt").read_text().split()))]
    fig = plt.figure(figsize=(15, 11), layout="constrained")
    subs = fig.subfigures(3, 1, hspace=0.04)
    summary = {}
    for r, (label, emp, models, keep) in enumerate(rows):
        sub = subs[r]
        sub.suptitle(label, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
        axrow = sub.subplots(1, 4, sharex=True)
        val = coll(emp, "validation")
        labels = list(val.channel_names.astype(str))
        f = np.asarray(val.frequency_hz, float)
        P = {m: preds(m) for m, _, _ in models}
        ids = [s for s in val.subject_ids.astype(str) if all(s in P[m] for m, _, _ in models) and (keep is None or s in keep)]
        index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
        meas = val.csd[[index[s] for s in ids]]
        for c, (gname, chans) in enumerate(GROUPS.items()):
            ax = axrow[c]
            y = group_curve(meas, labels, chans)
            m, se = y.mean(0), y.std(0) / np.sqrt(len(y))
            ax.fill_between(f, m - 1.96 * se, m + 1.96 * se, color=INK, alpha=0.12, lw=0)
            ax.plot(f, m, color=INK, lw=2.4, label="measured, unseen half")
            for name, color, ls in models:
                yp = group_curve(np.stack([P[name][s] for s in ids]), labels, chans).mean(0)
                ax.plot(f, yp, color=color, lw=2, ls=ls, label=name.replace(" ext", "").replace(" (v1)", ""))
                summary[f"{label} | {gname} | {name}"] = float(np.corrcoef(np.log(m), np.log(yp))[0, 1])
            style_spectrum_axis(ax, ylabel=c == 0)
            ax.set_title(gname, fontsize=11, fontweight="normal", color=MUTED)
            if r < 2:
                ax.set_xlabel("")
        axrow[3].legend(loc="upper right", fontsize=10)
    NUMBERS["group_spectra_log_shape_r"] = summary
    fig.savefig(OUT / "group_spectra.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# subject selection (by M5.4 unseen deviance ratio)
# ---------------------------------------------------------------------------
def pick_subjects():
    t = pd.read_csv(ROOT / FITS_M54).set_index("subject_id")
    both = set(preds("M5.3")) & set(t.index)
    t = t.loc[sorted(both)]
    picks = {}
    for q, name in ((0.1, "better fit (10th percentile)"), (0.5, "typical fit (median)"), (0.9, "harder fit (90th percentile)")):
        target = t.validation_deviance_ratio.quantile(q)
        sid = (t.validation_deviance_ratio - target).abs().idxmin()
        picks[sid] = name
    NUMBERS["example_subjects"] = {s: {"label": n, "group": t.loc[s, "group"],
                                       "m54_unseen_deviance_ratio": float(t.loc[s, "validation_deviance_ratio"])}
                                   for s, n in picks.items()}
    return picks


# ---------------------------------------------------------------------------
# 2. individual spectra
# ---------------------------------------------------------------------------
def fig_subject_spectra(picks):
    fit, val = coll(V2, "fit"), coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    fig = plt.figure(figsize=(15, 3.7 * len(picks)), layout="constrained")
    subs = fig.subfigures(len(picks), 1, hspace=0.04)
    for r, (sid, name) in enumerate(picks.items()):
        i = index[sid]
        subs[r].suptitle(f"{name}: {val.groups[i]} subject {sid}", x=0.01, ha="left", fontsize=13,
                         fontweight="bold", color=INK)
        axrow = subs[r].subplots(1, 4, sharex=True)
        for c, (gname, chans) in enumerate(GROUPS.items()):
            ax = axrow[c]
            ax.plot(f, group_curve(fit.csd[i], labels, chans), color=C_FIT, lw=1.3, label="measured, fitted half")
            ax.plot(f, group_curve(val.csd[i], labels, chans), color=INK, lw=2.2, label="measured, unseen half")
            ax.plot(f, group_curve(P53[sid], labels, chans), color=C_M53, lw=1.8, label="M5.3")
            ax.plot(f, group_curve(P54[sid], labels, chans), color=C_M54, lw=1.8, label="M5.4")
            style_spectrum_axis(ax, ylabel=c == 0)
            ax.set_title(gname, fontsize=11, fontweight="normal", color=MUTED)
            if r < len(picks) - 1:
                ax.set_xlabel("")
        if r == 0:
            axrow[3].legend(loc="upper right", fontsize=9.5)
    fig.savefig(OUT / "subject_spectra.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. scalp maps
# ---------------------------------------------------------------------------
def _info(labels):
    import mne
    info = mne.create_info(labels, 500.0, "eeg")
    info.set_montage("standard_1005")
    return info


def topo_values(csd, freq, lo, hi):
    p = power(csd)[..., (freq >= lo) & (freq <= hi), :].mean(-2)
    v = np.log10(p)
    return v - v.mean(-1, keepdims=True)


def fig_topomaps(picks):
    import mne
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    info = _info(labels)
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    ids = [s for s in dev_ids() if s in index and s in P53 and s in P54]
    cols = [("Group mean\n(252 subjects)", ids)] + [(n.split(" (")[0].capitalize() + f"\n{sid}", [sid]) for sid, n in picks.items()]
    rows = [("Measured\n(unseen half)", lambda s: val.csd[index[s]]), ("M5.3", lambda s: P53[s]), ("M5.4", lambda s: P54[s])]
    for band, (lo, hi), fname in (("alpha 8–12 Hz", (8, 12), "topomaps_alpha.png"), ("beta 13–19 Hz", (13, 19), "topomaps_beta.png")):
        fig, axes = plt.subplots(3, len(cols), figsize=(3.2 * len(cols) + 1.0, 8.4), constrained_layout=True)
        vals = [[topo_values(np.stack([get(s) for s in sub]), f, lo, hi).mean(0) for _, sub in cols] for _, get in rows]
        lim = np.percentile(np.abs(np.concatenate([np.ravel(v) for row in vals for v in row])), 98)
        corr = {}
        for r, (rname, _) in enumerate(rows):
            for c, (cname, _) in enumerate(cols):
                im, _ = mne.viz.plot_topomap(vals[r][c], info, axes=axes[r, c], show=False, cmap="RdBu_r",
                                             vlim=(-lim, lim), contours=0, sensors=True)
                if r == 0:
                    axes[r, c].set_title(cname, fontsize=11.5)
                if r > 0:
                    rr = float(np.corrcoef(vals[0][c], vals[r][c])[0, 1])
                    corr[f"{rname}|{cname.splitlines()[0]}"] = rr
                    axes[r, c].text(0.5, -0.08, f"r = {rr:.2f} with measured", transform=axes[r, c].transAxes,
                                    ha="center", fontsize=10, color=MUTED)
            axes[r, 0].text(-0.28, 0.5, rname, transform=axes[r, 0].transAxes, ha="right", va="center",
                            fontsize=12, fontweight="bold", color=INK)
        cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02)
        cb.set_label(f"log10 {band} power, centred across channels")
        NUMBERS[f"topomap_r_{band.split()[0]}"] = corr
        fig.savefig(OUT / fname)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. zero-lag coherence vs distance, and one subject's coherence matrices
# ---------------------------------------------------------------------------
def real_coherence(csd, freq, lo, hi):
    c = csd[..., (freq >= lo) & (freq <= hi), :, :].mean(-3)
    d = np.sqrt(np.einsum("...ii->...i", c).real)
    return np.real(c) / (d[..., :, None] * d[..., None, :])


def fig_coherence(picks):
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    xyz = pd.read_csv(ROOT / "data/sensors/TDBRAIN_Table3_electrode_coordinates.csv").set_index("label").loc[labels, ["x_mm", "y_mm", "z_mm"]].to_numpy()
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(labels), 1)
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    P51 = preds("M5.1 (v1)")
    ids = [s for s in dev_ids() if s in index and s in P53 and s in P54 and s in P51]
    fig = plt.figure(figsize=(15, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])
    stats = {}
    bins = np.arange(20, 220, 20)
    for c, (band, (lo, hi)) in enumerate((("alpha 8–12 Hz", (8, 12)), ("beta 13–19 Hz", (13, 19)))):
        ax = fig.add_subplot(gs[0, 2 * c:2 * c + 2])
        series = [("measured, unseen half", np.stack([val.csd[index[s]] for s in ids]), INK, "o"),
                  ("M5.1", np.stack([P51[s] for s in ids]), C_M51, "s"),
                  ("M5.3", np.stack([P53[s] for s in ids]), C_M53, "^"),
                  ("M5.4", np.stack([P54[s] for s in ids]), C_M54, "D")]
        for name, arr, color, marker in series:
            coh = real_coherence(arr, f, lo, hi).mean(0)[iu]
            ax.scatter(dist[iu], coh, s=9, color=color, alpha=0.35, lw=0)
            which = np.digitize(dist[iu], bins)
            mids = [(bins[k - 1] + 10, coh[which == k].mean()) for k in range(1, len(bins)) if np.any(which == k)]
            ax.plot(*zip(*mids), color=color, lw=2.4, marker=marker, ms=6, label=name)
            if name != "measured, unseen half":
                ref = real_coherence(series[0][1], f, lo, hi).mean(0)[iu]
                stats[f"{band}|{name}"] = {"rmse_vs_measured": float(np.sqrt(np.mean((coh - ref) ** 2))),
                                           "r_vs_measured": float(np.corrcoef(coh, ref)[0, 1])}
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xlabel("distance between electrodes (mm)")
        ax.set_ylabel("zero-lag coherence (real part)")
        ax.set_title(f"{band}: group mean of 252 subjects")
        ax.grid(True, color=GRID, lw=0.6)
        if c == 0:
            ax.legend(loc="upper right", fontsize=10)
    sid = [s for s, n in picks.items() if n.startswith("typical")][0]
    mats = [("Measured, unseen half", val.csd[index[sid]]), ("M5.1 (original data)", P51[sid]), ("M5.3", P53[sid]), ("M5.4", P54[sid])]
    for c, (name, csd) in enumerate(mats):
        ax = fig.add_subplot(gs[1, c])
        im = ax.imshow(real_coherence(csd, f, 8, 12), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_title(f"{name}\nsubject {sid}, alpha", fontsize=11)
    fig.colorbar(im, ax=fig.axes[-4:], shrink=0.8, label="zero-lag coherence")
    NUMBERS["coherence_vs_measured"] = stats
    fig.savefig(OUT / "coherence.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. real EEG vs EEG generated by the fitted model
# ---------------------------------------------------------------------------
def sample_from_csd(csd, freq, fs, n, rng):
    """Stationary Gaussian multichannel series whose one-sided CSD is ``csd`` (V^2/Hz)."""
    fk = np.fft.rfftfreq(n, 1.0 / fs)
    F = csd.shape[-1]
    X = np.zeros((len(fk), F), dtype=complex)
    for k, fr in enumerate(fk):
        if fr < freq[0] - 0.5 or fr > freq[-1] + 0.5:
            continue
        j = np.clip(np.searchsorted(freq, fr) - 1, 0, len(freq) - 2)
        w = np.clip((fr - freq[j]) / (freq[j + 1] - freq[j]), 0, 1)
        S = (1 - w) * csd[j] + w * csd[j + 1]
        S = 0.5 * (S + S.conj().T)
        lam, V = np.linalg.eigh(S)
        L = V * np.sqrt(np.clip(lam, 0, None))
        z = (rng.normal(size=F) + 1j * rng.normal(size=F)) / np.sqrt(2)
        X[k] = L @ z * np.sqrt(n * fs / 2.0)
    return np.fft.irfft(X, n=n, axis=0)


def band(x, fs, lo=2.0, hi=40.0):
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def fig_traces(picks):
    from mdd_tvb.preprocess_v2 import clean_epoch_starts, preprocess_bdf, split_halves
    val, fit = coll(V2, "validation"), coll(V2, "fit")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    sid = [s for s, n in picks.items() if n.startswith("typical")][0]
    res = preprocess_bdf(TDBRAIN / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-restEC_eeg.bdf")
    fs = res.fs
    nper = int(4 * fs)
    starts = clean_epoch_starts(res.clean, nper, nper // 2)
    _, second = split_halves(starts, nper)
    seconds = 6.0
    n = int(seconds * fs)
    real = None
    lo = second[0] if len(second) else res.clean.size // 2
    for s0 in range(lo, res.clean.size - n, int(fs / 4)):  # first 6-s artefact-free stretch of the unseen half
        if res.clean[s0:s0 + n].all():
            real = res.data_uv[:, s0:s0 + n].T.astype(float)
            break
    if real is None:
        raise RuntimeError(f"no clean {seconds}-s stretch in the unseen half of {sid}")
    real = band(real, fs)
    rng = np.random.default_rng(3)
    i = index[sid]
    level = power(fit.csd[i]).mean()  # M5.3 has no absolute scale: match the fitted half's mean power
    s53 = P53[sid] * level / power(P53[sid]).mean()
    # check the sampler: the Welch spectrum of a long generated segment must reproduce the prediction
    from mdd_tvb.spectral_config import load_spectral_m5_config
    from mdd_tvb.spectral_features import estimate_cross_spectrum
    long = sample_from_csd(P54[sid], f, fs, int(240 * fs), np.random.default_rng(11))
    _, est, _ = estimate_cross_spectrum(long, fs, load_spectral_m5_config(ROOT / "configs/m5_spectral.toml").spectral)
    ratio = power(est) / power(P54[sid])
    NUMBERS["sampler_check_power_ratio"] = {"median": float(np.median(ratio)), "p5": float(np.percentile(ratio, 5)),
                                            "p95": float(np.percentile(ratio, 95))}
    gen54 = band(sample_from_csd(P54[sid] * 1e12, f, fs, n, rng), fs)
    gen53 = band(sample_from_csd(s53 * 1e12, f, fs, n, rng), fs)
    gen51 = m51_simulated_eeg(sid, seconds, fs)
    # M5.1 output is in model units: give it the fitted half's 2-40 Hz variance (per-channel mean)
    target_var = power(fit.csd[i]).sum(0).mean() * 1e12
    gen51 = gen51 * np.sqrt(target_var / gen51.var(0).mean())
    show = ["Fp1", "F3", "Fz", "F4", "T7", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "Oz", "O2"]
    idx = [labels.index(c) for c in show]
    spacing = 5 * np.median(np.std(real[:, idx], axis=0))
    t = np.arange(n) / fs
    fig, axes = plt.subplots(1, 4, figsize=(19, 7.8), sharey=True, constrained_layout=True)
    for ax, (name, x, color) in zip(axes, (("Real EEG, unseen half", real, INK),
                                           ("M5.1, simulated (neural part)", gen51, "#6F7F8C"),
                                           ("M5.3, generated", gen53, C_M53), ("M5.4, generated", gen54, C_M54))):
        for k, j in enumerate(idx):
            ax.plot(t, x[:, j] - k * spacing, color=color, lw=0.7)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, seconds)
        ax.spines["left"].set_visible(False)
        ax.grid(False)
    axes[0].set_yticks([-k * spacing for k in range(len(idx))], show)
    axes[0].tick_params(axis="y", length=0)
    bar = 20.0
    y0 = -(len(idx) + 0.2) * spacing
    axes[0].plot([0.1, 0.1], [y0, y0 + bar], color=INK, lw=2.2, clip_on=False)
    axes[0].text(0.2, y0 + bar / 2, f"{bar:.0f} µV", ha="left", va="center", fontsize=10, color=INK)
    fig.suptitle(f"Subject {sid} ({val.groups[i]}): 2–40 Hz, average reference, same scale", fontsize=12.5, color=INK)
    fig.savefig(OUT / "eeg_traces.png")
    plt.close(fig)
    CACHE["traces"] = {"sid": sid, "group": str(val.groups[i]), "fs": fs, "labels": labels, "real": real,
                       "M5.1": gen51, "M5.3": gen53, "M5.4": gen54}
    NUMBERS["traces"] = {"subject": sid, "real_rms_uv": float(real.std()), "m54_rms_uv": float(gen54.std()),
                         "m53_rms_uv": float(gen53.std()), "m51_rms_uv": float(gen51.std())}


def m51_simulated_eeg(sid, seconds, fs):
    """Stochastic simulation of this subject's M5.1 MAP candidate with the repository's JAX simulator."""
    from mdd_tvb.jax_backend import JaxDualBatch, run_dual_jansen_rit_jax
    from mdd_tvb.linear_spectral import CandidateLinearizer
    from mdd_tvb.spectral_config import load_spectral_m5_config
    k = None
    for fold in range(5):
        t = pd.read_csv(ROOT / f"outputs/m52_nested_baseline/outer_{fold}/subject_posteriors.csv")
        hit = t[(t.subject_id == sid) & (t.subject_split == "holdout")]
        if len(hit):
            k = int(hit.map_candidate_index.iloc[0])
    P = np.load(ROOT / "outputs/audit_claude/m51bank_parameters.npy")
    lin = CandidateLinearizer(load_spectral_m5_config(ROOT / "configs/m5_spectral.toml"))
    p = lin.parameters(P[k])
    batch = JaxDualBatch(weights=p.weights[None], tract_lengths_ms_at_unit_speed=lin.connectome.tract_lengths,
                         gain_matrix=lin.gain, regional_a=p.a[None], regional_b=p.b[None], regional_mu=p.mu[None],
                         regional_noise_nsig=np.asarray(p.noise_nsig)[None], global_coupling=np.array([p.global_coupling]),
                         speed_mm_per_ms=P[[k], 1], fast_ratio=np.array([p.fast_ratio]),
                         fast_fraction=np.array([p.fast_fraction]), noise_tau_ms=np.array([p.noise_tau_ms]),
                         seeds=np.array([7], dtype=np.int32))
    res = run_dual_jansen_rit_jax(batch, duration_ms=(seconds + 3.0) * 1000.0, transient_ms=2000.0, dt_ms=0.5,
                                  monitor_period_ms=1000.0 / fs, history_steps=int(250 / 2 / 0.5) + 2,
                                  precision="float64")
    x = np.asarray(res.eeg[0], float)
    x = x - x.mean(1, keepdims=True)  # average reference
    NUMBERS["m51_trace_candidate"] = {"subject": sid, "map_candidate_index": k}
    return band(x, fs)[-int(seconds * fs):]


# ---------------------------------------------------------------------------
# 6. what an M5.4 prediction is made of
# ---------------------------------------------------------------------------
def m54_components(sid):
    if ("components", sid) in CACHE:
        return CACHE[("components", sid)]
    import m54_core as M
    from m54_core import jax, jnp
    population = json.loads((ROOT / POP_M54).read_text())
    setup = M.make_setup("bem", population, use_pop=True)
    row = pd.read_csv(ROOT / FITS_M54).set_index("subject_id").loc[sid]
    theta = jnp.asarray([row[n] for n in setup.theta_names()])
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    fold = splits[(splits.subject_id == sid) & (splits.outer_role == "validation")].outer_fold.iloc[0]
    fr = splits[splits.outer_fold == fold]
    fit = coll(V2, "fit")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    keep = set(dev_ids())
    train = [index[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str) if s in index and s in keep]
    B = jnp.asarray(M.pooled_null([fit.csd[i] for i in train]))
    k = len(M.FREE_INDEX)
    C, common, _, _ = M.contributions(setup, M.full_u(setup, theta[:k]))
    z = theta[k:]
    d = setup.d
    beta = setup.mapping @ (z[:d] - jnp.mean(z[:d]))
    frac, expo = 0.95 * jax.nn.sigmoid(z[d]), 2.5 * jax.nn.sigmoid(z[d + 1])
    sfrac, sexp = 0.95 * jax.nn.sigmoid(z[d + 2]), 3.0 * jax.nn.sigmoid(z[d + 3])
    share, pshare = 0.95 * jax.nn.sigmoid(z[d + 4]), 0.95 * jax.nn.sigmoid(z[d + 5])
    freq = setup.freq
    diag = lambda a: jnp.mean(jnp.real(jnp.diagonal(a, axis1=1, axis2=2)))  # noqa: E731
    gains = jnp.einsum("k,kfcd->fcd", jnp.exp(beta), C)
    level = diag(gains)
    drive = common * (level / diag(common)) * share / (1.0 - share)
    level = diag(gains + drive)
    shape = freq ** (-expo); shape = shape / jnp.mean(shape)
    sshape = freq ** (-sexp); sshape = sshape / jnp.mean(sshape)
    parts = {"neural: network gains": gains, "neural: shared alpha drive": drive,
             "sensor noise": (level * frac / (1 - frac) * shape)[:, None, None] * jnp.eye(26)[None],
             "source background": (level * sfrac / (1 - sfrac) * sshape)[:, None, None] * setup.source_cov[None],
             "population background": B * (level / diag(B)) * pshare / (1 - pshare)}
    scale = float(np.exp(row["log_scale"]))
    parts = {n: np.asarray(v, np.complex128) * scale for n, v in parts.items()}
    total = sum(parts.values())
    saved = preds("M5.4")[sid]
    err = float(np.abs(power(total) - power(saved)).max() / power(saved).max())
    CACHE[("components", sid)] = (parts, err)
    return parts, err


def fig_decomposition(picks):
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    chosen = [s for s, n in picks.items() if not n.startswith("harder")]
    fig = plt.figure(figsize=(15, 4.3 * len(chosen)), layout="constrained")
    subs = fig.subfigures(len(chosen), 1, hspace=0.05)
    axes = np.stack([sub.subplots(1, 3, sharex=True) for sub in subs])
    colors = {"neural: network gains": C_M54, "neural: shared alpha drive": "#7FA8CC", "population background": C_POP,
              "source background": "#B9C4CC", "sensor noise": "#DDE3E9"}
    shares = {}
    for r, sid in enumerate(chosen):
        parts, err = m54_components(sid)
        shares[sid] = {n: float(power(v).mean() / sum(power(p).mean() for p in parts.values())) for n, v in parts.items()}
        shares[sid]["reconstruction_error"] = err
        for c, chans in enumerate((("O1", "Oz", "O2"), ("F3", "Fz", "F4"), ("T7", "T8"))):
            ax = axes[r, c]
            idx = [labels.index(ch) for ch in chans]
            base = np.zeros_like(f)
            for name in ("sensor noise", "source background", "population background", "neural: shared alpha drive",
                         "neural: network gains"):
                y = power(parts[name])[:, idx].mean(1) * 1e12
                ax.fill_between(f, base, base + y, color=colors[name], lw=0, label=name)
                base = base + y
            meas = power(val.csd[index[sid]])[:, idx].mean(1) * 1e12
            ax.plot(f, meas, color=INK, lw=2.2, label="measured, unseen half")
            ax.set_xlim(2, 40)
            ax.set_ylim(0, max(meas.max(), base.max()) * 1.08)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_title(" / ".join(chans), fontsize=11, fontweight="normal", color=MUTED)
            if c == 0:
                ax.set_ylabel("power (µV²/Hz)")
            if r == len(chosen) - 1:
                ax.set_xlabel("frequency (Hz)")
        subs[r].suptitle(f"{picks[sid]}: subject {sid}", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    handles, labs = axes[0, 2].get_legend_handles_labels()
    axes[0, 2].legend(handles[::-1], labs[::-1], loc="upper right", fontsize=9.5)
    NUMBERS["m54_power_shares"] = shares
    fig.savefig(OUT / "m54_decomposition.png")
    plt.close(fig)


def alpha_peak_stats():
    """Occipital alpha-peak height (relative power, 8-12 Hz max) and frequency: prediction vs unseen half."""
    out = {}
    for data, emp, models in (("v1", V1, ("M5.1 (v1)", "M5.3 (v1)")), ("v2", V2, ("M5.3", "M5.4"))):
        val = coll(emp, "validation")
        labels = list(val.channel_names.astype(str))
        f = np.asarray(val.frequency_hz, float)
        index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
        sel = (f >= 7) & (f <= 13)
        P = {m: preds(m) for m in models}
        ids = [s for s in val.subject_ids.astype(str) if all(s in P[m] for m in models)]
        if data == "v2":
            keep = set(dev_ids())
            ids = [s for s in ids if s in keep]
        meas = group_curve(val.csd[[index[s] for s in ids]], labels, ("O1", "Oz", "O2"))
        for m in models:
            pred = group_curve(np.stack([P[m][s] for s in ids]), labels, ("O1", "Oz", "O2"))
            ratio = pred[:, sel].max(1) / meas[:, sel].max(1)
            ferr = np.abs(f[sel][pred[:, sel].argmax(1)] - f[sel][meas[:, sel].argmax(1)])
            out[f"{data}|{m}"] = {"peak_height_ratio_median": float(np.median(ratio)),
                                  "peak_height_ratio_iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
                                  "fraction_peak_below_half": float(np.mean(ratio < 0.5)),
                                  "peak_bin_error_hz_median": float(np.median(ferr)), "n": len(ids)}
    NUMBERS["occipital_alpha_peak"] = out


def fig_group_effect_maps():
    """Healthy-MDD difference maps: measured, predicted, and the batch difference without depressed subjects."""
    import mne
    from m54_evaluate import features
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    T = T[T.sessID == 1].drop_duplicates("TDBRAIN_ID").set_index("TDBRAIN_ID")
    keep = set(dev_ids())
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    info = _info(labels)
    P53, P54 = preds("M5.3"), preds("M5.4")
    ids = [s for s in val.subject_ids.astype(str) if s in keep and s in P53 and s in P54]
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    grp = np.asarray([str(val.groups[index[s]]) for s in ids])
    # batch effect from first halves of non-depressed cohorts only
    rows = []
    for cohort in ("dev", "controls", "smc"):
        c = coll(f"outputs/preproc_v2/{cohort}_restEC/empirical", "fit")
        for s, g, csd in zip(c.subject_ids.astype(str), c.groups.astype(str), c.csd):
            if g == "MDD" or (cohort == "dev" and s not in keep):
                continue
            rows.append((s, g, csd))
    early = np.asarray([int(s[4:]) < 88000000 for s, _, _ in rows])
    cov_b = np.column_stack([T.loc[[s for s, _, _ in rows], "age"].astype(float), T.loc[[s for s, _, _ in rows], "gender"].astype(float)])
    cov_d = np.column_stack([T.loc[ids, "age"].astype(float), T.loc[ids, "gender"].astype(float)])

    def resid(A, cov):
        X = np.column_stack([np.ones(len(cov)), cov])
        b, *_ = np.linalg.lstsq(X, A, rcond=None)
        return A - X[:, 1:] @ b[1:]

    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4), layout="constrained")
    fig.suptitle("Healthy–MDD difference in the unseen halves, and the acquisition-batch difference", fontsize=13, fontweight="bold", color=INK)
    stats = {}
    for r, (key, bname) in enumerate((("topo_alpha", "alpha 8–12 Hz"), ("topo_beta", "beta 13–19 Hz"))):
        feat = lambda csd: features(csd, f, labels)[key]  # noqa: E731
        meas = resid(np.stack([feat(val.csd[index[s]]) for s in ids]), cov_d)
        p53 = resid(np.stack([feat(P53[s]) for s in ids]), cov_d)
        p54 = resid(np.stack([feat(P54[s]) for s in ids]), cov_d)
        bat = resid(np.stack([feat(csd) for _, _, csd in rows]), cov_b)
        eff = lambda A: A[grp == "MDD"].mean(0) - A[grp == "Healthy"].mean(0)  # noqa: E731
        maps = [("MDD − Healthy\nmeasured, unseen half", eff(meas)), ("MDD − Healthy\npredicted by M5.3", eff(p53)),
                ("MDD − Healthy\npredicted by M5.4", eff(p54)),
                ("Late − early batch\nno depressed subjects", bat[~early].mean(0) - bat[early].mean(0))]
        CACHE[("effect_maps", bname)] = maps
        lim = max(np.abs(m).max() for _, m in maps)
        for c, (name, m) in enumerate(maps):
            im, _ = mne.viz.plot_topomap(m, info, axes=axes[r, c], show=False, cmap="RdBu_r", vlim=(-lim, lim),
                                         contours=0, sensors=True)
            if r == 0:
                axes[r, c].set_title(name, fontsize=11.5, pad=14)
            if c > 0:
                rr = float(np.corrcoef(maps[0][1], m)[0, 1])
                stats[f"{bname}|{name.splitlines()[1]}"] = rr
                axes[r, c].text(0.5, -0.1, f"r = {rr:.2f} with measured difference", transform=axes[r, c].transAxes,
                                ha="center", fontsize=10, color=MUTED)
        axes[r, 0].text(-0.25, 0.5, bname, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=12,
                        fontweight="bold", color=INK, rotation=90)
        fig.colorbar(im, ax=axes[r, :], shrink=0.75, label="difference in centred log10 power")
    NUMBERS["group_effect_maps_r"] = stats
    fig.savefig(OUT / "group_effect_maps.png")
    plt.close(fig)



# ---------------------------------------------------------------------------
# slide-sized versions (drawn at their exact size on a 13.33 x 7.5 in slide)
# ---------------------------------------------------------------------------
SLIDE_RC = {"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "savefig.dpi": 250}
LEFT = (8.2, 4.85)
WIDE = (12.3, 4.85)
TWO = {"occipital (O1, Oz, O2)": ("O1", "Oz", "O2"), "frontal (F3, Fz, F4)": ("F3", "Fz", "F4")}


def slide_group_spectra():
    with plt.rc_context(SLIDE_RC):
        fig = plt.figure(figsize=LEFT, layout="constrained")
        subs = fig.subfigures(2, 1, hspace=0.03)
        for r, (label, emp, models, keep) in enumerate((
                ("Original preprocessing (262 subjects)", V1, [("M5.1 (v1)", C_M51, "--"), ("M5.3 (v1)", C_M53, "-")], None),
                ("Re-cleaned EEG (252 subjects)", V2, [("M5.3", C_M53, "-"), ("M5.4", C_M54, "-")], set(dev_ids())))):
            subs[r].suptitle(label, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
            axrow = subs[r].subplots(1, 2)
            val = coll(emp, "validation")
            labels = list(val.channel_names.astype(str))
            f = np.asarray(val.frequency_hz, float)
            P = {m: preds(m) for m, _, _ in models}
            ids = [x for x in val.subject_ids.astype(str) if all(x in P[m] for m, _, _ in models) and (keep is None or x in keep)]
            index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
            meas = val.csd[[index[x] for x in ids]]
            for c, (gname, chans) in enumerate(TWO.items()):
                ax = axrow[c]
                y = group_curve(meas, labels, chans).mean(0)
                ax.plot(f, y, color=INK, lw=2.4, label="measured, unseen half")
                for name, color, ls in models:
                    yp = group_curve(np.stack([P[name][x] for x in ids]), labels, chans).mean(0)
                    ax.plot(f, yp, color=color, lw=2, ls=ls, label=name.replace(" (v1)", ""))
                style_spectrum_axis(ax, ylabel=c == 0)
                ax.set_title(gname, fontsize=10.5, fontweight="normal", color=MUTED)
                if r == 0:
                    ax.set_xlabel("")
            axrow[1].legend(loc="upper right")
        fig.savefig(OUT / "slide_group_spectra.png")
        plt.close(fig)


def slide_subject_spectra(picks):
    fit, val = coll(V2, "fit"), coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 3, figsize=LEFT, sharex=True, layout="constrained")
        for c, (sid, name) in enumerate(picks.items()):
            i = index[sid]
            for r, (gname, chans) in enumerate(TWO.items()):
                ax = axes[r, c]
                ax.plot(f, group_curve(fit.csd[i], labels, chans), color=C_FIT, lw=1.2, label="fitted half")
                ax.plot(f, group_curve(val.csd[i], labels, chans), color=INK, lw=2, label="unseen half")
                ax.plot(f, group_curve(P53[sid], labels, chans), color=C_M53, lw=1.7, label="M5.3")
                ax.plot(f, group_curve(P54[sid], labels, chans), color=C_M54, lw=1.7, label="M5.4")
                style_spectrum_axis(ax, ylabel=c == 0)
                if r == 0:
                    ax.set_title(f"{name.split(' (')[0].capitalize()} ({val.groups[i]})\n{gname.split(' (')[0]}", fontsize=10.5)
                else:
                    ax.set_title(gname.split(" (")[0], fontsize=10.5, fontweight="normal", color=MUTED)
                if r == 0:
                    ax.set_xlabel("")
        axes[0, 2].legend(loc="upper right", fontsize=9)
        fig.savefig(OUT / "slide_subject_spectra.png")
        plt.close(fig)


def slide_topomaps(picks):
    import mne
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    info = _info(labels)
    index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
    P53, P54 = preds("M5.3"), preds("M5.4")
    ids = [x for x in dev_ids() if x in index and x in P53 and x in P54]
    cols = [("Group mean", ids)] + [(n.split(" (")[0].capitalize(), [sid]) for sid, n in picks.items()]
    rows = [("Measured", lambda x: val.csd[index[x]]), ("M5.3", lambda x: P53[x]), ("M5.4", lambda x: P54[x])]
    vals = [[topo_values(np.stack([get(x) for x in sub]), f, 8, 12).mean(0) for _, sub in cols] for _, get in rows]
    lim = np.percentile(np.abs(np.concatenate([np.ravel(v) for row in vals for v in row])), 98)
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(3, len(cols), figsize=LEFT, layout="constrained")
        for r, (rname, _) in enumerate(rows):
            for c, (cname, _) in enumerate(cols):
                im, _ = mne.viz.plot_topomap(vals[r][c], info, axes=axes[r, c], show=False, cmap="RdBu_r",
                                             vlim=(-lim, lim), contours=0, sensors=False)
                if r == 0:
                    axes[r, c].set_title(cname, fontsize=11)
                if r > 0:
                    axes[r, c].text(0.5, -0.12, f"r = {np.corrcoef(vals[0][c], vals[r][c])[0, 1]:.2f}",
                                    transform=axes[r, c].transAxes, ha="center", fontsize=9.5, color=MUTED)
            axes[r, 0].text(-0.12, 0.5, rname, transform=axes[r, 0].transAxes, ha="right", va="center",
                            fontsize=11, fontweight="bold", color=INK)
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
        cb.set_label("centred log10 alpha power")
        fig.savefig(OUT / "slide_topomaps_alpha.png")
        plt.close(fig)


def slide_coherence(picks):
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    xyz = pd.read_csv(ROOT / "data/sensors/TDBRAIN_Table3_electrode_coordinates.csv").set_index("label").loc[labels, ["x_mm", "y_mm", "z_mm"]].to_numpy()
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(labels), 1)
    index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
    P51, P53, P54 = preds("M5.1 (v1)"), preds("M5.3"), preds("M5.4")
    ids = [x for x in dev_ids() if x in index and x in P53 and x in P54 and x in P51]
    bins = np.arange(20, 220, 20)
    with plt.rc_context(SLIDE_RC):
        fig = plt.figure(figsize=LEFT, layout="constrained")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1])
        ax = fig.add_subplot(gs[:, 0])
        for name, arr, color, marker in (("measured", np.stack([val.csd[index[x]] for x in ids]), INK, "o"),
                                         ("M5.1", np.stack([P51[x] for x in ids]), C_M51, "s"),
                                         ("M5.3", np.stack([P53[x] for x in ids]), C_M53, "^"),
                                         ("M5.4", np.stack([P54[x] for x in ids]), C_M54, "D")):
            coh = real_coherence(arr, f, 8, 12).mean(0)[iu]
            ax.scatter(dist[iu], coh, s=5, color=color, alpha=0.3, lw=0)
            which = np.digitize(dist[iu], bins)
            mids = [(bins[k - 1] + 10, coh[which == k].mean()) for k in range(1, len(bins)) if np.any(which == k)]
            ax.plot(*zip(*mids), color=color, lw=2.2, marker=marker, ms=5, label=name)
        ax.axhline(0, color=GRID, lw=1)
        ax.set_xlabel("electrode distance (mm)")
        ax.set_ylabel("alpha zero-lag coherence")
        ax.set_title("Group mean, 252 subjects", fontsize=11)
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(loc="upper right")
        sid = [x for x, n in picks.items() if n.startswith("typical")][0]
        for k, (name, csd) in enumerate((("Measured", val.csd[index[sid]]), ("M5.1", P51[sid]), ("M5.3", P53[sid]), ("M5.4", P54[sid]))):
            a = fig.add_subplot(gs[k // 2, 1 + k % 2])
            im = a.imshow(real_coherence(csd, f, 8, 12), cmap="RdBu_r", vmin=-1, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{name}, one subject", fontsize=10.5)
        fig.colorbar(im, ax=fig.axes[1:], shrink=0.7, label="coherence")
        fig.savefig(OUT / "slide_coherence.png")
        plt.close(fig)


def slide_traces():
    tr = CACHE["traces"]
    labels, fs = tr["labels"], tr["fs"]
    show = ["Fp1", "Fz", "T7", "C3", "Cz", "P3", "Pz", "O1", "Oz"]
    idx = [labels.index(c) for c in show]
    spacing = 5 * np.median(np.std(tr["real"][:, idx], axis=0))
    seconds = 4.0
    n = int(seconds * fs)
    t = np.arange(n) / fs
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, 4, figsize=WIDE, sharey=True, layout="constrained")
        for ax, (name, key, color) in zip(axes, (("Real EEG (unseen half)", "real", INK), ("M5.1, simulated", "M5.1", "#6F7F8C"),
                                                 ("M5.3, generated", "M5.3", C_M53), ("M5.4, generated", "M5.4", C_M54))):
            x = tr[key][:n]
            for k, j in enumerate(idx):
                ax.plot(t, x[:, j] - k * spacing, color=color, lw=0.8)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("time (s)")
            ax.set_xlim(0, seconds)
            ax.spines["left"].set_visible(False)
        axes[0].set_yticks([-k * spacing for k in range(len(idx))], show)
        axes[0].tick_params(axis="y", length=0)
        bar = 20.0
        y0 = -(len(idx) - 0.4) * spacing
        axes[3].plot([seconds * 0.98] * 2, [y0 - bar, y0], color=INK, lw=2, clip_on=False)
        axes[3].text(seconds * 0.96, y0 - bar / 2, "20 µV", ha="right", va="center", fontsize=9.5, color=INK)
        fig.savefig(OUT / "slide_eeg_traces.png")
        plt.close(fig)


def slide_decomposition(picks):
    val = coll(V2, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
    chosen = [x for x, n in picks.items() if not n.startswith("harder")]
    colors = {"neural: network gains": C_M54, "neural: shared alpha drive": "#7FA8CC", "population background": C_POP,
              "source background": "#B9C4CC", "sensor noise": "#DDE3E9"}
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.85), sharex=True, layout="constrained")
        for r, sid in enumerate(chosen):
            parts, _ = m54_components(sid)
            for c, (gname, chans) in enumerate(TWO.items()):
                ax = axes[r, c]
                ix = [labels.index(ch) for ch in chans]
                base = np.zeros_like(f)
                for name in ("sensor noise", "source background", "population background", "neural: shared alpha drive",
                             "neural: network gains"):
                    y = power(parts[name])[:, ix].mean(1) * 1e12
                    ax.fill_between(f, base, base + y, color=colors[name], lw=0, label=name)
                    base = base + y
                meas = power(val.csd[index[sid]])[:, ix].mean(1) * 1e12
                ax.plot(f, meas, color=INK, lw=2, label="measured, unseen half")
                ax.set_xlim(2, 30)
                ax.set_ylim(0, max(meas.max(), base.max()) * 1.08)
                ax.grid(True, color=GRID, lw=0.6)
                ax.set_title(f"{picks[sid].split(' (')[0].capitalize()}, {gname.split(' (')[0]}", fontsize=10.5)
                if c == 0:
                    ax.set_ylabel("µV²/Hz")
                if r == 1:
                    ax.set_xlabel("frequency (Hz)")
        handles, labs = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles[::-1], labs[::-1], loc="outside right center", fontsize=9)
        fig.savefig(OUT / "slide_decomposition.png")
        plt.close(fig)


def slide_group_effect_maps():
    import mne
    val = coll(V2, "validation")
    info = _info(list(val.channel_names.astype(str)))
    titles = ["MDD − Healthy\nmeasured", "MDD − Healthy\nM5.3", "MDD − Healthy\nM5.4", "Late − early batch\nno MDD subjects"]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.85), layout="constrained")
        for r, bname in enumerate(("alpha 8–12 Hz", "beta 13–19 Hz")):
            maps = CACHE[("effect_maps", bname)]
            lim = max(np.abs(m).max() for _, m in maps)
            for c, (_, m) in enumerate(maps):
                im, _ = mne.viz.plot_topomap(m, info, axes=axes[r, c], show=False, cmap="RdBu_r", vlim=(-lim, lim),
                                             contours=0, sensors=False)
                if r == 0:
                    axes[r, c].set_title(titles[c], fontsize=10.5)
                if c > 0:
                    axes[r, c].text(0.5, -0.1, f"r = {np.corrcoef(maps[0][1], m)[0, 1]:.2f}", transform=axes[r, c].transAxes,
                                    ha="center", fontsize=9.5, color=MUTED)
            axes[r, 0].text(-0.12, 0.5, bname.split()[0], transform=axes[r, 0].transAxes, ha="right", va="center",
                            fontsize=11, fontweight="bold", color=INK)
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
        cb.set_label("difference (centred log10 power)")
        fig.savefig(OUT / "slide_group_effect_maps.png")
        plt.close(fig)


def main():
    picks = pick_subjects()
    alpha_peak_stats()
    fig_group_effect_maps()
    for step in (fig_group_spectra, lambda: fig_subject_spectra(picks), lambda: fig_topomaps(picks),
                 lambda: fig_coherence(picks), lambda: fig_traces(picks), lambda: fig_decomposition(picks)):
        step()
        print("done", getattr(step, "__name__", "step"), flush=True)
    for step in (slide_group_spectra, lambda: slide_subject_spectra(picks), lambda: slide_topomaps(picks),
                 lambda: slide_coherence(picks), slide_traces, lambda: slide_decomposition(picks), slide_group_effect_maps):
        step()
    print("slide figures done", flush=True)
    (OUT / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float))
    print(json.dumps(NUMBERS, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
