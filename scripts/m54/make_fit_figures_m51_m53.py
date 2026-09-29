"""Fit figures for M5.1 and M5.3 only (no M5.4), for presenting M5.3 on its own.

Same figures as make_fit_figures.py (spectra, scalp maps, coherence, EEG traces), all
on the original preprocessing (v1) that M5.1 was fitted to, so both models are compared
on the same data and against each subject's *unseen* second half:
    M5.1  bank posterior (outputs/m54_eval/m51_v1, m51_ext)
    M5.3  frozen audit model, TVB lead field, refined (final_tvb_refined, m53_ext)
Rows cover the 262 development subjects and the 164 external rTMS patients.  Example
subjects are chosen by M5.3's own unseen spectrum error (10th, 50th, 90th percentile).

    .conda/python.exe scripts/m54/make_fit_figures_m51_m53.py
Output: docs/figures/fit_examples_m51_m53/slide_*.png and numbers.json.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fit_figures as F  # noqa: E402  (helpers, colours, sampler and the M5.1 simulator)
from make_fit_figures import (INK, MUTED, GRID, C_M53, ROOT, SLIDE_RC, LEFT, WIDE, TWO, coll, group_curve, power,  # noqa: E402
                              real_coherence, sample_from_csd, band, style_spectrum_axis, topo_values, _info)

OUT = ROOT / "docs/figures/fit_examples_m51_m53"
OUT.mkdir(parents=True, exist_ok=True)
C_M51 = "#7C8D99"      # dashed slate: M5.1
C_FIT = "#C5CED6"      # light grey: the fitted first half
V1 = "outputs/m5_spectral_m51_production/empirical"
V1X = "outputs/external/rtms_restEC/empirical"
PRED = {"M5.1": "outputs/m54_eval/m51_v1/predictions.npz",
        "M5.3": "outputs/linear_regime/kaggle/final/final_tvb_refined/predictions.npz",
        "M5.1 ext": "outputs/m54_eval/m51_ext/predictions.npz",
        "M5.3 ext": "outputs/m54_eval/m53_ext/predictions.npz"}
BLOCKS_M53 = "outputs/m54_eval/v1_m51_vs_m53/blocks_M53_tvb.csv"
NUMBERS: dict = {}
CACHE: dict = {}


def preds(key):
    if key not in CACHE:
        with np.load(ROOT / PRED[key]) as p:
            CACHE[key] = dict(zip(p["subject_ids"].astype(str), p["csd"].astype(np.complex128)))
    return CACHE[key]


def common(val, *keys):
    ids = [s for s in val.subject_ids.astype(str) if all(s in preds(k) for k in keys)]
    index = {s: i for i, s in enumerate(val.subject_ids.astype(str))}
    return ids, index


def pick_subjects():
    t = pd.read_csv(ROOT / BLOCKS_M53).set_index("subject_id")
    both = set(preds("M5.1")) & set(preds("M5.3")) & set(t.index)
    t = t.loc[sorted(both)]
    picks = {}
    for q, name in ((0.1, "better fit (10th percentile)"), (0.5, "typical fit (median)"), (0.9, "harder fit (90th percentile)")):
        target = t.spectrum_ratio.quantile(q)
        picks[(t.spectrum_ratio - target).abs().idxmin()] = name
    NUMBERS["example_subjects"] = {s: {"label": n, "group": t.loc[s, "group"], "m53_spectrum_ratio": float(t.loc[s, "spectrum_ratio"])}
                                   for s, n in picks.items()}
    return picks


# ---------------------------------------------------------------------------
# numbers shown next to the figures
# ---------------------------------------------------------------------------
def alpha_peak_stats():
    out = {}
    for cohort, emp, models in (("dev", V1, ("M5.1", "M5.3")), ("ext", V1X, ("M5.1 ext", "M5.3 ext"))):
        val = coll(emp, "validation")
        labels = list(val.channel_names.astype(str))
        f = np.asarray(val.frequency_hz, float)
        ids, index = common(val, *models)
        sel = (f >= 7) & (f <= 13)
        meas = group_curve(val.csd[[index[s] for s in ids]], labels, ("O1", "Oz", "O2"))
        for m in models:
            pred = group_curve(np.stack([preds(m)[s] for s in ids]), labels, ("O1", "Oz", "O2"))
            ratio = pred[:, sel].max(1) / meas[:, sel].max(1)
            out[f"{cohort}|{m.split()[0]}"] = {"peak_height_ratio_median": float(np.median(ratio)),
                                              "fraction_peak_below_half": float(np.mean(ratio < 0.5)), "n": len(ids)}
    NUMBERS["occipital_alpha_peak"] = out


# ---------------------------------------------------------------------------
# slide figures
# ---------------------------------------------------------------------------
def slide_group_spectra():
    summary = {}
    with plt.rc_context(SLIDE_RC):
        fig = plt.figure(figsize=LEFT, layout="constrained")
        subs = fig.subfigures(2, 1, hspace=0.03)
        for r, (label, emp, m51, m53) in enumerate((("Development, 262 subjects", V1, "M5.1", "M5.3"),
                                                    ("External rTMS patients, 164 (never used for fitting)", V1X,
                                                     "M5.1 ext", "M5.3 ext"))):
            subs[r].suptitle(label, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
            axrow = subs[r].subplots(1, 2)
            val = coll(emp, "validation")
            labels = list(val.channel_names.astype(str))
            f = np.asarray(val.frequency_hz, float)
            ids, index = common(val, m51, m53)
            meas = val.csd[[index[s] for s in ids]]
            for c, (gname, chans) in enumerate(TWO.items()):
                ax = axrow[c]
                y = group_curve(meas, labels, chans).mean(0)
                ax.plot(f, y, color=INK, lw=2.4, label="measured, unseen half")
                for name, color, ls in ((m51, C_M51, "--"), (m53, C_M53, "-")):
                    yp = group_curve(np.stack([preds(name)[s] for s in ids]), labels, chans).mean(0)
                    ax.plot(f, yp, color=color, lw=2, ls=ls, label=name.split()[0])
                    summary[f"{'ext' if 'ext' in name else 'dev'}|{gname.split()[0]}|{name.split()[0]}"] = \
                        float(np.corrcoef(np.log(y), np.log(yp))[0, 1])
                style_spectrum_axis(ax, ylabel=c == 0)
                ax.set_title(gname, fontsize=10.5, fontweight="normal", color=MUTED)
                if r == 0:
                    ax.set_xlabel("")
            if r == 0:
                axrow[1].legend(loc="upper right")
        fig.savefig(OUT / "slide_group_spectra.png")
        plt.close(fig)
    NUMBERS["group_spectra_log_shape_r"] = summary


def slide_subject_spectra(picks):
    fit, val = coll(V1, "fit"), coll(V1, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    index = {x: i for i, x in enumerate(val.subject_ids.astype(str))}
    P51, P53 = preds("M5.1"), preds("M5.3")
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, 3, figsize=LEFT, sharex=True, layout="constrained")
        for c, (sid, name) in enumerate(picks.items()):
            i = index[sid]
            for r, (gname, chans) in enumerate(TWO.items()):
                ax = axes[r, c]
                ax.plot(f, group_curve(fit.csd[i], labels, chans), color=C_FIT, lw=1.3, label="fitted half")
                ax.plot(f, group_curve(val.csd[i], labels, chans), color=INK, lw=2, label="unseen half")
                ax.plot(f, group_curve(P51[sid], labels, chans), color=C_M51, lw=1.7, ls="--", label="M5.1")
                ax.plot(f, group_curve(P53[sid], labels, chans), color=C_M53, lw=1.7, label="M5.3")
                style_spectrum_axis(ax, ylabel=c == 0)
                if r == 0:
                    ax.set_title(f"{name.split(' (')[0].capitalize()} ({val.groups[i]})\n{gname.split(' (')[0]}", fontsize=10.5)
                    ax.set_xlabel("")
                else:
                    ax.set_title(gname.split(" (")[0], fontsize=10.5, fontweight="normal", color=MUTED)
        axes[0, 2].legend(loc="upper right", fontsize=9)
        fig.savefig(OUT / "slide_subject_spectra.png")
        plt.close(fig)


def slide_topomaps(picks):
    import mne
    val = coll(V1, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    info = _info(labels)
    ids, index = common(val, "M5.1", "M5.3")
    P51, P53 = preds("M5.1"), preds("M5.3")
    cols = [("Group mean", ids)] + [(n.split(" (")[0].capitalize(), [sid]) for sid, n in picks.items()]
    rows = [("Measured", lambda x: val.csd[index[x]]), ("M5.1", lambda x: P51[x]), ("M5.3", lambda x: P53[x])]
    vals = [[topo_values(np.stack([get(x) for x in sub]), f, 8, 12).mean(0) for _, sub in cols] for _, get in rows]
    lim = np.percentile(np.abs(np.concatenate([np.ravel(v) for row in vals for v in row])), 98)
    corr = {}
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(3, len(cols), figsize=LEFT, layout="constrained")
        for r, (rname, _) in enumerate(rows):
            for c, (cname, _) in enumerate(cols):
                im, _ = mne.viz.plot_topomap(vals[r][c], info, axes=axes[r, c], show=False, cmap="RdBu_r",
                                             vlim=(-lim, lim), contours=0, sensors=False)
                if r == 0:
                    axes[r, c].set_title(cname, fontsize=11)
                if r > 0:
                    rr = float(np.corrcoef(vals[0][c], vals[r][c])[0, 1])
                    corr[f"{rname}|{cname}"] = rr
                    axes[r, c].text(0.5, -0.12, f"r = {rr:.2f}", transform=axes[r, c].transAxes, ha="center",
                                    fontsize=9.5, color=MUTED)
            axes[r, 0].text(-0.12, 0.5, rname, transform=axes[r, 0].transAxes, ha="right", va="center",
                            fontsize=11, fontweight="bold", color=INK)
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
        cb.set_label("centred log10 alpha power")
        fig.savefig(OUT / "slide_topomaps_alpha.png")
        plt.close(fig)
    # the typical-subject correlation over all development subjects, not only the three shown
    per = {m: [float(np.corrcoef(topo_values(val.csd[index[s]], f, 8, 12), topo_values(preds(m)[s], f, 8, 12))[0, 1])
               for s in ids] for m in ("M5.1", "M5.3")}
    corr.update({f"{m}|median over subjects": float(np.median(v)) for m, v in per.items()})
    NUMBERS["topomap_r_alpha"] = corr


def slide_coherence(picks):
    val = coll(V1, "validation")
    labels = list(val.channel_names.astype(str))
    f = np.asarray(val.frequency_hz, float)
    xyz = pd.read_csv(ROOT / "data/sensors/TDBRAIN_Table3_electrode_coordinates.csv").set_index("label").loc[labels, ["x_mm", "y_mm", "z_mm"]].to_numpy()
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    iu = np.triu_indices(len(labels), 1)
    ids, index = common(val, "M5.1", "M5.3")
    P51, P53 = preds("M5.1"), preds("M5.3")
    bins = np.arange(20, 220, 20)
    stats = {}
    with plt.rc_context(SLIDE_RC):
        fig = plt.figure(figsize=LEFT, layout="constrained")
        gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
        for c, (bname, (lo, hi)) in enumerate((("alpha 8–12 Hz", (8, 12)), ("beta 13–19 Hz", (13, 19)))):
            ax = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
            ref = None
            for name, arr, color, marker in (("measured", np.stack([val.csd[index[x]] for x in ids]), INK, "o"),
                                             ("M5.1", np.stack([P51[x] for x in ids]), C_M51, "s"),
                                             ("M5.3", np.stack([P53[x] for x in ids]), C_M53, "^")):
                coh = real_coherence(arr, f, lo, hi).mean(0)[iu]
                ax.scatter(dist[iu], coh, s=5, color=color, alpha=0.3, lw=0)
                which = np.digitize(dist[iu], bins)
                mids = [(bins[k - 1] + 10, coh[which == k].mean()) for k in range(1, len(bins)) if np.any(which == k)]
                ax.plot(*zip(*mids), color=color, lw=2.2, marker=marker, ms=5, label=name)
                if ref is None:
                    ref = coh
                else:
                    stats[f"{bname}|{name}"] = {"rmse_vs_measured": float(np.sqrt(np.mean((coh - ref) ** 2))),
                                                "r_vs_measured": float(np.corrcoef(coh, ref)[0, 1])}
            ax.axhline(0, color=GRID, lw=1)
            ax.set_xlabel("electrode distance (mm)")
            if c == 0:
                ax.set_ylabel("zero-lag coherence")
                ax.legend(loc="upper right")
            ax.set_title(f"{bname}, group mean", fontsize=10.5)
            ax.grid(True, color=GRID, lw=0.6)
        sid = [x for x, n in picks.items() if n.startswith("typical")][0]
        mats = []
        for k, (name, csd) in enumerate((("Measured", val.csd[index[sid]]), ("M5.1", P51[sid]), ("M5.3", P53[sid]))):
            a = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
            im = a.imshow(real_coherence(csd, f, 8, 12), cmap="RdBu_r", vmin=-1, vmax=1)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(f"{name}, one subject, alpha", fontsize=10.5)
            mats.append(a)
        fig.colorbar(im, ax=mats, shrink=0.8, label="coherence")
        fig.savefig(OUT / "slide_coherence.png")
        plt.close(fig)
    NUMBERS["coherence_vs_measured"] = stats


def traces(picks):
    """Real EEG (unseen half), M5.1 simulated and M5.3 generated, for the typical subject."""
    from mdd_tvb.preprocess_v2 import clean_epoch_starts, preprocess_bdf, split_halves
    fit = coll(V1, "fit")
    labels = list(fit.channel_names.astype(str))
    f = np.asarray(fit.frequency_hz, float)
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    sid = [s for s, n in picks.items() if n.startswith("typical")][0]
    res = preprocess_bdf(F.TDBRAIN / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-restEC_eeg.bdf")
    fs = res.fs
    nper = int(4 * fs)
    _, second = split_halves(clean_epoch_starts(res.clean, nper, nper // 2), nper)
    seconds = 4.0
    n = int(seconds * fs)
    real = None
    lo = second[0] if len(second) else res.clean.size // 2
    for s0 in range(lo, res.clean.size - n, int(fs / 4)):
        if res.clean[s0:s0 + n].all():
            real = band(res.data_uv[:, s0:s0 + n].T.astype(float), fs)
            break
    if real is None:
        raise RuntimeError(f"no clean {seconds}-s stretch in the unseen half of {sid}")
    i = index[sid]
    level = power(fit.csd[i]).mean()             # neither model has an absolute scale: match the fitted half's power
    P53 = preds("M5.3")[sid]
    gen53 = band(sample_from_csd(P53 * level / power(P53).mean() * 1e12, f, fs, n, np.random.default_rng(3)), fs)
    gen51 = F.m51_simulated_eeg(sid, seconds, fs)
    target_var = power(fit.csd[i]).sum(0).mean() * 1e12
    gen51 = gen51 * np.sqrt(target_var / gen51.var(0).mean())
    NUMBERS["traces"] = {"subject": sid, "group": str(fit.groups[i]), "real_rms_uv": float(real.std()),
                         "m51_rms_uv": float(gen51.std()), "m53_rms_uv": float(gen53.std()),
                         "m51_candidate": F.NUMBERS.get("m51_trace_candidate")}
    show = ["Fp1", "Fz", "T7", "C3", "Cz", "P3", "Pz", "O1", "Oz"]
    idx = [labels.index(c) for c in show]
    spacing = 5 * np.median(np.std(real[:, idx], axis=0))
    t = np.arange(n) / fs
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, 3, figsize=WIDE, sharey=True, layout="constrained")
        for ax, (name, x, color) in zip(axes, (("Real EEG (unseen half)", real, INK), ("M5.1, simulated", gen51, "#6F7F8C"),
                                               ("M5.3, generated", gen53, C_M53))):
            for k, j in enumerate(idx):
                ax.plot(t, x[:n, j] - k * spacing, color=color, lw=0.8)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("time (s)")
            ax.set_xlim(0, seconds)
            ax.spines["left"].set_visible(False)
        axes[0].set_yticks([-k * spacing for k in range(len(idx))], show)
        axes[0].tick_params(axis="y", length=0)
        bar = 20.0
        y0 = -(len(idx) - 0.4) * spacing
        axes[2].plot([seconds * 0.98] * 2, [y0 - bar, y0], color=INK, lw=2, clip_on=False)
        axes[2].text(seconds * 0.96, y0 - bar / 2, "20 µV", ha="right", va="center", fontsize=9.5, color=INK)
        fig.savefig(OUT / "slide_eeg_traces.png")
        plt.close(fig)


def main():
    picks = pick_subjects()
    alpha_peak_stats()
    for step in (slide_group_spectra, lambda: slide_subject_spectra(picks), lambda: slide_topomaps(picks),
                 lambda: slide_coherence(picks), lambda: traces(picks)):
        step()
        print("done", flush=True)
    (OUT / "numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float), encoding="utf-8")
    print(json.dumps(NUMBERS, indent=1, default=float))


if __name__ == "__main__":
    main()
