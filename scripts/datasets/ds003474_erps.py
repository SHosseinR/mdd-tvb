"""Event-related potentials of ds003474 (probabilistic selection; same people as ds003478).

Per subject: the v2 artefact-aware cleaning of the whole task recording (EOG
regression, artefact masks, bridged/bad-channel repair, average reference over
the 26 TDBRAIN positions), a 30 Hz zero-phase low-pass, and epochs from -0.2 to
0.8 s around
  * stimulus onsets (training pairs 10-21 and test pairs 200-229),
  * correct (94) and incorrect (104) feedback,
baseline -0.2-0 s, rejected if any channel exceeds 100 uV or 150 uV peak-to-peak.
Averages (all / odd / even trials) are stored at 100 Hz with trial counts and the
spatial noise covariance of single-trial baselines (per sample).

Output: outputs/erp/ds003474/erps.npz, qc.csv, and a data-level summary of the
feedback effect (reward positivity: correct - incorrect, FCz/Cz, 250-350 ms) by
BDI group (the effect Cavanagh et al. 2019 reported smaller with depression).
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mdd_tvb import new_datasets as ND  # noqa: E402
from mdd_tvb.preprocess_v2 import EEG_LABELS, preprocess_array  # noqa: E402

CONDITIONS = {"stim": list(range(10, 22)) + list(range(200, 230)), "fb_correct": [94], "fb_incorrect": [104]}
T_MIN, T_MAX, FS_OUT = -0.2, 0.8, 100.0
MAX_ABS_UV, MAX_PTP_UV = 100.0, 150.0
OUT = ROOT / "outputs/erp/ds003474"


def subjects():
    t = pd.read_csv(ND.DS003474_ROOT / "participants.tsv", sep="\t")
    t["subject_id"] = "ds003478_" + t.participant_id.astype(str)  # same people: use the rest-data IDs
    t["group"] = np.where(t.BDI >= 13, "HighBDI", np.where(t.BDI <= 6, "LowBDI", "MidBDI"))
    return t


def one(row):
    pid = row["participant_id"]
    out = {"subject_id": row["subject_id"], "participant_id": pid, "group": row["group"], "bdi": row["BDI"]}
    f = ND.DS003474_ROOT / pid / "eeg" / f"{pid}_task-ProbabilisticSelection_eeg.fdt"
    if not f.is_file():
        return {**out, "status": "missing"}
    try:
        x, eog, fs, labels, line_hz, valid, meta = ND.read_ds003474(pid)
        res = preprocess_array(x, eog, fs, labels, line_hz=line_hz, valid=valid)
    except Exception as error:  # noqa: BLE001
        return {**out, "status": "failed", "note": repr(error), "trace": traceback.format_exc()[-500:]}
    sos = butter(4, 30.0 / (0.5 * fs), btype="lowpass", output="sos")
    data = sosfiltfilt(sos, res.data_uv.astype(float), axis=1)
    present = [EEG_LABELS.index(c) for c in res.labels]
    step = int(round(fs / FS_OUT))
    i0, i1 = int(round(T_MIN * fs)), int(round(T_MAX * fs))
    base_n = -i0
    ev = meta["events"]
    out.update({"status": "ok" if res.qc_ok else "qc_flag", "note": res.qc_reason, "fs": fs,
                "repaired_channels": ";".join(res.repaired_channels), "emg_channels": ";".join(res.emg_channels),
                "missing_channels": ";".join(c for c in EEG_LABELS if c not in res.labels),
                "clean_fraction": res.stats["clean_fraction"]})
    erps, noise = {}, []
    for cond, codes in CONDITIONS.items():
        onsets = ev.loc[ev.code.isin(codes), "onset"].to_numpy(float)
        trials = []
        for t in onsets:
            c = int(round(t * fs))
            if c + i0 < 0 or c + i1 > data.shape[1]:
                continue
            seg = data[:, c + i0:c + i1]
            seg = seg - seg[:, :base_n].mean(1, keepdims=True)
            # standard ERP rejection after ocular regression (the continuous resting-state
            # artefact mask is too strict for task data with frequent blinks)
            if np.abs(seg).max() > MAX_ABS_UV or np.ptp(seg, axis=1).max() > MAX_PTP_UV:
                continue
            trials.append(seg)
            noise.append(seg[:, :base_n])
        out[f"n_{cond}"], out[f"n_{cond}_total"] = len(trials), len(onsets)
        if len(trials) < 4:
            continue
        tr = np.stack(trials)[:, :, ::step]
        full = np.zeros((3, 26, tr.shape[-1]))
        for j, sel in enumerate((slice(None), slice(0, None, 2), slice(1, None, 2))):
            full[j][present] = tr[sel].mean(0)
        erps[cond] = full
        out[f"n_{cond}_odd"], out[f"n_{cond}_even"] = len(tr[0::2]), len(tr[1::2])
    if noise:
        nb = np.concatenate(noise, axis=1)
        cov = np.zeros((26, 26))
        cov[np.ix_(present, present)] = np.cov(nb)
    else:
        cov = np.full((26, 26), np.nan)
    out["_erps"], out["_noise_cov"], out["_repair"] = erps, cov, res.repair_matrix
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()
    table = subjects()
    if args.max_subjects:
        table = table.head(args.max_subjects)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.n_jobs) as pool:
        futures = [pool.submit(one, r) for r in table.to_dict("records")]
        for k, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if k % 10 == 0 or k == len(futures):
                print(f"{k}/{len(futures)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: r["subject_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    qc.to_csv(OUT / "qc.csv", index=False)
    good = [r for r in rows if "_erps" in r and all(c in r["_erps"] for c in CONDITIONS)]
    n_t = next(iter(good[0]["_erps"].values())).shape[-1]
    times = T_MIN + np.arange(n_t) / FS_OUT
    np.savez_compressed(OUT / "erps.npz", subject_ids=np.asarray([r["subject_id"] for r in good]),
                        groups=np.asarray([r["group"] for r in good]), times=times,
                        channel_names=np.asarray(EEG_LABELS), conditions=np.asarray(list(CONDITIONS)),
                        erp=np.stack([np.stack([r["_erps"][c] for c in CONDITIONS]) for r in good]),  # (N, cond, all/odd/even, 26, T)
                        counts=np.asarray([[[r[f"n_{c}"], r[f"n_{c}_odd"], r[f"n_{c}_even"]] for c in CONDITIONS] for r in good]),
                        noise_cov=np.stack([r["_noise_cov"] for r in good]), repair=np.stack([r["_repair"] for r in good]))
    print(f"{len(good)}/{len(rows)} subjects with all conditions; status {qc.status.value_counts().to_dict()}")
    # data-level check: reward positivity by BDI group
    import scipy.stats as st
    ch = [EEG_LABELS.index(c) for c in ("FCz", "Cz")]
    win = (times >= 0.25) & (times <= 0.35)
    rewp = np.array([(r["_erps"]["fb_correct"][0] - r["_erps"]["fb_incorrect"][0])[ch][:, win].mean() for r in good])
    grp = np.array([r["group"] for r in good])
    bdi = np.array([r["bdi"] for r in good], float)
    hi, lo = rewp[grp == "HighBDI"], rewp[grp == "LowBDI"]
    summary = {"n": len(good), "rewp_uV_high_bdi": float(hi.mean()), "rewp_uV_low_bdi": float(lo.mean()),
               "t_high_vs_low": float(st.ttest_ind(hi, lo).statistic), "p_high_vs_low": float(st.ttest_ind(hi, lo).pvalue),
               "cohen_d": float((hi.mean() - lo.mean()) / np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2)),
               "spearman_rewp_bdi": float(st.spearmanr(rewp, bdi).correlation),
               "rewp_mean_uV": float(rewp.mean()), "rewp_t_vs_zero": float(st.ttest_1samp(rewp, 0).statistic),
               "median_trials": {c: float(np.median([r[f"n_{c}"] for r in good])) for c in CONDITIONS}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
