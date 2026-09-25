"""Run the artefact-aware v2 preprocessing and extract M5-format cross-spectra.

Cohorts: ``dev`` = the 327 Healthy/MDD recordings of the original M5 set
(including the consumed M5.1 holdout, which downstream code excludes through
the split file), ``rtms`` = the 164 session-1 MDD-rTMS recordings.  Tasks:
restEC and restEO.  Output per cohort/task: ``<out>/<cohort>_<task>/empirical``
with fit / validation / reliability collections (same format as M5), a QC
table and per-subject artefact summaries.  Spectra use only 4-s epochs that lie
entirely inside artefact-free time; halves share no samples.
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mdd_tvb.preprocess_v2 import EEG_LABELS, clean_epoch_starts, preprocess_bdf, split_halves  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import (CrossSpectralCollection, estimate_cross_spectrum,  # noqa: E402
                                       save_cross_spectral_collection)

TDBRAIN = Path("D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1")
MIN_EPOCHS_PER_HALF = 6


def cohort_subjects(cohort: str) -> list[tuple[str, str]]:
    if cohort == "dev":
        s = pd.read_csv(TDBRAIN / "output/preprocessed_tdbrain_restEC_set2/processing_summary.csv")
        s = s[s.status == "saved"]
        return list(zip(s.subject_id.astype(str), s.group.astype(str)))
    T = pd.read_excel(TDBRAIN / "TDBRAIN_participants_V3.xlsx")
    r = T[(T["Dataset"] == "MDD-rTMS") & (T["sessID"] == 1)]
    return [(sid, "rTMS") for sid in sorted(r["TDBRAIN_ID"].astype(str).unique())]


def one(sid: str, group: str, task: str, settings) -> dict:
    path = TDBRAIN / sid / "ses-1" / "eeg" / f"{sid}_ses-1_task-{task}_eeg.bdf"
    row = {"subject_id": sid, "group": group, "task": task, "source_file": str(path)}
    if not path.is_file():
        return {**row, "status": "missing_bdf"}
    try:
        res = preprocess_bdf(path)
    except Exception as error:  # noqa: BLE001
        return {**row, "status": "failed", "note": repr(error), "trace": traceback.format_exc()[-800:]}
    fs = res.fs
    nperseg = int(round(settings.epoch_seconds * fs))
    step = nperseg // 2
    starts = clean_epoch_starts(res.clean, nperseg, step)
    first, second = split_halves(starts, nperseg)
    rel_a, rel_b = split_halves(first, nperseg)
    row.update({"status": "ok" if res.qc_ok else "qc_flag", "note": res.qc_reason,
                "repaired_channels": ";".join(res.repaired_channels),
                "emg_channels": ";".join(res.emg_channels),
                "bridged_pairs": ";".join(f"{a}-{b}" for a, b in res.bridged_pairs),
                "n_repaired": len(res.repaired_channels), "clean_epochs": len(starts),
                "fit_epochs": len(first), "validation_epochs": len(second),
                "reliability_a_epochs": len(rel_a), "reliability_b_epochs": len(rel_b),
                **{k: v for k, v in res.stats.items() if not isinstance(v, (dict, list))},
                "slope_20_40": json.dumps(res.stats["slope_20_40"]),
                "hf_robust_z": json.dumps(res.stats["hf_robust_z"]),
                "eog_coef_fp1_fp2": json.dumps(np.round(res.eog_coefficients[:2], 4).tolist())})
    if min(len(rel_a), len(rel_b), len(second)) < MIN_EPOCHS_PER_HALF // 2 or min(len(first), len(second)) < MIN_EPOCHS_PER_HALF:
        row["status"] = "too_little_clean_data"
        return row
    eeg = res.data_uv.T.astype(float) * 1e-6  # volts, (samples, channels), as the M5 loader returns
    out = {}
    for name, s in (("fit", first), ("validation", second), ("reliability_a", rel_a), ("reliability_b", rel_b)):
        freq, csd, n = estimate_cross_spectrum(eeg, fs, settings, starts=s)
        out[name] = csd
    row["_csd"] = out
    row["_freq"] = freq
    row["_repair"] = res.repair_matrix
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["dev", "rtms"])
    ap.add_argument("--tasks", nargs="+", default=["restEC", "restEO"])
    ap.add_argument("--out", default="outputs/preproc_v2")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()
    settings = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml").spectral
    for cohort in args.cohorts:
        subjects = cohort_subjects(cohort)
        if args.max_subjects:
            subjects = subjects[: args.max_subjects]
        for task in args.tasks:
            t0 = time.time()
            rows = []
            with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
                futures = [pool.submit(one, sid, group, task, settings) for sid, group in subjects]
                for k, fut in enumerate(as_completed(futures), start=1):
                    rows.append(fut.result())
                    if k % 25 == 0 or k == len(futures):
                        print(f"{cohort} {task}: {k}/{len(futures)} {time.time()-t0:.0f}s", flush=True)
            rows.sort(key=lambda r: (r["group"], r["subject_id"]))
            out = ROOT / args.out / f"{cohort}_{task}" / "empirical"
            out.mkdir(parents=True, exist_ok=True)
            good = [r for r in rows if "_csd" in r]
            qc = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
            qc.to_csv(out / "qc.csv", index=False)
            if not good:
                continue
            ids = np.asarray([r["subject_id"] for r in good], dtype="U32")
            groups = np.asarray([r["group"] for r in good], dtype="U32")
            files = np.asarray([r["source_file"] for r in good], dtype="U512")
            for name in ("fit", "validation", "reliability_a", "reliability_b"):
                n_ep = np.asarray([r[f"{name}_epochs"] for r in good], dtype=int)
                save_cross_spectral_collection(out / f"cross_spectra_{name}.npz", CrossSpectralCollection(
                    subject_ids=ids, groups=groups, source_files=files,
                    durations_s=(n_ep + 1) * settings.epoch_seconds / 2.0, epoch_counts=n_ep,
                    frequency_hz=good[0]["_freq"], channel_names=np.asarray(EEG_LABELS, dtype="U16"),
                    csd=np.stack([r["_csd"][name] for r in good])))
            np.savez_compressed(out / "repair_matrices.npz", subject_ids=ids,
                                repair=np.stack([r["_repair"] for r in good]).astype(np.float32))
            print(f"{cohort} {task}: {len(good)}/{len(rows)} usable; status "
                  f"{qc.status.value_counts().to_dict()} -> {out}", flush=True)


if __name__ == "__main__":
    main()
