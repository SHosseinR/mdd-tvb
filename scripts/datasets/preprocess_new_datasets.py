"""Artefact-aware (v2) cleaning and M5-format cross-spectra for the batch-free datasets.

Output per dataset: ``outputs/preproc_v2/<dataset>_restEC/empirical`` with
fit / validation / reliability collections (26 x 26; channels the dataset lacks
are zero rows and listed in qc.csv ``missing_channels``), a QC table with the
subject metadata, repair matrices (26 x 26), and ``configs/m54_lists/<dataset>_restEC_v2.txt``
plus a 5-fold stratified split file ``configs/splits_<dataset>.csv``.
"""
from __future__ import annotations

import argparse, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mdd_tvb import new_datasets as ND  # noqa: E402
from mdd_tvb.preprocess_v2 import EEG_LABELS, clean_epoch_starts, preprocess_array, split_halves  # noqa: E402
from mdd_tvb.spectral_config import load_spectral_m5_config  # noqa: E402
from mdd_tvb.spectral_features import (CrossSpectralCollection, estimate_cross_spectrum,  # noqa: E402
                                       save_cross_spectral_collection)

MIN_EPOCHS_PER_HALF = 6


def subjects(dataset):
    if dataset == "ds003478":
        return ND.ds003478_subjects()
    if dataset == "mumtaz":
        return ND.mumtaz_subjects()
    return ND.modma_subjects()


def read(dataset, row):
    if dataset == "ds003478":
        return ND.read_ds003478(row["participant_id"])
    if dataset == "mumtaz":
        return ND.read_mumtaz(row["file"])
    return ND.read_modma(row["participant_id"])


def one(dataset, row, settings) -> dict:
    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    try:
        x, eog, fs, labels, line_hz, valid, _ = read(dataset, row)
        res = preprocess_array(x, eog, fs, labels, line_hz=line_hz, valid=valid)
    except Exception as error:  # noqa: BLE001
        return {**out, "status": "failed", "note": repr(error), "trace": traceback.format_exc()[-600:]}
    nperseg = int(round(settings.epoch_seconds * fs))
    starts = clean_epoch_starts(res.clean, nperseg, nperseg // 2)
    first, second = split_halves(starts, nperseg)
    rel_a, rel_b = split_halves(first, nperseg)
    present = [EEG_LABELS.index(c) for c in res.labels]
    out.update({"status": "ok" if res.qc_ok else "qc_flag", "note": res.qc_reason, "fs": fs,
                "missing_channels": ";".join(c for c in EEG_LABELS if c not in res.labels),
                "repaired_channels": ";".join(res.repaired_channels), "emg_channels": ";".join(res.emg_channels),
                "bridged_pairs": ";".join(f"{a}-{b}" for a, b in res.bridged_pairs),
                "n_repaired": len(res.repaired_channels), "clean_epochs": len(starts),
                "fit_epochs": len(first), "validation_epochs": len(second),
                "reliability_a_epochs": len(rel_a), "reliability_b_epochs": len(rel_b),
                **{k: v for k, v in res.stats.items() if not isinstance(v, (dict, list))},
                "slope_20_40": json.dumps(res.stats["slope_20_40"])})
    if min(len(rel_a), len(rel_b), len(second)) < MIN_EPOCHS_PER_HALF // 2 or min(len(first), len(second)) < MIN_EPOCHS_PER_HALF:
        out["status"] = "too_little_clean_data"
        return out
    eeg = res.data_uv.T.astype(float) * 1e-6
    csd = {}
    for name, s in (("fit", first), ("validation", second), ("reliability_a", rel_a), ("reliability_b", rel_b)):
        freq, c, _ = estimate_cross_spectrum(eeg, fs, settings, starts=s)
        full = np.zeros((len(freq), 26, 26), complex)
        full[np.ix_(range(len(freq)), present, present)] = c
        csd[name] = full
    P = np.eye(26)
    P[np.ix_(present, present)] = res.repair_matrix
    out["_csd"], out["_freq"], out["_repair"] = csd, freq, P
    return out


def write(dataset, rows, settings):
    rows.sort(key=lambda r: (str(r["group"]), str(r["subject_id"])))
    out = ROOT / "outputs/preproc_v2" / f"{dataset}_restEC" / "empirical"
    out.mkdir(parents=True, exist_ok=True)
    qc = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    qc.to_csv(out / "qc.csv", index=False)
    good = [r for r in rows if "_csd" in r]
    ids = np.asarray([r["subject_id"] for r in good], dtype="U32")
    groups = np.asarray([r["group"] for r in good], dtype="U32")
    files = np.asarray([str(r.get("participant_id", r.get("file", ""))) for r in good], dtype="U512")
    for name in ("fit", "validation", "reliability_a", "reliability_b"):
        n_ep = np.asarray([r[f"{name}_epochs"] for r in good], dtype=int)
        save_cross_spectral_collection(out / f"cross_spectra_{name}.npz", CrossSpectralCollection(
            subject_ids=ids, groups=groups, source_files=files,
            durations_s=(n_ep + 1) * settings.epoch_seconds / 2.0, epoch_counts=n_ep,
            frequency_hz=good[0]["_freq"], channel_names=np.asarray(EEG_LABELS, dtype="U16"),
            csd=np.stack([r["_csd"][name] for r in good])))
    np.savez_compressed(out / "repair_matrices.npz", subject_ids=ids,
                        repair=np.stack([r["_repair"] for r in good]).astype(np.float32))
    usable = qc[qc.status.isin(["ok", "qc_flag"]) & (qc.note.fillna("") != "unrepairable channel neighbourhood")]
    usable = usable[usable.subject_id.isin(ids)]
    (ROOT / "configs/m54_lists" / f"{dataset}_restEC_v2.txt").write_text("\n".join(sorted(usable.subject_id)) + "\n")
    # 5-fold stratified split (same layout as configs/m52_nested_splits.csv)
    rng = np.random.default_rng(20260927)
    fold_of = {}
    for g, sub in usable.groupby("group"):
        s = rng.permutation(sub.subject_id.to_numpy())
        for k, sid in enumerate(s):
            fold_of[sid] = k % 5
    split_rows = [{"subject_id": sid, "group": usable.set_index("subject_id").loc[sid, "group"], "outer_fold": f,
                   "outer_role": "validation" if fold_of[sid] == f else "training", "inner_fold": -1}
                  for f in range(5) for sid in sorted(fold_of)]
    pd.DataFrame(split_rows).to_csv(ROOT / f"configs/splits_{dataset}.csv", index=False)
    pd.DataFrame(split_rows).to_csv(ROOT / f"configs/m54_lists/splits_{dataset}.csv", index=False)
    print(f"{dataset}: {len(good)}/{len(rows)} with spectra, {len(usable)} usable; "
          f"status {qc.status.value_counts().to_dict()}; groups {usable.group.value_counts().to_dict()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["modma", "mumtaz", "ds003478"])
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--max-subjects", type=int, default=0)
    args = ap.parse_args()
    settings = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml").spectral
    for dataset in args.datasets:
        table = subjects(dataset)
        if args.max_subjects:
            table = table.head(args.max_subjects)
        t0, rows = time.time(), []
        with ProcessPoolExecutor(args.n_jobs) as pool:
            futures = [pool.submit(one, dataset, row._asdict() if hasattr(row, "_asdict") else dict(row), settings)
                       for _, row in table.iterrows()]
            for k, fut in enumerate(as_completed(futures), start=1):
                rows.append(fut.result())
                if k % 20 == 0 or k == len(futures):
                    print(f"{dataset}: {k}/{len(futures)} {time.time() - t0:.0f}s", flush=True)
        write(dataset, rows, settings)


if __name__ == "__main__":
    main()
