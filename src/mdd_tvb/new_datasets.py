"""Readers for batch-free depression EEG datasets, mapped onto the 26 TDBRAIN channels.

Every reader returns ``(x_uv, eog_uv, fs, labels, line_hz, valid, meta)`` for
``preprocess_v2.preprocess_array``: EEG rows in canonical TDBRAIN order (only the
channels the dataset has), an optional (2, n) VEOG/HEOG array, the sampling rate,
the mains frequency, an optional usable-time mask, and subject metadata.

* ds003478 (Cavanagh, OpenNeuro): 64-channel Neuroscan; all 26 labels exist;
  alternating 1-min eyes-open / eyes-closed blocks, eyes-closed time from the
  event markers; HEOG/VEOG channels; 60 Hz mains.
* Mumtaz 2016 (figshare 4244171): 19-channel 10-20 EEG against linked ears
  (T3/T4/T5/T6 = T7/T8/P7/P8); separate eyes-closed file; no EOG; 50 Hz mains.
* MODMA (Lanzhou): 128-channel HydroCel net referenced to Cz (not stored). The Cz
  reference is restored as a zero channel and the data are average-referenced over
  all 129 electrodes before the 26 TDBRAIN positions are picked: 20 by EGI's
  published 10-20 equivalents, FC3/FCz/FC4/CP3/CPz/CP4 by nearest electrode after a
  similarity alignment on those anchors (E29, E6, E111, E42, E55, E93); EOG derived
  from the periocular electrodes; 50 Hz mains.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess_v2 import EEG_LABELS

DS003478_ROOT = Path("D:/university/projects/mdd-new-datasets/ds003478")
MUMTAZ_ROOT = Path("D:/university/projects/mdd-new-datasets/mumtaz2016")
MODMA_ROOT = Path("D:/university/projects/graph-opt/mdd-dataset-2/MODMA_EEG_BIDS_format/MODMA_EEG_BIDS_format/"
                  "EEG_LZU_2015_2_resting state")

MODMA_MAP = {"Fp1": "E22", "Fp2": "E9", "F7": "E33", "F3": "E24", "Fz": "E11", "F4": "E124", "F8": "E122",
             "FC3": "E29", "FCz": "E6", "FC4": "E111", "T7": "E45", "C3": "E36", "Cz": "Cz", "C4": "E104",
             "T8": "E108", "CP3": "E42", "CPz": "E55", "CP4": "E93", "P7": "E58", "P3": "E52", "Pz": "E62",
             "P4": "E92", "P8": "E96", "O1": "E70", "Oz": "E75", "O2": "E83"}
MUMTAZ_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


# ---------------------------------------------------------------------------
# subject lists
# ---------------------------------------------------------------------------
def ds003478_subjects() -> pd.DataFrame:
    t = pd.read_csv(DS003478_ROOT / "participants.tsv", sep="\t")
    t["subject_id"] = "ds003478_" + t.participant_id.astype(str)
    t["group"] = np.where(t.BDI >= 13, "HighBDI", np.where(t.BDI <= 6, "LowBDI", "MidBDI"))
    t["mdd_current"] = t.SCID.astype(str).str.contains("Current MDD")
    return t.rename(columns={"BDI": "bdi"})[["subject_id", "participant_id", "group", "age", "sex", "bdi",
                                             "mdd_current", "SCID"]]


def mumtaz_subjects() -> pd.DataFrame:
    rows = []
    for f in sorted(MUMTAZ_ROOT.glob("*EC.edf")):
        parts = f.stem.split()
        group = "MDD" if parts[0] == "MDD" else "HC"
        rows.append({"subject_id": f"mumtaz_{group}_{parts[1]}", "file": f.name, "group": group})
    return pd.DataFrame(rows)


def modma_subjects() -> pd.DataFrame:
    # the file mixes single and double tabs: split on any whitespace
    t = pd.read_csv(MODMA_ROOT / "participants.tsv", sep=r"\s+", engine="python")
    t = t[t.participant_id.astype(str).str.startswith("sub-")]
    out = pd.DataFrame({"subject_id": "modma_" + t.participant_id.astype(str).str.strip(),
                        "participant_id": t.participant_id.astype(str).str.strip(),
                        "group": t.group.astype(str).str.strip(),
                        "age": pd.to_numeric(t.age, errors="coerce"),
                        "sex": t.gender.astype(str).str.strip().map({"M": 1, "F": 0}),
                        "phq9": pd.to_numeric(t["PHQ-9"], errors="coerce")})
    return out


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def read_ds003478(participant_id: str, run: int = 1):
    import mne

    base = DS003478_ROOT / participant_id / "eeg" / f"{participant_id}_task-Rest_run-0{run}_eeg"
    raw = mne.io.read_raw_eeglab(str(base) + ".set", preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    upper = {c.upper(): c for c in raw.ch_names}
    labels = [c for c in EEG_LABELS if c.upper() in upper]
    x = raw.get_data(picks=[upper[c.upper()] for c in labels]) * 1e6
    eog = raw.get_data(picks=[upper["VEOG"], upper["HEOG"]]) * 1e6 if {"VEOG", "HEOG"} <= set(upper) else None
    ev = pd.read_csv(str(base) + "_events.tsv", sep="\t")
    closed = ev[ev.trial_type.astype(str).str.contains("Eyes Closed", case=False)].onset.to_numpy(float)
    valid = np.zeros(raw.n_times, bool)
    if len(closed):
        blocks = np.split(closed, np.flatnonzero(np.diff(closed) > 5.0) + 1)
        for b in blocks:  # markers run every 0.5-2 s through each eyes-closed minute
            s0, s1 = int((b[0] + 1.0) * fs), int((b[-1] + 1.0) * fs)
            valid[max(s0, 0):min(s1, raw.n_times)] = True
    return x, eog, fs, labels, 60.0, valid, {}


def read_mumtaz(filename: str):
    import mne

    raw = mne.io.read_raw_edf(str(MUMTAZ_ROOT / filename), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    found = {}
    for ch in raw.ch_names:
        name = ch.replace("EEG", "").replace("-LE", "").strip()
        name = MUMTAZ_NAMES.get(name, name)
        if name in EEG_LABELS:
            found[name] = ch
    labels = [c for c in EEG_LABELS if c in found]
    x = raw.get_data(picks=[found[c] for c in labels]) * 1e6
    return x, None, fs, labels, 50.0, None, {}


def read_modma(participant_id: str):
    import mne

    f = MODMA_ROOT / participant_id / "eeg" / f"{participant_id}_task-Resting-state_eeg.EDF"
    raw = mne.io.read_raw_edf(str(f), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    data = raw.get_data() * 1e6                      # 128 channels against the Cz reference
    names = [c.strip() for c in raw.ch_names]
    data = np.vstack([data, np.zeros((1, data.shape[1]))])  # restore Cz (the reference) as zeros
    names = names + ["Cz"]
    data = data - data.mean(axis=0, keepdims=True)   # average reference over all 129 electrodes
    idx = {n: i for i, n in enumerate(names)}
    labels = list(EEG_LABELS)
    x = data[[idx[MODMA_MAP[c]] for c in labels]]
    veog = data[[idx["E25"], idx["E8"]]].mean(0) - data[[idx["E127"], idx["E126"]]].mean(0)
    heog = data[idx["E128"]] - data[idx["E125"]]
    return x, np.stack([veog, heog]), fs, labels, 50.0, None, {}
