"""M5.1 bank posterior on an external cohort (comparator for the frozen M5.3 model).

Training subjects = the 262 M5.2 development subjects (transformer, temperature,
pooled null); test subjects = every subject of the external collection.  The
consumed 65-subject M5.1 holdout is not read.  Group-effect plots need two
groups among the test subjects and are skipped for a single external cohort;
every per-subject output is written before that point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mdd_tvb.spectral_bank import load_spectral_simulation_bank
from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_features import (
    CrossSpectralCollection,
    load_cross_spectral_collection,
    subset_cross_spectral_collection,
)
from mdd_tvb.spectral_fit import fit_spectral_subjects

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("cross_spectra_fit.npz", "cross_spectra_validation.npz",
         "cross_spectra_reliability_a.npz", "cross_spectra_reliability_b.npz")


def concat(a: CrossSpectralCollection, b: CrossSpectralCollection) -> CrossSpectralCollection:
    if not (np.array_equal(a.frequency_hz, b.frequency_hz) and np.array_equal(a.channel_names, b.channel_names)):
        raise ValueError("collections differ in frequency grid or channel order")
    return CrossSpectralCollection(
        subject_ids=np.concatenate([a.subject_ids, b.subject_ids]).astype("U32"),
        groups=np.concatenate([a.groups, b.groups]).astype("U32"),
        source_files=np.concatenate([a.source_files, b.source_files]).astype("U512"),
        durations_s=np.concatenate([a.durations_s, b.durations_s]),
        epoch_counts=np.concatenate([a.epoch_counts, b.epoch_counts]),
        frequency_hz=a.frequency_hz, channel_names=a.channel_names,
        csd=np.concatenate([a.csd, b.csd]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-emp", default="outputs/external/rtms_restEC/empirical")
    ap.add_argument("--dev-emp", default="outputs/m5_spectral_m51_production/empirical")
    ap.add_argument("--out", default="outputs/external/rtms_restEC/m51")
    args = ap.parse_args()
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    dev_ids = set(splits.subject_id.astype(str))
    dev = [load_cross_spectral_collection(ROOT / args.dev_emp / n) for n in NAMES]
    keep = np.asarray([i for i, s in enumerate(dev[0].subject_ids.astype(str)) if s in dev_ids])
    assert len(keep) == 262, len(keep)
    dev = [subset_cross_spectral_collection(c, keep) for c in dev]
    ext = [load_cross_spectral_collection(ROOT / args.external_emp / n) for n in NAMES]
    assert not set(ext[0].subject_ids.astype(str)) & dev_ids
    joined = [concat(d, e) for d, e in zip(dev, ext)]
    train = np.arange(len(keep))
    test = np.arange(len(keep), len(keep) + len(ext[0].subject_ids))
    bank = load_spectral_simulation_bank(ROOT / cfg.paths.output_dir / "bank" / "spectral_simulation_bank.npz")
    out = ROOT / args.out
    try:
        fit_spectral_subjects(cfg, joined[0], joined[1], bank, joined[2], joined[3],
                              split_indices=(train, test), fit_directory=out)
    except Exception as error:  # group-effect plots need two held-out groups
        if not (out / "subject_posteriors.csv").is_file():
            raise
        print("post-fit reporting skipped:", repr(error))
    table = pd.read_csv(out / "subject_posteriors.csv")
    held = table[table.subject_split == "holdout"]
    summary = {
        "subjects": int(len(held)),
        "median_unseen_total_ratio": float(held.validation_cost_ratio_to_pooled_null.median()),
        "fraction_beating_null": float((held.validation_cost_ratio_to_pooled_null < 1).mean()),
        "median_unseen_auto_ratio": float(held.validation_auto_spectrum_cost_ratio_to_pooled_null.median()),
        "median_unseen_cross_ratio": float(held.validation_complex_coherency_cost_ratio_to_pooled_null.median()),
        "median_unseen_topo_ratio": float(held.validation_alpha_topography_cost_ratio_to_pooled_null.median()),
        "median_observation_noise_fraction": float(held.observation_noise_fraction_posterior_mean.median()),
    }
    (out / "external_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
