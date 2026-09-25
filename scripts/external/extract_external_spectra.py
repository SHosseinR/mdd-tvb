"""Extract complex cross-spectra of an external cohort with the unchanged M5 pipeline.

The recordings must have been preprocessed exactly like the development data
(``preprocess_tdbrain_rtms_restEC.m`` is the original EEGLAB script restricted to
the rTMS cohort).  Halves, epochs, frequency bins and channel order are those
of ``configs/m5_spectral.toml``.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from mdd_tvb.spectral_config import load_spectral_m5_config
from mdd_tvb.spectral_empirical import extract_cross_spectral_collection

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="D:/university/projects/graph-opt/tbdbrain/TDBRAIN_Dataset_V3_1/output/"
                                              "preprocessed_tdbrain_restEC_rtms")
    ap.add_argument("--group", default="rTMS")
    ap.add_argument("--out", default="outputs/external/rtms_restEC")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_spectral_m5_config(ROOT / "configs/m5_spectral.toml")
    cfg = dataclasses.replace(
        cfg,
        paths=dataclasses.replace(cfg.paths, dataset_root=Path(args.dataset_root), output_dir=ROOT / args.out),
        empirical=dataclasses.replace(cfg.empirical, groups=(args.group,), n_jobs=args.n_jobs),
    )
    fitting, validation = extract_cross_spectral_collection(cfg)
    print(f"extracted {len(fitting.subject_ids)} subjects -> {cfg.paths.output_dir / 'empirical'}")


if __name__ == "__main__":
    main()
