"""Evaluation plans shared by the linear-regime fitting scripts.

A plan yields ``(fold, transformer, null_csds, test_ids)``:

* nested (default): the M5.2 outer folds on the 262 development subjects; each
  fold's transformer and pooled null come from that fold's training subjects;
* external (``external_transformer`` given): one plan with fold id ``-1``.  The
  transformer was learned on all development subjects, the pooled null is the
  mean over the development fit halves, and every subject of the (external)
  empirical collection is a test subject.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mdd_tvb.linear_fit import load_transformer_arrays
from mdd_tvb.spectral_features import load_cross_spectral_collection

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_FOLD = -1


def add_plan_arguments(ap) -> None:
    ap.add_argument("--external-transformer", default=None,
                    help="transformer learned on all development subjects; switches to external mode")
    ap.add_argument("--null-emp-dir", default="outputs/m5_spectral_m51_production/empirical",
                    help="external mode: collection holding the development fit halves for the pooled null")


def development_ids() -> list[str]:
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    return sorted(splits.subject_id.astype(str).unique())


def plans(args, fit, folds=None):
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    if args.external_transformer:
        T = load_transformer_arrays(ROOT / args.external_transformer)
        dev = load_cross_spectral_collection(ROOT / args.null_emp_dir / "cross_spectra_fit.npz")
        dev_index = {s: i for i, s in enumerate(dev.subject_ids.astype(str))}
        null = [dev.csd[dev_index[s]] for s in development_ids()]
        test = list(fit.subject_ids.astype(str))
        yield EXTERNAL_FOLD, T, null, test
        return
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    for fold in sorted(splits.outer_fold.unique()):
        if folds is not None and fold not in folds:
            continue
        fr = splits[splits.outer_fold == fold]
        train = [index[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str)]
        test = list(fr.loc[fr.outer_role == "validation", "subject_id"].astype(str))
        T = load_transformer_arrays(ROOT / f"outputs/m52_nested_baseline/outer_{fold}/spectral_transformer.npz")
        yield int(fold), T, [fit.csd[i] for i in train], test
