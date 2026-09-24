"""Nested posterior averaging over each subject's best locally-optimised states.

For outer fold k the kernel temperature is chosen using only subjects of the
other folds (all of which are training subjects of fold k), by minimising the
median unseen total-cost ratio; it is then applied to fold k.  Predictions are
CSD-space posterior averages over the 12 best (state, spatial, nuisance) fits.
"""
from __future__ import annotations

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp  # noqa: E402
from mdd_tvb.spectral_features import load_cross_spectral_collection  # noqa: E402
from mdd_tvb.linear_fit import load_transformer_arrays, feature_blocks, weighted_features  # noqa: E402

N_CH = 26
IU = np.triu_indices(N_CH)


def full(upper: np.ndarray) -> np.ndarray:
    out = np.zeros(upper.shape[:-1] + (N_CH, N_CH), dtype=np.complex128)
    out[..., IU[0], IU[1]] = upper
    diag = np.real(np.diagonal(out, axis1=-2, axis2=-1)).copy()
    out = out + np.conjugate(np.swapaxes(out, -1, -2))
    idx = np.arange(N_CH)
    out[..., idx, idx] = diag
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/linear_regime/nested_tvb_common")
    ap.add_argument("--multipliers", type=float, nargs="*",
                    default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    args = ap.parse_args()
    run = ROOT / args.run
    top = np.load(run / "top_states.npz")
    fits = pd.read_csv(run / "subject_fits.csv").set_index("subject_id")
    ids = top["subject_ids"].astype(str)
    loss = top["loss"]
    csds = top["csd_upper"]
    splits = pd.read_csv(ROOT / "configs/m52_nested_splits.csv")
    emp = ROOT / "outputs/m5_spectral_m51_production/empirical"
    fit = load_cross_spectral_collection(emp / "cross_spectra_fit.npz")
    val = load_cross_spectral_collection(emp / "cross_spectra_validation.npz")
    index = {s: i for i, s in enumerate(fit.subject_ids.astype(str))}
    fold_of = {s: int(fits.loc[s, "outer_fold"]) for s in ids}

    folds = sorted(splits.outer_fold.unique())
    transformers, nulls = {}, {}
    for k in folds:
        fr = splits[splits.outer_fold == k]
        T = load_transformer_arrays(ROOT / f"outputs/m52_nested_baseline/outer_{k}/spectral_transformer.npz")
        train = [index[s] for s in fr.loc[fr.outer_role == "training", "subject_id"].astype(str)]
        blocks = [feature_blocks(jnp.asarray(fit.csd[i]), T) for i in train]
        nulls[k] = ([np.mean([np.asarray(b[j]) for b in blocks], 0) for j in range(3)],
                    np.mean([np.asarray(weighted_features(jnp.asarray(fit.csd[i]), T)) for i in train], 0))
        transformers[k] = T
    val_feats = {s: np.asarray(weighted_features(jnp.asarray(val.csd[index[s]]), transformers[fold_of[s]])) for s in ids}
    val_blocks = {s: [np.asarray(b) for b in feature_blocks(jnp.asarray(val.csd[index[s]]), transformers[fold_of[s]])]
                  for s in ids}
    # Scale from first-half information only (median best penalised fit loss);
    # the multiplier itself is chosen per fold on other folds' subjects.
    base_T = float(np.median(loss[:, 0]))

    def prediction(n: int, multiplier: float) -> np.ndarray:
        if multiplier == 0.0:
            w = np.zeros(loss.shape[1]); w[0] = 1.0
        else:
            lw = -(loss[n] - loss[n].min()) / (2.0 * multiplier * base_T)
            w = np.exp(lw - lw.max()); w /= w.sum()
        return np.einsum("m,mfcd->fcd", w, full(csds[n]))

    def total_ratio(n: int, multiplier: float) -> float:
        s = ids[n]
        k = fold_of[s]
        f = np.asarray(weighted_features(jnp.asarray(prediction(n, multiplier)), transformers[k]))
        return float(np.sum((f - val_feats[s]) ** 2) / np.sum((nulls[k][1] - val_feats[s]) ** 2))

    ratios = {m: np.asarray([total_ratio(n, m) for n in range(len(ids))]) for m in args.multipliers}
    chosen = {}
    for k in folds:
        other = np.asarray([fold_of[s] != k for s in ids])
        med = {m: float(np.median(r[other])) for m, r in ratios.items()}
        chosen[k] = min(med, key=med.get)
    rows, preds = [], []
    for n, s in enumerate(ids):
        k = fold_of[s]
        csd = prediction(n, chosen[k])
        preds.append(csd)
        pb = [np.asarray(b) for b in feature_blocks(jnp.asarray(csd), transformers[k])]
        f = np.asarray(weighted_features(jnp.asarray(csd), transformers[k]))
        row = {"subject_id": s, "outer_fold": k, "temperature_multiplier": chosen[k],
               "validation_ratio": float(np.sum((f - val_feats[s]) ** 2) / np.sum((nulls[k][1] - val_feats[s]) ** 2))}
        for name, j in (("auto", 0), ("cross", 1), ("topo", 2)):
            row[f"validation_{name}_ratio"] = float(np.mean((pb[j] - val_blocks[s][j]) ** 2) /
                                                    np.mean((nulls[k][0][j] - val_blocks[s][j]) ** 2))
        rows.append(row)
    out = run / "posterior"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_fits.csv", index=False)
    np.savez_compressed(out / "predictions.npz", subject_ids=ids, csd=np.stack(preds))
    summary = {"chosen_multiplier_per_fold": {int(k): v for k, v in chosen.items()},
               "multiplier_scan_median_total_all_subjects": {str(m): float(np.median(r)) for m, r in ratios.items()},
               "median_unseen_total": float(table.validation_ratio.median()),
               "median_unseen_auto": float(table.validation_auto_ratio.median()),
               "median_unseen_cross": float(table.validation_cross_ratio.median()),
               "median_unseen_topo": float(table.validation_topo_ratio.median())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
