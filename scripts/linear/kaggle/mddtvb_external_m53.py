"""Kaggle GPU job: frozen M5.3 model on the external rTMS cohort (docs/M53_FROZEN_MODEL.md).

Inputs (private datasets): the linear bundle (code, development spectra, transformers),
the code overlay, and ``mdd-tvb-external`` (rTMS spectra + the all-development
transformer written by scripts/external/run_m51_external.py).
Banks are regenerated from the frozen population fits and stage-1 fits, then a
reproduction check refits the development subjects (TVB, bank MAP) before the
external subjects are fitted and refined for both lead fields.
"""
import glob, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path

T0 = time.time()
work = Path("/tmp/mdd-tvb")
results = Path("/kaggle/working/results")
results.mkdir(parents=True, exist_ok=True)


def extract(pattern, fallback_marker=None):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if hits:
        with tarfile.open(hits[0]) as tar:
            tar.extractall(work)
        return True
    return False


if not extract("mddtvb_bundle.tar.gz"):
    configs = glob.glob("/kaggle/input/**/configs/baseline.toml", recursive=True)
    shutil.copytree(Path(configs[0]).parents[1], work, dirs_exist_ok=True)
if not extract("mddtvb_code.tar.gz"):
    for sub in ("src/mdd_tvb", "scripts/linear", "configs/m53_frozen"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{sub}", recursive=True) if "linear-code" in h]
        if hits:
            shutil.copytree(hits[0], work / sub, dirs_exist_ok=True)
if not extract("mddtvb_external.tar.gz"):
    hits = glob.glob("/kaggle/input/**/outputs/external", recursive=True)
    shutil.copytree(hits[0], work / "outputs/external", dirs_exist_ok=True)
assert (work / "scripts/linear/plans.py").is_file(), "stale code dataset"
assert (work / "configs/m53_frozen/population_fit_tvb.json").is_file(), "frozen configs missing"
assert (work / "outputs/external/rtms_restEC/empirical/cross_spectra_fit.npz").is_file(), "external data missing"
placeholder = work / "raw_eeg_placeholder"
placeholder.mkdir(exist_ok=True)
for cfg_path in (work / "configs").glob("*.toml"):
    cfg_path.write_text(re.sub(r'dataset_root = ".*"', f'dataset_root = "{placeholder}"', cfg_path.read_text()))
base = work / "configs/baseline.toml"
text = base.read_text()
text = re.sub(r'count_matrix = ".*"', f'count_matrix = "{work}/pytepfit/Schaefer2018_200Parcels_7Networks_count.csv"', text)
text = re.sub(r'distance_matrix = ".*"', f'distance_matrix = "{work}/pytepfit/Schaefer2018_200Parcels_7Networks_distance.csv"', text)
base.write_text(text)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(work)], check=True)
subprocess.run(["nvidia-smi"], check=False)


def run(name, args, x64="0"):
    env = dict(os.environ, MDD_TVB_JAX_X64=x64, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t = time.time()
    with open(results / f"{name}.log", "w") as log:
        proc = subprocess.run([sys.executable] + args, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"{name}: exit {proc.returncode} in {time.time() - t:.0f}s (total {time.time() - T0:.0f}s)", flush=True)
    print("\n".join((results / f"{name}.log").read_text().splitlines()[-12:]), flush=True)
    return proc.returncode == 0


R = "outputs/external/m53"
EXT = ["--emp-dir", "outputs/external/rtms_restEC/empirical",
       "--external-transformer", "outputs/external/rtms_restEC/m51/spectral_transformer.npz"]
FINAL = ["--spatial", "network_hemisphere", "--common-drive", "--source-background", "--max-residual", "1e-4"]
for lead in ("tvb", "bem"):
    pop = f"configs/m53_frozen/population_fit_{lead}.json"
    run(f"bank_{lead}", ["scripts/linear/build_analytic_bank.py", "--lead", lead, "--samples", "1600",
                         "--half-width", "1.2", "--population", pop, "--out", f"{R}/bank_{lead}.npz"])
    run(f"bank_{lead}_adaptive", ["scripts/linear/build_analytic_bank.py", "--lead", lead, "--samples", "1200",
                                  "--half-width", "0.4", "--seed", "11", "--population", pop,
                                  "--adaptive-from", f"configs/m53_frozen/stage1_subject_fits_{lead}.csv",
                                  "--adaptive-bank", f"{R}/bank_{lead}.npz", "--out", f"{R}/bank_{lead}_adaptive.npz"])
    banks = [f"{R}/bank_{lead}.npz", f"{R}/bank_{lead}_adaptive.npz"]
    if lead == "tvb":  # reproduction check of the frozen development result (bank MAP stage)
        run("repro_dev_tvb", ["scripts/linear/fit_subjects_nested.py", "--bank", *banks, "--lead", lead,
                              "--population", pop, "--out", f"{R}/repro_dev_tvb"] + FINAL)
    if run(f"ext_{lead}", ["scripts/linear/fit_subjects_nested.py", "--bank", *banks, "--lead", lead,
                           "--population", pop, "--out", f"{R}/ext_{lead}"] + FINAL + EXT):
        run(f"ext_{lead}_refine", ["scripts/linear/refine_subjects.py", "--run", f"{R}/ext_{lead}", "--bank", *banks,
                                   "--lead", lead, "--common-drive", "--source-background", "--steps", "40",
                                   "--out", f"{R}/ext_{lead}_refined"] + EXT)

import numpy as np
for path in (work / R).rglob("*"):
    if not path.is_file() or path.name.startswith("bank_"):
        continue
    rel = path.relative_to(work / R)
    target = results / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "predictions.npz":
        with np.load(path) as payload:
            np.savez_compressed(target, subject_ids=payload["subject_ids"], csd=payload["csd"].astype(np.complex64))
    elif path.suffix in {".json", ".csv"}:
        shutil.copy2(path, target)
print("ALL DONE", time.time() - T0)
