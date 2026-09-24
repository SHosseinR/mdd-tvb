"""Kaggle GPU job: linear-regime analytic fitting with the fsaverage BEM lead field.

Self-contained: code + data come from the private dataset
shahmadi/mdd-tvb-linear-bundle.  Evaluation is the M5.2 nested protocol on the
262 development subjects (unseen second halves only).
"""
import glob, os, re, shutil, subprocess, sys, tarfile, time, json
from pathlib import Path

T0 = time.time()
work = Path("/tmp/mdd-tvb")  # keep the extracted repo out of /kaggle/working outputs
results = Path("/kaggle/working/results")
results.mkdir(parents=True, exist_ok=True)
bundle = glob.glob("/kaggle/input/**/mddtvb_bundle.tar.gz", recursive=True)
if bundle:
    with tarfile.open(bundle[0]) as tar:
        tar.extractall(work)
else:  # Kaggle may have auto-extracted the archive
    configs = [h for h in glob.glob("/kaggle/input/**/configs/baseline.toml", recursive=True)]
    print("bundle candidates:", configs, flush=True)
    shutil.copytree(Path(configs[0]).parents[1], work, dirs_exist_ok=True)
# Overlay the latest code (small, separately versioned dataset).
code = glob.glob("/kaggle/input/**/mddtvb_code.tar.gz", recursive=True)
if code:
    with tarfile.open(code[0]) as tar:
        tar.extractall(work)
else:
    for sub in ("src/mdd_tvb", "scripts/linear"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{sub}", recursive=True) if "linear-code" in h]
        if hits:
            shutil.copytree(hits[0], work / sub, dirs_exist_ok=True)
# Guard against a stale code-dataset version: required options must exist.
fsn = (work / "scripts/linear/fit_subjects_nested.py").read_text()
assert "--max-residual" in fsn and "--common-drive" in fsn, "stale code dataset version attached"
assert "m,mfcd->fcd" in (work / "scripts/linear/posterior_average.py").read_text(), "stale posterior script"
assert "--source-background" in (work / "scripts/linear/refine_subjects.py").read_text(), "stale refine script"
print("linear_jax has NaN guard:", "LAPACK eig can hang" in (work / "src/mdd_tvb/linear_jax.py").read_text(), flush=True)
# The spectral config validates the raw-EEG folder even though only the
# extracted spectra are used here.
placeholder = work / "raw_eeg_placeholder"
placeholder.mkdir(exist_ok=True)
for cfg_path in (work / "configs").glob("*.toml"):
    cfg_text = cfg_path.read_text()
    cfg_path.write_text(re.sub(r'dataset_root = ".*"', f'dataset_root = "{placeholder}"', cfg_text))
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
    tail = (results / f"{name}.log").read_text().splitlines()[-25:]
    print("\n".join(tail), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed")


run("check_devices", ["-c", "import jax; print(jax.devices()); import mdd_tvb.linear_jax as L; print('ok')"])
R = "outputs/linear_regime"


def step(name, args, x64="0"):
    try:
        run(name, args, x64)
        return True
    except SystemExit as error:
        print("step failed:", error, flush=True)
        return False


FINAL = ["--spatial", "network_hemisphere", "--common-drive", "--source-background"]
for lead in ("tvb", "bem"):
    run(f"population_fit_{lead}", ["scripts/linear/population_fit.py", "--lead", lead, "--maxiter", "80",
                                   "--out", f"{R}/population_fit_{lead}.json"], x64="1")
    run(f"bank_{lead}", ["scripts/linear/build_analytic_bank.py", "--lead", lead, "--samples", "1600",
                         "--half-width", "1.2", "--population", f"{R}/population_fit_{lead}.json",
                         "--out", f"{R}/bank_{lead}.npz"])
    first = f"{R}/final_{lead}_stage1"
    step(f"final_{lead}_stage1", ["scripts/linear/fit_subjects_nested.py", "--bank", f"{R}/bank_{lead}.npz",
                                  "--lead", lead, "--population", f"{R}/population_fit_{lead}.json",
                                  "--max-residual", "1e-4", "--out", first] + FINAL)
    run(f"bank_{lead}_adaptive", ["scripts/linear/build_analytic_bank.py", "--lead", lead, "--samples", "1200",
                                  "--half-width", "0.4", "--seed", "11",
                                  "--population", f"{R}/population_fit_{lead}.json",
                                  "--adaptive-from", f"{first}/subject_fits.csv",
                                  "--adaptive-bank", f"{R}/bank_{lead}.npz",
                                  "--out", f"{R}/bank_{lead}_adaptive.npz"])
    banks = [f"{R}/bank_{lead}.npz", f"{R}/bank_{lead}_adaptive.npz"]
    final = f"{R}/final_{lead}"
    if step(f"final_{lead}", ["scripts/linear/fit_subjects_nested.py", "--bank", *banks, "--lead", lead,
                              "--population", f"{R}/population_fit_{lead}.json", "--max-residual", "1e-4",
                              "--out", final] + FINAL):
        step(f"final_{lead}_compare_map", ["scripts/linear/compare_with_m51.py", "--run", final,
                                           "--label", f"FINAL {lead}: drive+hemisphere+source bg (MAP)"])
        step(f"final_{lead}_posterior", ["scripts/linear/posterior_average.py", "--run", final])
        step(f"final_{lead}_compare_post", ["scripts/linear/compare_with_m51.py", "--run", f"{final}/posterior",
                                            "--label", f"FINAL {lead}: drive+hemisphere+source bg (posterior)"])
        if step(f"final_{lead}_refine", ["scripts/linear/refine_subjects.py", "--run", final, "--bank", *banks,
                                         "--lead", lead, "--common-drive", "--source-background", "--steps", "40",
                                         "--out", f"{final}_refined"]):
            step(f"final_{lead}_compare_refined", ["scripts/linear/compare_with_m51.py", "--run", f"{final}_refined",
                                                   "--label", f"FINAL {lead}: + continuous refinement"])
import numpy as np
for path in (work / R).rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(work / R)
    target = results / rel
    if path.suffix in {".json", ".csv", ".png"}:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    elif path.name == "predictions.npz" and str(rel).startswith("final_") and "stage1" not in str(rel):
        target.parent.mkdir(parents=True, exist_ok=True)
        with np.load(path) as payload:
            np.savez_compressed(target, subject_ids=payload["subject_ids"], csd=payload["csd"].astype(np.complex64))
print("ALL DONE", time.time() - T0)
