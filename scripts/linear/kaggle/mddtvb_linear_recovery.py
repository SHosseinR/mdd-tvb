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
    for sub in ("src/mdd_tvb", "scripts/linear", "outputs/linear_regime/kaggle"):
        hits = [h for h in glob.glob(f"/kaggle/input/**/{sub}", recursive=True) if "linear-code" in h]
        if hits:
            shutil.copytree(hits[0], work / sub, dirs_exist_ok=True)
# Guard against a stale code-dataset version: required options must exist.
fsn = (work / "scripts/linear/fit_subjects_nested.py").read_text()
assert "--max-residual" in fsn and "--common-drive" in fsn, "stale code dataset version attached"
assert "m,mfcd->fcd" in (work / "scripts/linear/posterior_average.py").read_text(), "stale posterior script"
assert "--emp-dir" in (work / "scripts/linear/refine_subjects.py").read_text(), "stale refine script"
assert (work / "scripts/linear/synthetic_recovery_linear.py").is_file(), "missing recovery script"
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
S = f"{R}/synthetic_recovery"
SYN = ["--emp-dir", S, "--subjects-file", f"{S}/subjects.txt"]
run("generate", ["scripts/linear/synthetic_recovery_linear.py", "generate", "--n", "100", "--dof", "60", "--out", S], x64="1")
run("bank", ["scripts/linear/build_analytic_bank.py", "--lead", "tvb", "--samples", "1600", "--half-width", "1.2",
             "--population", f"{R}/population_fit_tvb.json", "--out", f"{R}/bank_tvb.npz"])
run("stage1", ["scripts/linear/fit_subjects_nested.py", "--bank", f"{R}/bank_tvb.npz", "--lead", "tvb",
               "--population", f"{R}/population_fit_tvb.json", "--max-residual", "1e-4", "--out", f"{S}/stage1"] + FINAL + SYN)
run("bank_adaptive", ["scripts/linear/build_analytic_bank.py", "--lead", "tvb", "--samples", "1200", "--half-width", "0.4",
                      "--seed", "11", "--population", f"{R}/population_fit_tvb.json",
                      "--adaptive-from", f"{S}/stage1/subject_fits.csv", "--adaptive-bank", f"{R}/bank_tvb.npz",
                      "--out", f"{R}/bank_tvb_adaptive.npz"])
banks = [f"{R}/bank_tvb.npz", f"{R}/bank_tvb_adaptive.npz"]
run("final", ["scripts/linear/fit_subjects_nested.py", "--bank", *banks, "--lead", "tvb",
              "--population", f"{R}/population_fit_tvb.json", "--max-residual", "1e-4", "--out", f"{S}/final"] + FINAL + SYN)
run("refine", ["scripts/linear/refine_subjects.py", "--run", f"{S}/final", "--bank", *banks, "--lead", "tvb",
               "--common-drive", "--source-background", "--steps", "40", "--out", f"{S}/final_refined"] + SYN)
run("summarise", ["scripts/linear/synthetic_recovery_linear.py", "summarise", "--out", S,
                  "--map-run", f"{S}/final", "--refined-run", f"{S}/final_refined"])
for path in (work / S).rglob("*"):
    if path.is_file() and path.suffix in {".json", ".csv", ".txt"}:
        target = results / path.relative_to(work / R)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
print("ALL DONE", time.time() - T0)
