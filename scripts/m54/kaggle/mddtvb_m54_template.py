"""Kaggle GPU job for M5.4 (Whittle likelihood, reduced parameters, Laplace posteriors).

STAGES is filled in by make_kernels.py.  Every stage writes its results into
/kaggle/working/results immediately, so a later failure or timeout keeps them.
Datasets: mdd-tvb-linear-bundle (code, development v1 spectra, transformers),
mdd-tvb-linear-code (code overlay), mdd-tvb-external (rTMS v1 spectra, v2 spectra
of both cohorts, subject lists).
"""
import glob, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path

STAGES = __STAGES__
T0 = time.time()
work = Path("/tmp/mdd-tvb")
results = Path("/kaggle/working/results")
results.mkdir(parents=True, exist_ok=True)


def extract(name, fallback, slug):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        with tarfile.open(hits[0]) as tar:
            tar.extractall(work)
        return
    for sub in fallback:
        found = [h for h in glob.glob(f"/kaggle/input/**/{sub}", recursive=True) if slug in h]
        if found:
            shutil.copytree(found[0], work / sub, dirs_exist_ok=True)


bundle = glob.glob("/kaggle/input/**/mddtvb_bundle.tar.gz", recursive=True)
if bundle:
    with tarfile.open(bundle[0]) as tar:
        tar.extractall(work)
else:  # Kaggle auto-extracted the archive: copy the whole bundle root
    roots = [h for h in glob.glob("/kaggle/input/**/configs/baseline.toml", recursive=True) if "linear-bundle" in h]
    shutil.copytree(Path(roots[0]).parents[1], work, dirs_exist_ok=True)
extract("mddtvb_code.tar.gz", ["src/mdd_tvb", "scripts/linear", "scripts/m54", "configs/m53_frozen"], "linear-code")
extract("mddtvb_external.tar.gz", ["outputs/external", "outputs/preproc_v2", "configs/m54_lists"], "tvb-external")
for need in ("scripts/m54/m54_fit.py", "configs/m53_frozen/population_fit_tvb.json",
             "outputs/preproc_v2/dev_restEC/empirical/cross_spectra_fit.npz", "configs/m54_lists/dev_restEC_v2.txt"):
    assert (work / need).exists(), f"missing {need}"
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

R = "outputs/m54"
V2 = "outputs/preproc_v2"
V1 = "outputs/m5_spectral_m51_production/empirical"
LISTS = "configs/m54_lists"
FIX = ["--fix", "speed_mm_per_ms", "fast_ratio", "fast_fraction", "noise_tau_ms"]


def publish():
    import numpy as np
    for path in (work / R).rglob("*"):
        if not path.is_file() or path.name.startswith("bank_"):
            continue
        target = results / path.relative_to(work / R)
        if target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "predictions.npz":
            with np.load(path) as payload:
                np.savez_compressed(target, subject_ids=payload["subject_ids"], csd=payload["csd"].astype(np.complex64))
        else:
            shutil.copy2(path, target)


def run(name, args, x64="0"):
    env = dict(os.environ, MDD_TVB_JAX_X64=x64, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t = time.time()
    with open(results / f"{name}.log", "w") as log:
        proc = subprocess.run([sys.executable] + args, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"{name}: exit {proc.returncode} in {time.time() - t:.0f}s (total {time.time() - T0:.0f}s)", flush=True)
    print("\n".join((results / f"{name}.log").read_text().splitlines()[-8:]), flush=True)
    publish()
    return proc.returncode == 0


# variant -> (fit options, population/bank tag, population options)
VARIANTS = {"full": ([], "", []), "m10": (["--modes", "10"], "_m10", ["--modes", "10"]),
            "m10pop": (["--modes", "10", "--pop-background"], "_m10", ["--modes", "10"])}


def fit(name, lead, emp, extra, variant="full"):
    opts, ptag, _ = VARIANTS[variant]
    return run(name, ["scripts/m54/m54_fit.py", "--lead", lead, "--population", f"{R}/population_{lead}{ptag}.json",
                      "--bank", f"{R}/bank_{lead}{ptag}.npz", "--emp-dir", emp, "--out", f"{R}/{name}"] + extra + opts)


def population(lead, variant):
    _, ptag, popts = VARIANTS[variant]
    run(f"population_{lead}{ptag}", ["scripts/m54/m54_population.py", "--lead", lead, "--emp-dir", f"{V2}/dev_restEC/empirical",
                                     "--subjects-file", f"{LISTS}/dev_restEC_v2.txt", "--maxiter", "60",
                                     "--start", f"configs/m53_frozen/population_fit_{lead}.json",
                                     "--out", f"{R}/population_{lead}{ptag}.json"] + popts, x64="1")
    bank(lead, ptag)


def bank(lead, ptag):
    run(f"bank_{lead}{ptag}", ["scripts/linear/build_analytic_bank.py", "--lead", lead, "--samples", "2000", "--half-width", "1.2",
                               "--population", f"{R}/population_{lead}{ptag}.json", "--out", f"{R}/bank_{lead}{ptag}.npz"] + FIX)


DEV = ["--subjects-file", f"{LISTS}/dev_restEC_v2.txt"]
EXT = ["--subjects-file", f"{LISTS}/rtms_restEC_v2.txt", "--null-emp-dir", f"{V2}/dev_restEC/empirical",
       "--dev-subjects-file", f"{LISTS}/dev_restEC_v2.txt"]
for stage in STAGES:
    parts = stage.split(":")
    kind, lead = parts[0], parts[1]
    variant = parts[2] if len(parts) > 2 else "full"
    vtag = "" if variant == "full" else f"_{variant}"
    if kind == "population":
        population(lead, variant)
    elif kind == "dev":
        fit(f"dev_{lead}{vtag}", lead, f"{V2}/dev_restEC/empirical", DEV, variant)
    elif kind == "ext":
        fit(f"ext_{lead}{vtag}", lead, f"{V2}/rtms_restEC/empirical", EXT, variant)
    elif kind == "devswap":
        fit(f"dev_{lead}{vtag}_swap", lead, f"{V2}/dev_restEC/empirical", DEV + ["--swap-halves"], variant)
    elif kind == "extswap":
        fit(f"ext_{lead}{vtag}_swap", lead, f"{V2}/rtms_restEC/empirical", EXT + ["--swap-halves"], variant)
    elif kind == "devv1":  # objective effect on the original preprocessing (same subjects)
        fit(f"dev_{lead}{vtag}_v1", lead, V1, DEV, variant)
    elif kind == "devnomask":  # muscle-channel masking switched off
        fit(f"dev_{lead}{vtag}_nomask", lead, f"{V2}/dev_restEC/empirical", DEV + ["--no-emg-mask"], variant)
    elif kind == "synth":  # synthetic recovery from a finished M5.4 job (attached as a kernel source)
        _, ptag, _ = VARIANTS[variant]
        (work / R).mkdir(parents=True, exist_ok=True)
        for rel in (f"population_{lead}{ptag}.json", f"dev_{lead}{vtag}/subject_fits.csv"):
            hits = [h for h in glob.glob(f"/kaggle/input/**/{rel}", recursive=True) if "m54" in h]
            (work / R / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hits[0], work / R / rel)
        bank(lead, ptag)
        S = f"{R}/synthetic_{lead}{vtag}"
        gen = ["scripts/m54/m54_synthetic.py", "generate", "--lead", lead, "--fits", f"{R}/dev_{lead}{vtag}/subject_fits.csv",
               "--population", f"{R}/population_{lead}{ptag}.json", "--emp-dir", f"{V2}/dev_restEC/empirical",
               "--n", "120", "--out", S]
        if variant == "m10pop":
            gen += ["--pop-background"]
        run(f"synthetic_{lead}{vtag}_generate", gen, x64="1")
        if fit(f"synthetic_{lead}{vtag}_fit", lead, f"{S}/empirical", ["--subjects-file", f"{S}/subjects.txt"], variant):
            run(f"synthetic_{lead}{vtag}_summary", ["scripts/m54/m54_synthetic.py", "summarise", "--out", S,
                                                    "--recovered", f"{R}/synthetic_{lead}{vtag}_fit/subject_fits.csv"])
    elif kind == "m53v2":  # frozen M5.3 pipeline on the v2 spectra (TVB)
        pop = "configs/m53_frozen/population_fit_tvb.json"
        B = "outputs/m53v2"
        run("m53_bank", ["scripts/linear/build_analytic_bank.py", "--lead", "tvb", "--samples", "1600", "--half-width", "1.2",
                         "--population", pop, "--out", f"{B}/bank_tvb.npz"])
        run("m53_bank_adaptive", ["scripts/linear/build_analytic_bank.py", "--lead", "tvb", "--samples", "1200",
                                  "--half-width", "0.4", "--seed", "11", "--population", pop,
                                  "--adaptive-from", "configs/m53_frozen/stage1_subject_fits_tvb.csv",
                                  "--adaptive-bank", f"{B}/bank_tvb.npz", "--out", f"{B}/bank_tvb_adaptive.npz"])
        banks = [f"{B}/bank_tvb.npz", f"{B}/bank_tvb_adaptive.npz"]
        FINAL = ["--spatial", "network_hemisphere", "--common-drive", "--source-background", "--max-residual", "1e-4"]
        for tag, emp, extra in (("dev", f"{V2}/dev_restEC/empirical", DEV[:2]),
                                ("ext", f"{V2}/rtms_restEC/empirical",
                                 ["--subjects-file", f"{LISTS}/rtms_restEC_v2.txt", "--external-transformer",
                                  "outputs/external/rtms_restEC/m51/spectral_transformer.npz",
                                  "--null-emp-dir", f"{V2}/dev_restEC/empirical"])):
            if run(f"m53v2_{tag}", ["scripts/linear/fit_subjects_nested.py", "--bank", *banks, "--lead", "tvb",
                                    "--population", pop, "--emp-dir", emp, "--out", f"{R}/m53v2_{tag}"] + FINAL + extra):
                run(f"m53v2_{tag}_refine", ["scripts/linear/refine_subjects.py", "--run", f"{R}/m53v2_{tag}", "--bank", *banks,
                                            "--lead", "tvb", "--common-drive", "--source-background", "--steps", "40",
                                            "--emp-dir", emp, "--out", f"{R}/m53v2_{tag}_refined"] + extra)
publish()
print("ALL DONE", time.time() - T0)
