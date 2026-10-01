"""Write the M5.4 Kaggle kernel folders from the template."""
import json, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[3]
template = (root / "scripts/m54/kaggle/mddtvb_m54_template.py").read_text()
kernels = {
    "mddtvb-m54-tvb": (["population:tvb", "dev:tvb", "ext:tvb", "devswap:tvb", "extswap:tvb", "devv1:tvb", "devnomask:tvb"], []),
    "mddtvb-m54-bem": (["population:bem", "dev:bem", "ext:bem", "m53v2:tvb", "devswap:bem"], []),
    "mddtvb-m54b-tvb": (["population:tvb:m10", "dev:tvb:m10pop", "dev:tvb:m10", "ext:tvb:m10pop", "ext:tvb:m10",
                         "devswap:tvb:m10pop", "extswap:tvb:m10pop"], []),
    "mddtvb-m54b-bem": (["population:bem:m10", "dev:bem:m10pop", "dev:bem:m10", "ext:bem:m10pop", "ext:bem:m10",
                         "devswap:bem:m10pop"], []),
    "mddtvb-m54-synth": (["synth:bem:m10pop"], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54-ctrl": (["ctrl:bem:m10pop"], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54-smc": (["ctrl:bem:m10pop:smc"], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54-newds": (["newds:bem:m10pop:modma", "newdsswap:bem:m10pop:modma",
                          "newds:bem:m10pop:mumtaz", "newdsswap:bem:m10pop:mumtaz"], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54d-somot": (["reuse:bem:m10popsomot", "dev:bem:m10popsomot", "synth:bem:m10popsomot"],
                          ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54-ds003478": (["newds:bem:m10pop:ds003478", "newdsswap:bem:m10pop:ds003478"], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54c-fast": (["reuse:bem:m10popfast", "dev:bem:m10popfast", "synth:bem:m10popfast"],
                         ["shahmadi/mddtvb-m54b-bem"]),
    # corticothalamic loop: population fit with the loop, T1 / T2 subject fits, synthetic recovery
    "mddtvb-m54t-thal": (["reuse:bem:m10", "script:bem:m10:scripts/m54/thal_scan.py",
                          "population:bem:m10popthal1", "dev:bem:m10popthal1", "bank:bem:m10popthal2",
                          "dev:bem:m10popthal2", "synth:bem:m10popthal1", "synth:bem:m10popthal2"],
                         ["shahmadi/mddtvb-m54b-bem"]),
    # joint eyes-closed + eyes-open fits: which parameters change when the eyes open (100 dev subjects)
    "mddtvb-m54e-joint1": ([f"joint:bem:m10pop:dev:{d}:configs/m54_lists/dev_eoec_compare100.txt"
                            for d in ("none", "level", "mu", "b_scale", "common")], ["shahmadi/mddtvb-m54b-bem"]),
    "mddtvb-m54e-joint2": ([f"joint:bem:m10pop:dev:{d}:configs/m54_lists/dev_eoec_compare100.txt"
                            for d in ("vis", "a_scale", "coupling", "all")], ["shahmadi/mddtvb-m54b-bem"]),
    # production joint fits with the best mechanism (visual-network drive changes with eyes open)
    "mddtvb-m54e-joint3a": (["joint:bem:m10pop:dev:vis:configs/m54_lists/dev_eoec_v2.txt",
                             "jointswap:bem:m10pop:dev:vis:configs/m54_lists/dev_eoec_v2.txt",
                             "joint:bem:m10pop:rtms:vis", "joint:bem:m10pop:controls:vis", "joint:bem:m10pop:smc:vis"],
                            ["shahmadi/mddtvb-m54b-bem"]),
    # T2 thalamus (per-subject loop delay) with wide-band certification, on every cohort (replication)
    "mddtvb-m54t-thal2": (["reuse:bem:m10popthal2", "dev:bem:m10popthal2", "devswap:bem:m10popthal2",
                           "ext:bem:m10popthal2", "ctrl:bem:m10popthal2", "ctrl:bem:m10popthal2:smc",
                           "newds:bem:m10popthal2:modma", "newds:bem:m10popthal2:mumtaz",
                           "newds:bem:m10popthal2:ds003478"], ["shahmadi/mddtvb-m54t-thal"]),
    # shared thalamic nuclei: population fit (matrix share), S1 / S2 subject fits, recovery
    "mddtvb-m54s-shared": (["script:bem:m10:scripts/m54/winding_check.py",
                            "population:bem:m10popthalS1", "dev:bem:m10popthalS1", "bank:bem:m10popthalS2",
                            "dev:bem:m10popthalS2", "synth:bem:m10popthalS2"], ["shahmadi/mddtvb-m54t-thal2"]),
    "mddtvb-m54e-joint3b": (["joint:bem:m10pop:ds003478:vis", "jointswap:bem:m10pop:ds003478:vis",
                             "joint:bem:m10pop:mumtaz:vis", "jointswap:bem:m10pop:mumtaz:vis",
                             "joint:bem:m10pop:dev:all:configs/m54_lists/dev_eoec_compare100.txt"],
                            ["shahmadi/mddtvb-m54b-bem"]),
}
EO_KERNELS = {"mddtvb-m54t-thal", "mddtvb-m54e-joint1", "mddtvb-m54e-joint2"}  # need shahmadi/mdd-tvb-eo
only = sys.argv[1:] or list(kernels)
for slug in only:
    stages, sources = kernels[slug]
    folder = root / "outputs/kaggle_linear" / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(template.replace("__STAGES__", json.dumps(stages)))
    json.dump({"id": f"shahmadi/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
               "enable_internet": True, "dataset_sources": ["shahmadi/mdd-tvb-linear-bundle", "shahmadi/mdd-tvb-linear-code",
                                                            "shahmadi/mdd-tvb-external"]
               + (["shahmadi/mdd-tvb-eo"] if slug.startswith(("mddtvb-m54e", "mddtvb-m54t", "mddtvb-m54f"))
                  or slug in EO_KERNELS else []),
               "kernel_sources": sources, "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
              open(folder / "kernel-metadata.json", "w"), indent=1)
    print(slug, stages, sources)
