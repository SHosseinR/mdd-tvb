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
}
only = sys.argv[1:] or list(kernels)
for slug in only:
    stages, sources = kernels[slug]
    folder = root / "outputs/kaggle_linear" / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel.py").write_text(template.replace("__STAGES__", json.dumps(stages)))
    json.dump({"id": f"shahmadi/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
               "enable_internet": True, "dataset_sources": ["shahmadi/mdd-tvb-linear-bundle", "shahmadi/mdd-tvb-linear-code",
                                                            "shahmadi/mdd-tvb-external"],
               "kernel_sources": sources, "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
              open(folder / "kernel-metadata.json", "w"), indent=1)
    print(slug, stages, sources)
