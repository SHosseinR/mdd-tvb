"""Pack external-cohort spectra (and, when present, v2 spectra) for Kaggle."""
import tarfile, pathlib, json, sys
out = pathlib.Path('outputs/kaggle_linear/external/mddtvb_external.tar.gz')
out.parent.mkdir(parents=True, exist_ok=True)
pats = ['outputs/external/rtms_restEC/empirical/cross_spectra_fit.npz',
        'outputs/external/rtms_restEC/empirical/cross_spectra_validation.npz',
        'outputs/external/rtms_restEC/empirical/subjects.csv',
        'outputs/external/rtms_restEC/m51/spectral_transformer.npz']
pats += sys.argv[1:]
with tarfile.open(out, 'w:gz') as tar:
    for pat in pats:
        for f in sorted(pathlib.Path('.').glob(pat)):
            tar.add(f, arcname=f.as_posix())
json.dump({"title": "mdd-tvb-external", "id": "shahmadi/mdd-tvb-external",
           "licenses": [{"name": "CC0-1.0"}]}, open(out.parent / 'dataset-metadata.json', 'w'))
print(out.stat().st_size / 1e6, 'MB')
