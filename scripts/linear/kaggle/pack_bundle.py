import tarfile, pathlib, json
root = pathlib.Path('.')
out = pathlib.Path('outputs/kaggle_linear/dataset/mddtvb_bundle.tar.gz')
files = []
for pat in ['src/mdd_tvb/*.py', 'scripts/linear/*.py', 'configs/*.toml', 'configs/m52_nested_splits.csv', 'pyproject.toml', 'README.md',
            'data/atlas/*.csv', 'data/sensors/*.csv', 'data/forward/template_bem_corrected/*',
            'outputs/m5_spectral_m51_production/empirical/cross_spectra_fit.npz',
            'outputs/m5_spectral_m51_production/empirical/cross_spectra_validation.npz',
            'outputs/m52_nested_baseline/out_of_fold_subject_posteriors.csv', 'outputs/m52_nested_baseline/nested_summary.json',
            'outputs/m52_nested_baseline/outer_*/spectral_transformer.npz', 'outputs/linear_regime/population_fit_tvb.json']:
    files += sorted(root.glob(pat))
with tarfile.open(out, 'w:gz') as tar:
    for f in files:
        tar.add(f, arcname=f.as_posix())
    for name in ['count', 'distance']:
        src = pathlib.Path(f'D:/university/projects/PyTepFit/data/Schaefer2018_200Parcels_7Networks_{name}.csv')
        tar.add(src, arcname=f'pytepfit/{src.name}')
print(len(files), out.stat().st_size / 1e6, 'MB')
json.dump({"title": "mdd-tvb-linear-bundle", "id": "shahmadi/mdd-tvb-linear-bundle",
           "licenses": [{"name": "CC0-1.0"}]}, open('outputs/kaggle_linear/dataset/dataset-metadata.json', 'w'))
