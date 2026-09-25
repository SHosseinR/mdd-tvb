import tarfile, pathlib, json
out = pathlib.Path('outputs/kaggle_linear/code/mddtvb_code.tar.gz')
out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(out, 'w:gz') as tar:
    for pat in ['src/mdd_tvb/*.py', 'scripts/linear/*.py', 'scripts/m54/*.py', 'configs/m53_frozen/*',
                'outputs/linear_regime/kaggle/final/final_tvb/subject_fits.csv',
                'outputs/linear_regime/kaggle/final/final_tvb_refined/subject_fits.csv',
                'outputs/linear_regime/kaggle/final/final_bem_refined/subject_fits.csv']:
        for f in sorted(pathlib.Path('.').glob(pat)):
            tar.add(f, arcname=f.as_posix())
json.dump({"title": "mdd-tvb-linear-code", "id": "shahmadi/mdd-tvb-linear-code",
           "licenses": [{"name": "CC0-1.0"}]}, open('outputs/kaggle_linear/code/dataset-metadata.json', 'w'))
print(out.stat().st_size)
