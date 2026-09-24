import tarfile, pathlib, json
out = pathlib.Path('outputs/kaggle_linear/code/mddtvb_code.tar.gz')
with tarfile.open(out, 'w:gz') as tar:
    for pat in ['src/mdd_tvb/*.py', 'scripts/linear/*.py']:
        for f in sorted(pathlib.Path('.').glob(pat)):
            tar.add(f, arcname=f.as_posix())
json.dump({"title": "mdd-tvb-linear-code", "id": "shahmadi/mdd-tvb-linear-code",
           "licenses": [{"name": "CC0-1.0"}]}, open('outputs/kaggle_linear/code/dataset-metadata.json', 'w'))
print(out.stat().st_size)
