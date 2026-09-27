"""Download the batch-free depression EEG datasets.

* OpenNeuro ds003478 "EEG: Depression rest" (Cavanagh; public S3 bucket, no credentials);
  ``--runs 1`` fetches only run-01 (the pre-task rest) plus all metadata.
* Mumtaz 2016 "MDD Patients and Healthy Controls EEG Data (New)" (figshare 4244171).
Files already present with the right size are skipped, so the script can be re-run.
"""
from __future__ import annotations

import argparse, json, re, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3 = "https://s3.amazonaws.com/openneuro.org"


def s3_listing(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{S3}?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml, re.S)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(url, dest: Path, size: int | None, retries=5):
    if dest.is_file() and (size is None or dest.stat().st_size == size):
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            if size is not None and tmp.stat().st_size != size:  # connection closed early without an error
                raise OSError(f"truncated: {tmp.stat().st_size} of {size} bytes")
            tmp.replace(dest)
            return dest.stat().st_size
        except Exception as error:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: {error!r}")
            time.sleep(2 * (attempt + 1))


def run_pool(jobs, workers, label):
    done, total, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch, *j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), start=1):
            try:
                total += fut.result()
            except RuntimeError as error:  # log and continue (e.g. a file withdrawn from the repository)
                print(f"{label}: FAILED {error}", flush=True)
            if k % 50 == 0 or k == len(futures):
                print(f"{label}: {k}/{len(futures)} files, {total / 1e9:.2f} GB new, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/university/projects/mdd-new-datasets")
    ap.add_argument("--which", nargs="+", default=["mumtaz", "ds003478"])
    ap.add_argument("--runs", default="1", help="ds003478 runs to fetch: '1' or 'all'")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    if "mumtaz" in args.which:
        meta = json.loads(urllib.request.urlopen("https://api.figshare.com/v2/articles/4244171", timeout=60).read())
        jobs = [(f["download_url"], out / "mumtaz2016" / f["name"], f["size"]) for f in meta["files"]]
        (out / "mumtaz2016").mkdir(parents=True, exist_ok=True)
        (out / "mumtaz2016" / "figshare_metadata.json").write_text(json.dumps(meta, indent=1))
        run_pool(jobs, args.workers, "mumtaz")
    if "ds003478" in args.which:
        keys = s3_listing("ds003478/")
        if args.runs != "all":
            keys = [(k, s) for k, s in keys if "_run-" not in k or f"_run-0{args.runs}_" in k]
        jobs = [(f"{S3}/{urllib.parse.quote(k)}", out / k, s) for k, s in keys]
        print(f"ds003478: {len(jobs)} files, {sum(s for _, s in keys) / 1e9:.2f} GB", flush=True)
        run_pool(jobs, args.workers, "ds003478")


if __name__ == "__main__":
    main()
