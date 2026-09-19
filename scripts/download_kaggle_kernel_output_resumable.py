"""Stream a large Kaggle notebook output with byte-range resume support.

Kaggle's CLI currently reads each output file into memory before writing it;
an interrupted large response therefore leaves a zero-byte destination.  This
helper writes a .partial file incrementally and resumes it when the server
supports HTTP byte ranges.  It never prints the signed download URL.

Kaggle's output-list endpoint serves the current notebook output even when a
version suffix is supplied; verify the downloaded summary's commit and hashes
with the experiment-specific validator before using the large bank.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi


def output_url(kernel: str, filename: str) -> str:
    api = KaggleApi()
    owner, slug, _version = api.parse_kernel_string(kernel)
    request_type = api.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]
    token = None
    while True:
        request = request_type()
        request.user_name = owner
        request.kernel_slug = slug
        api._set_paging(request, 20, token)
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            if item.file_name.replace("\\", "/") == filename.replace("\\", "/"):
                return item.url
        token = response.next_page_token
        if not token:
            raise FileNotFoundError(f"Output {filename!r} absent from {kernel}")


def download(kernel: str, filename: str, destination: Path, attempts: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already complete: {destination} ({destination.stat().st_size} bytes)")
        return
    for attempt in range(1, attempts + 1):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with requests.get(output_url(kernel, filename), headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if start:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if response.status_code != 206 or not match or int(match[1]) != start:
                        raise RuntimeError(
                            "Server did not honor byte-range resume; kept partial file"
                        )
                    expected = int(match[3])
                else:
                    expected = int(response.headers["Content-Length"])
                with partial.open("ab" if start else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                size = partial.stat().st_size
                print(f"Attempt {attempt}: {size}/{expected} bytes", flush=True)
                if size != expected:
                    raise IOError(f"Incomplete response: {size}/{expected} bytes")
                os.replace(partial, destination)
                return
        except (requests.RequestException, IOError) as error:
            size = partial.stat().st_size if partial.exists() else 0
            print(f"Attempt {attempt} interrupted at {size} bytes: {error}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(10 * attempt, 30))
    raise RuntimeError("Download attempts exhausted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/notebook (current output only)")
    parser.add_argument("filename", help="path of file within notebook output")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    download(args.kernel, args.filename, args.destination, args.attempts)


if __name__ == "__main__":
    main()
