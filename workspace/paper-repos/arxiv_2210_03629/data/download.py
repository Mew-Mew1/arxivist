#!/usr/bin/env python
"""
data/download.py
=================
Downloads the two automatable datasets (HotpotQA, FEVER) used by this repo.
ALFWorld and WebShop require separate, non-automated setup -- see
data/README_data.md; this script prints those instructions rather than
attempting to download them.

Paper section: Section 3.1 (HotpotQA, FEVER); Section 4 (ALFWorld, WebShop).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

# Official source URLs. No MD5/SHA256 is published by the paper itself for
# these third-party benchmark files, so integrity is verified by a
# reasonable-file-size sanity check rather than a checksum -- see
# `_sanity_check_size` below.
_HOTPOTQA_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
_FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"

_MIN_EXPECTED_BYTES = {
    "hotpotqa": 10_000_000,  # ~40MB official file; flag anything suspiciously small (e.g. an HTML error page)
    "fever": 5_000_000,  # ~14MB official file
}


def _sanity_check_size(path: str, dataset: str) -> None:
    size = os.path.getsize(path)
    if size < _MIN_EXPECTED_BYTES.get(dataset, 0):
        print(
            f"WARNING: downloaded file {path} is only {size} bytes, smaller "
            f"than expected for {dataset}. The download may have failed "
            f"silently (e.g. redirected to an error page). Please verify "
            f"manually.",
            file=sys.stderr,
        )


def _download(url: str, dest_path: str, dataset: str) -> None:
    if os.path.exists(dest_path):
        print(f"[{dataset}] already exists at {dest_path}, skipping download.")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[{dataset}] downloading {url} -> {dest_path} ...")

    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r[{dataset}] {pct:5.1f}% ({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print()
    except Exception as e:
        print(f"\n[{dataset}] ERROR: download failed: {e}", file=sys.stderr)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)

    _sanity_check_size(dest_path, dataset)
    print(f"[{dataset}] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HotpotQA / FEVER data. See data/README_data.md for ALFWorld/WebShop.")
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "fever", "alfworld", "webshop", "all"])
    parser.add_argument("--data-root", type=str, default="data", help="Root directory to download into.")
    args = parser.parse_args()

    targets = ["hotpotqa", "fever", "alfworld", "webshop"] if args.dataset == "all" else [args.dataset]

    for dataset in targets:
        if dataset == "hotpotqa":
            _download(_HOTPOTQA_DEV_URL, os.path.join(args.data_root, "hotpotqa", "hotpot_dev_distractor_v1.json"), "hotpotqa")
        elif dataset == "fever":
            _download(_FEVER_DEV_URL, os.path.join(args.data_root, "fever", "shared_task_dev.jsonl"), "fever")
        elif dataset in ("alfworld", "webshop"):
            print(
                f"[{dataset}] is not automatically downloadable by this script. "
                f"See data/README_data.md for the required manual setup steps "
                f"(`pip install alfworld && alfworld-download`, or cloning and "
                f"running the WebShop server)."
            )


if __name__ == "__main__":
    main()
