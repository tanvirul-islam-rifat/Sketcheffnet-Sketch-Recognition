"""
Download a small subset of the TU-Berlin sketch dataset file by file.

kagglehub can only fetch the whole archive (about 4 GB). This script uses the
``kaggle`` CLI instead: it pages through ``kaggle datasets files`` (a metadata
listing, no image data) to find N files per wanted category, then fetches only
those with ``kaggle datasets download -f <path>``. Disk use stays in the tens of MB.

Requires the ``kaggle`` CLI (``pip install kaggle``) and a Kaggle API token.

    python scripts/download_tuberlin_subset.py          # the shared categories in config
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketcheffnet.config import (TUBERLIN_DIR, TUBERLIN_KAGGLE_DATASET,  # noqa: E402
                                 TUBERLIN_SAMPLES_PER_CATEGORY, TUBERLIN_SHARED_CATEGORIES)

MAX_PAGES = 60
PAGE_SIZE = 200


def _list_files_page(page_token=None):
    cmd = ["kaggle", "datasets", "files", TUBERLIN_KAGGLE_DATASET, "--format", "csv", "--page-size", str(PAGE_SIZE)]
    if page_token:
        cmd += ["--page-token", page_token]
    lines = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
    next_token, start = None, 0
    for i, line in enumerate(lines):
        if line.startswith("Next Page Token"):
            next_token, start = line.split("=", 1)[1].strip(), i + 1
            break
    return list(csv.DictReader(lines[start:])), next_token


def _find_candidate_files(categories, per_category):
    wanted = set(categories)
    found = {c: [] for c in categories}
    token = None
    for _ in range(MAX_PAGES):
        rows, token = _list_files_page(token)
        for row in rows:
            parts = (row.get("name") or "").split("/")
            if len(parts) >= 3 and parts[-2] in wanted and len(found[parts[-2]]) < per_category:
                found[parts[-2]].append(row["name"])
        if all(len(v) >= per_category for v in found.values()) or not token:
            break
    return found


def ensure_tuberlin_categories(categories, samples_per_category=20) -> Path:
    """Make sure TUBERLIN_DIR/<category>/ holds at least N images for each category."""
    TUBERLIN_DIR.mkdir(parents=True, exist_ok=True)
    missing = [c for c in categories
               if not (TUBERLIN_DIR / c).is_dir() or len(os.listdir(TUBERLIN_DIR / c)) < samples_per_category]
    if not missing:
        print("All requested TU-Berlin categories are already present.")
        return TUBERLIN_DIR
    print(f"Listing files for {len(missing)} categories: {missing}")
    for cat, files in _find_candidate_files(missing, samples_per_category).items():
        cat_dir = TUBERLIN_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for fpath in files:
            if (cat_dir / Path(fpath).name).exists():
                continue
            subprocess.run(["kaggle", "datasets", "download", "-d", TUBERLIN_KAGGLE_DATASET,
                            "-f", fpath, "-p", str(cat_dir), "--force"], capture_output=True, text=True, check=True)
        print(f"  {cat}: {len(os.listdir(cat_dir))} files")
    return TUBERLIN_DIR


if __name__ == "__main__":
    ensure_tuberlin_categories(TUBERLIN_SHARED_CATEGORIES, TUBERLIN_SAMPLES_PER_CATEGORY)
