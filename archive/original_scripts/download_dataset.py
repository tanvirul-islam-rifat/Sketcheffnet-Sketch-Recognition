"""
Download the QuickDraw dataset from Kaggle and stage it at local_config.DATA_DIR.

Prerequisites:
  1. A Kaggle account + API token. Create one at https://www.kaggle.com/settings
     -> "Create New Token", which downloads kaggle.json. Place it at
     ~/.kaggle/kaggle.json (chmod 600), or set the KAGGLE_USERNAME /
     KAGGLE_KEY environment variables.
  2. Update KAGGLE_DATASET in local_config.py (or set the
     QUICKDRAW_KAGGLE_DATASET env var) to the exact "owner/dataset-slug"
     of your uploaded QuickDraw dataset.

Usage:
    python download_dataset.py
"""
import os
import shutil

import kagglehub

from local_config import KAGGLE_DATASET, DATA_DIR, DATA_ROOT


def main():
    if 'CHANGE_ME' in KAGGLE_DATASET:
        raise SystemExit(
            "Set QUICKDRAW_KAGGLE_DATASET (or edit KAGGLE_DATASET in local_config.py) "
            "to your Kaggle dataset's 'owner/dataset-slug' before downloading."
        )

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Downloaded to kagglehub cache: {cache_path}")

    # The dataset may already contain a 'quickdraw_data' subfolder (matching
    # the original /kaggle/input/quickdraw20/quickdraw_data layout), or the
    # .npy files may sit directly at the dataset root. Handle both.
    candidate = os.path.join(cache_path, 'quickdraw_data')
    source_dir = candidate if os.path.isdir(candidate) else cache_path

    os.makedirs(DATA_ROOT, exist_ok=True)
    if os.path.islink(DATA_DIR) or os.path.exists(DATA_DIR):
        print(f"{DATA_DIR} already exists, leaving it as-is.")
    else:
        os.symlink(source_dir, DATA_DIR)
        print(f"Linked {DATA_DIR} -> {source_dir}")

    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    print(f"\n✓ Ready. Found {len(npy_files)} .npy files in {DATA_DIR}")


if __name__ == '__main__':
    main()
