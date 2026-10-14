"""
Download the 50-category QuickDraw subset from Kaggle and link it into ``data/quickdraw_data``.

Prerequisite: a Kaggle API token in ``~/.kaggle/kaggle.json`` (chmod 600), or the
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.
Not needed on Kaggle, where the dataset is attached as an input.

    python scripts/download_quickdraw.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketcheffnet.config import DATA_ROOT, ON_KAGGLE, QUICKDRAW_DIR, QUICKDRAW_KAGGLE_DATASET  # noqa: E402


def main() -> None:
    if ON_KAGGLE:
        print(f"On Kaggle - attach '{QUICKDRAW_KAGGLE_DATASET}' as an input instead; expected at {QUICKDRAW_DIR}")
        return
    import kagglehub

    print(f"Downloading {QUICKDRAW_KAGGLE_DATASET} (about 5 GB) ...")
    cache = Path(kagglehub.dataset_download(QUICKDRAW_KAGGLE_DATASET))
    source = cache / "quickdraw_data" if (cache / "quickdraw_data").is_dir() else cache
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if QUICKDRAW_DIR.exists() or QUICKDRAW_DIR.is_symlink():
        print(f"{QUICKDRAW_DIR} already exists - left as is.")
    else:
        os.symlink(source, QUICKDRAW_DIR)
        print(f"Linked {QUICKDRAW_DIR} -> {source}")
    n = len(list(Path(QUICKDRAW_DIR).glob("*.npy")))
    print(f"Ready: {n} .npy files in {QUICKDRAW_DIR}")


if __name__ == "__main__":
    main()
