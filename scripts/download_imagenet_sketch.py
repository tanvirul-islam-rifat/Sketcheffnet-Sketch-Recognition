"""
Download ImageNet-Sketch from Kaggle for the Day 6 cross-dataset experiment.

The full dataset is about 8 GB (50,000 images, 1,000 classes); Day 6 uses only
the first 50 classes. On Kaggle, attach the dataset as an input instead; the
config finds it at ``/kaggle/input/imagenetsketch/imagenet-sketch/sketch``.

The slug can be overridden with IMAGENET_SKETCH_KAGGLE_DATASET. Check it against
the dataset you attached in your Kaggle notebook.

    python scripts/download_imagenet_sketch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketcheffnet.config import (DATA_ROOT, IMAGENET_SKETCH_DIR, IMAGENET_SKETCH_KAGGLE_DATASET,  # noqa: E402
                                 ON_KAGGLE)


def main() -> None:
    if ON_KAGGLE:
        print(f"On Kaggle - attach '{IMAGENET_SKETCH_KAGGLE_DATASET}' as an input; expected at {IMAGENET_SKETCH_DIR}")
        return
    import kagglehub

    print(f"Downloading {IMAGENET_SKETCH_KAGGLE_DATASET} (about 8 GB) ...")
    cache = Path(kagglehub.dataset_download(IMAGENET_SKETCH_KAGGLE_DATASET))
    candidates = [cache / "imagenet-sketch" / "sketch", cache / "sketch", cache]
    source = next(c for c in candidates if c.is_dir())
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if IMAGENET_SKETCH_DIR.exists() or IMAGENET_SKETCH_DIR.is_symlink():
        print(f"{IMAGENET_SKETCH_DIR} already exists - left as is.")
    else:
        os.symlink(source, IMAGENET_SKETCH_DIR)
        print(f"Linked {IMAGENET_SKETCH_DIR} -> {source}")


if __name__ == "__main__":
    main()
