"""
Central configuration: paths, the category list and device selection.

Every script imports from here instead of hardcoding paths, so the same code
runs in three places without edits:

* **Kaggle** - detected automatically; reads ``/kaggle/input/...`` and writes
  to ``/kaggle/working``.
* **Locally** - datasets under ``./data``, outputs under ``./outputs``.
* **Anywhere else** - override any path with an environment variable.

Environment variables
---------------------
QUICKDRAW_DATA_DIR      folder holding the 50 ``<category>.npy`` files
IMAGENET_SKETCH_DIR     folder holding ImageNet-Sketch class folders (``n01440764/`` ...)
TUBERLIN_DATA_DIR       folder holding TU-Berlin class folders
SKETCH_OUTPUT_DIR       where processed data, checkpoints and run results are written
QUICKDRAW_KAGGLE_DATASET / IMAGENET_SKETCH_KAGGLE_DATASET / TUBERLIN_KAGGLE_DATASET
                        Kaggle dataset slugs used by the download scripts
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").is_dir()

# --------------------------------------------------------------------------
# Kaggle dataset slugs (used by scripts/download_*.py)
# --------------------------------------------------------------------------
QUICKDRAW_KAGGLE_DATASET = os.environ.get("QUICKDRAW_KAGGLE_DATASET", "mdtanvirulislamrifat/quickdraw20")
IMAGENET_SKETCH_KAGGLE_DATASET = os.environ.get("IMAGENET_SKETCH_KAGGLE_DATASET", "wanghaohan/imagenetsketch")
TUBERLIN_KAGGLE_DATASET = os.environ.get("TUBERLIN_KAGGLE_DATASET", "zara2099/tu-berlin-hand-sketch-image-dataset")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
if ON_KAGGLE:
    _default_qd = "/kaggle/input/quickdraw20/quickdraw_data"
    _default_ims = "/kaggle/input/imagenetsketch/imagenet-sketch/sketch"
    _default_out = "/kaggle/working"
else:
    _default_qd = str(PROJECT_ROOT / "data" / "quickdraw_data")
    _default_ims = str(PROJECT_ROOT / "data" / "imagenet_sketch")
    _default_out = str(PROJECT_ROOT / "outputs")

DATA_ROOT = Path(os.environ.get("SKETCH_DATA_ROOT", PROJECT_ROOT / "data"))
QUICKDRAW_DIR = Path(os.environ.get("QUICKDRAW_DATA_DIR", _default_qd))
IMAGENET_SKETCH_DIR = Path(os.environ.get("IMAGENET_SKETCH_DIR", _default_ims))
TUBERLIN_DIR = Path(os.environ.get("TUBERLIN_DATA_DIR", DATA_ROOT / "tuberlin_data"))

OUTPUT_DIR = Path(os.environ.get("SKETCH_OUTPUT_DIR", _default_out))
PROCESSED_DIR = OUTPUT_DIR / "processed_data"
MODELS_DIR = OUTPUT_DIR / "models"
RUN_RESULTS_DIR = OUTPUT_DIR / "results"

# Committed, recorded results of the full Kaggle run (see tools/extract_kaggle_results.py)
RECORDED_RESULTS_DIR = PROJECT_ROOT / "results"

# --------------------------------------------------------------------------
# Data constants
# --------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 64
SAMPLES_PER_CATEGORY = 2000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 50 QuickDraw categories used throughout (label = index in this list).
CATEGORIES = [
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel",
    "ant", "apple", "arm", "asparagus", "axe",
    "banana", "bandage", "barn", "baseball", "baseball bat",
    "basket", "basketball", "bat", "bathtub", "beach",
    "bear", "beard", "bed", "bee", "belt",
    "bench", "bicycle", "binoculars", "bird", "birthday cake",
    "blackberry", "blueberry", "book", "boomerang", "bottlecap",
    "bowtie", "bracelet", "brain", "bread", "bridge",
    "broccoli", "broom", "bucket", "bulldozer", "bus",
    "bush", "butterfly", "cactus", "cake", "calculator",
]
NUM_CLASSES = len(CATEGORIES)

# Category names confirmed present in BOTH QuickDraw and the zara2099 TU-Berlin
# Kaggle dataset (used by experiments/cross_dataset_tuberlin.py).
TUBERLIN_SHARED_CATEGORIES = ["airplane", "alarm clock", "angel", "ant", "apple", "arm"]
TUBERLIN_SAMPLES_PER_CATEGORY = 60

# ResNet18 parameter counts, for reference in reports.
RESNET18_PARAMS_1000_CLASS = 11_689_512   # torchvision default head (ImageNet)
RESNET18_PARAMS_50_CLASS = 11_202_162     # the baseline actually trained here


def get_device() -> str:
    """Best available torch device: CUDA > Apple Silicon (MPS) > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
