"""
Shared local configuration for the QuickDraw pipeline (Day 1-5).

Replaces the hardcoded /kaggle/input and /kaggle/working paths used when
this project ran as a Kaggle notebook. All day-scripts import DATA_DIR,
OUTPUT_DIR and get_device() from here instead of hardcoding Kaggle paths.

Override any of these via environment variables if you don't want to edit
this file directly, e.g.:
    QUICKDRAW_KAGGLE_DATASET="yourusername/quickdraw20" python day1_setup_baseline.py
"""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Kaggle dataset identifier ("owner/dataset-slug") used by download_dataset.py.
KAGGLE_DATASET = os.environ.get('QUICKDRAW_KAGGLE_DATASET', 'mdtanvirulislamrifat/quickdraw20')

# TU-Berlin hand-sketch dataset, used by master_pipeline_v2.py for
# cross-dataset validation (250 categories, ~20k images).
TUBERLIN_KAGGLE_DATASET = os.environ.get(
    'TUBERLIN_KAGGLE_DATASET', 'zara2099/tu-berlin-hand-sketch-image-dataset'
)

# Root folder datasets get downloaded/extracted into.
DATA_ROOT = os.environ.get('QUICKDRAW_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))

# Folder containing the QuickDraw .npy files directly
# (equivalent of the old /kaggle/input/quickdraw20/quickdraw_data).
DATA_DIR = os.environ.get('QUICKDRAW_DATA_DIR', os.path.join(DATA_ROOT, 'quickdraw_data'))

# Folder the TU-Berlin dataset gets staged into.
TUBERLIN_DIR = os.environ.get('TUBERLIN_DATA_DIR', os.path.join(DATA_ROOT, 'tuberlin_data'))

# Category names confirmed present in BOTH our local QuickDraw dataset and
# the zara2099 TU-Berlin dataset (discovered by paginating its file listing;
# it does not expose a full category index up front). Used for cross-dataset
# validation in master_pipeline_v2.py. Extend this list if you paginate
# further and find more overlap.
TUBERLIN_SHARED_CATEGORIES = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm']

# How many TU-Berlin images per category to keep staged locally. Kept small
# deliberately -- this machine doesn't have room for the full ~4GB/20k-image
# dataset, so we selectively download just this many files per category
# instead of the whole archive (see download_tuberlin_subset.py).
TUBERLIN_SAMPLES_PER_CATEGORY = 60

# Equivalent of the old /kaggle/working — where processed data, models,
# results and the generated paper get written.
OUTPUT_DIR = os.environ.get('QUICKDRAW_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'output'))


def get_device() -> str:
    """Pick the best available torch device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
