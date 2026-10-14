"""
Data loading, preprocessing, augmentation and splitting.

QuickDraw ships each category as a ``.npy`` array of shape (N, 784): one
flattened 28x28 uint8 bitmap per row. The pipeline:

    sample 2,000 rows per category (fixed seed)
      -> reshape 28x28 -> Lanczos upsample to 64x64 -> save PNG
      -> stratified 70 / 15 / 15 split (seed 42)
      -> 3-channel RGB + ImageNet normalisation at load time

Unlike the original ``utils.py``, importing this module does not reseed the
global RNGs; :func:`set_seed` does that explicitly. Sampling uses
``np.random.RandomState(seed)``, which gives the same indices as the original
``np.random.seed(seed); np.random.choice(...)``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# QuickDraw .npy -> PNG dataset
# --------------------------------------------------------------------------

def find_npy_file(data_dir: Path | str, category: str) -> Path | None:
    """Locate a category's .npy file across the naming conventions in the wild."""
    data_dir = Path(data_dir)
    for name in (f"{category}.npy",
                 f"{category.replace(' ', '_')}.npy",
                 f"{category.replace(' ', '%20')}.npy",
                 f"full_numpy_bitmap_{category}.npy",
                 f"full_numpy_bitmap_{category.replace(' ', '_')}.npy"):
        if (data_dir / name).exists():
            return data_dir / name
    return None


def load_npy_file(filepath: Path | str, max_samples: int = 2000, seed: int = SEED) -> np.ndarray:
    """Load a QuickDraw .npy file, keeping a reproducible random subset of rows."""
    data = np.load(filepath)
    if len(data) > max_samples:
        idx = np.random.RandomState(seed).choice(len(data), max_samples, replace=False)
        data = data[idx]
    return data


def npy_to_image(row: np.ndarray, size: int = IMG_SIZE) -> Image.Image:
    """Flattened 28x28 bitmap (784,) -> grayscale PIL image resized to size x size."""
    img = Image.fromarray(np.asarray(row, dtype=np.uint8).reshape(28, 28))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def prepare_dataset_from_npy(data_dir, categories, samples_per_category=2000,
                             output_dir="processed_data", img_size=IMG_SIZE, seed=SEED) -> pd.DataFrame:
    """Render every sampled bitmap to PNG and return a DataFrame(image_path, category, label)."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"QuickDraw directory not found: {data_dir}\n"
            "Run `python scripts/download_quickdraw.py`, or set QUICKDRAW_DATA_DIR."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"\nProcessing {len(categories)} categories...")
    for label, category in enumerate(categories):
        npy_file = find_npy_file(data_dir, category)
        if npy_file is None:
            print(f"  WARNING: no .npy file for '{category}' - skipped")
            continue
        rows = load_npy_file(npy_file, max_samples=samples_per_category, seed=seed)
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(rows):
            path = cat_dir / f"{idx:04d}.png"
            npy_to_image(row, size=img_size).save(path)
            records.append({"image_path": str(path), "category": category, "label": label})
        print(f"  {category:<18} {len(rows)} samples")
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "dataset.csv", index=False)
    print(f"Dataset prepared: {len(df)} images across {df['category'].nunique() if len(df) else 0} categories")
    return df


def split_dataset(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, seed=SEED):
    """Stratified split: each category is shuffled and cut train/val/test independently."""
    train, val, test = [], [], []
    for category in df["category"].unique():
        cat = df[df["category"] == category].sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(cat)
        a, b = int(n * train_ratio), int(n * (train_ratio + val_ratio))
        train.append(cat[:a])
        val.append(cat[a:b])
        test.append(cat[b:])

    def _merge(parts):
        return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    return _merge(train), _merge(val), _merge(test)


def load_splits(processed_dir: Path | str):
    """Read the train/val/test CSVs written by Day 1, with a clear error if missing."""
    processed_dir = Path(processed_dir)
    paths = [processed_dir / f"{s}.csv" for s in ("train", "val", "test")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "Processed splits not found:\n  " + "\n  ".join(missing)
            + "\nRun `python -m experiments.day1_baseline` first (it builds the splits)."
        )
    return tuple(pd.read_csv(p) for p in paths)


# --------------------------------------------------------------------------
# Datasets and transforms
# --------------------------------------------------------------------------

try:  # keep the pure-pandas helpers above importable without PyTorch installed
    from torch.utils.data import Dataset as _Dataset
except ImportError:  # pragma: no cover
    _Dataset = object


class ImageDataFrameDataset(_Dataset):
    """Images listed in a DataFrame with ``image_path`` and ``label`` columns, loaded as RGB."""

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# Backwards-compatible name used by the original scripts.
QuickDrawDataset = ImageDataFrameDataset


def _normalise():
    from torchvision import transforms
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def eval_transform(img_size: int = IMG_SIZE):
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((img_size, img_size)), *_normalise()])


def get_transforms(img_size: int = IMG_SIZE, augment: bool = True):
    """(train, eval) transforms. ``augment=True`` gives the *standard* augmentation."""
    from torchvision import transforms
    if augment:
        train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(0.3),
            *_normalise(),
        ])
    else:
        train = eval_transform(img_size)
    return train, eval_transform(img_size)


def augmentation_strategies(img_size: int = IMG_SIZE) -> dict:
    """The three strategies compared in Day 2 (Experiment 2)."""
    from torchvision import transforms
    return {
        "no_augmentation": eval_transform(img_size),
        "standard_augmentation": get_transforms(img_size, augment=True)[0],
        "sketch_specific_augmentation": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            *_normalise(),
        ]),
    }


def sketcheffnet_train_transform(img_size: int = IMG_SIZE):
    """The 'enhanced' augmentation SketchEffNet was trained with (Day 5)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),
        *_normalise(),
    ])


def image_folder_dataframe(root: Path | str, classes: list[str]) -> pd.DataFrame:
    """DataFrame for a ``root/<class>/<image>`` folder tree, labelled by position in ``classes``."""
    root = Path(root)
    exts = {".png", ".jpg", ".jpeg"}
    rows = []
    for label, cls in enumerate(classes):
        folder = root / cls
        if not folder.is_dir():
            continue
        for f in sorted(os.listdir(folder)):
            if Path(f).suffix.lower() in exts:
                rows.append({"image_path": str(folder / f), "category": cls, "label": label})
    return pd.DataFrame(rows, columns=["image_path", "category", "label"])
