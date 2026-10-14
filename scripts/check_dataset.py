"""
Check that the QuickDraw data is in place and in the expected format.

Prints, for each of the 50 configured categories, whether its .npy file was
found and how many sketches it holds, then checks one file's shape and dtype
(N x 784, uint8 0-255). Exits with status 1 if anything is missing.

    python scripts/check_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketcheffnet.config import CATEGORIES, QUICKDRAW_DIR  # noqa: E402
from sketcheffnet.data import find_npy_file  # noqa: E402


def main() -> int:
    print(f"QuickDraw directory: {QUICKDRAW_DIR}")
    if not QUICKDRAW_DIR.is_dir():
        print("  NOT FOUND - run `python scripts/download_quickdraw.py` or set QUICKDRAW_DATA_DIR")
        return 1
    missing, first = [], None
    for cat in CATEGORIES:
        f = find_npy_file(QUICKDRAW_DIR, cat)
        if f is None:
            missing.append(cat)
            print(f"  [MISSING] {cat}")
            continue
        n = np.load(f, mmap_mode="r").shape[0]
        first = first or f
        print(f"  [ok] {cat:<18} {n:>8,} sketches")
    if first is not None:
        arr = np.load(first, mmap_mode="r")
        ok = arr.ndim == 2 and arr.shape[1] == 784 and arr.dtype == np.uint8
        print(f"\nFormat check on {first.name}: shape {arr.shape}, dtype {arr.dtype} -> {'OK' if ok else 'UNEXPECTED'}")
        if not ok:
            return 1
    print(f"\n{len(CATEGORIES) - len(missing)}/{len(CATEGORIES)} categories present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
