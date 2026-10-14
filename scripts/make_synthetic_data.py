"""
Generate a tiny synthetic stand-in for the datasets so the full pipeline can run
without Kaggle (used by CI and for a quick local smoke test).

* QuickDraw: 50 ``<category>.npy`` files, each (N, 784) uint8. Every class is a
  distinct procedural "doodle": a polygon with a class-dependent number of sides,
  rotation and stroke offset, plus jitter. A model can learn it, but it is not
  real data.
* ImageNet-Sketch: a handful of ``n0000000k/`` class folders of 64x64 JPEGs.

These files exist only to test the code path. No reported number comes from them.

    python scripts/make_synthetic_data.py --out /tmp/synthetic
    export QUICKDRAW_DATA_DIR=/tmp/synthetic/quickdraw_data
    export IMAGENET_SKETCH_DIR=/tmp/synthetic/imagenet_sketch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sketcheffnet.config import CATEGORIES  # noqa: E402


def doodle(label: int, rng: np.random.Generator, size: int = 28) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    sides = 3 + label % 7
    angle0 = (label // 7) * 0.45 + rng.normal(0, 0.08)
    r = 8 + rng.normal(0, 0.8)
    cx, cy = 14 + rng.normal(0, 1), 14 + rng.normal(0, 1)
    pts = [(cx + r * np.cos(angle0 + 2 * np.pi * k / sides), cy + r * np.sin(angle0 + 2 * np.pi * k / sides))
           for k in range(sides + 1)]
    d.line(pts, fill=255, width=2)
    if label % 2:
        d.line([(cx - 4, cy), (cx + 4, cy)], fill=255, width=2)
    return np.asarray(img, dtype=np.uint8).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--imagenet-classes", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    qd = args.out / "quickdraw_data"
    qd.mkdir(parents=True, exist_ok=True)
    for label, cat in enumerate(CATEGORIES):
        np.save(qd / f"{cat}.npy", np.stack([doodle(label, rng) for _ in range(args.per_class)]))
    print(f"wrote {len(CATEGORIES)} synthetic QuickDraw classes to {qd}")

    ims = args.out / "imagenet_sketch"
    for k in range(args.imagenet_classes):
        folder = ims / f"n{k:08d}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            arr = 255 - doodle(k * 9, rng).reshape(28, 28)
            Image.fromarray(arr).resize((64, 64)).convert("RGB").save(folder / f"sketch_{i}.JPEG")
    print(f"wrote {args.imagenet_classes} synthetic ImageNet-Sketch classes to {ims}")


if __name__ == "__main__":
    main()
