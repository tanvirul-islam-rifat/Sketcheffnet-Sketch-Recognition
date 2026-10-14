"""
Classify sketch images with a trained SketchEffNet checkpoint.

    python -m sketcheffnet.predict my_drawing.png [more.png ...] --checkpoint models/sketchnet_best.pth

Any drawing works: dark strokes on a light background are inverted to match
QuickDraw's white-on-black. The drawing is then cropped to its strokes, padded
to a square and downsampled to 28x28, so it looks like a QuickDraw bitmap
before the usual 64x64 upsampling and normalisation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import CATEGORIES, IMG_SIZE, PROJECT_ROOT


def to_quickdraw_style(img: Image.Image) -> Image.Image:
    """Arbitrary drawing -> 64x64 grayscale, white strokes on black, QuickDraw-like framing."""
    g = img.convert("L")
    if np.asarray(g).mean() > 127:          # light background -> invert
        g = ImageOps.invert(g)
    arr = np.asarray(g)
    ys, xs = np.nonzero(arr > 32)
    if len(xs):                              # crop to the strokes, keep a small margin
        pad = int(0.08 * max(np.ptp(xs), np.ptp(ys), 1))
        g = g.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                    min(xs.max() + pad + 1, g.width), min(ys.max() + pad + 1, g.height)))
    side = max(g.size)
    square = Image.new("L", (side, side), 0)
    square.paste(g, ((side - g.width) // 2, (side - g.height) // 2))
    small = square.resize((28, 28), Image.Resampling.BOX)
    return small.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)


def load_model(checkpoint: Path, device: str = "cpu"):
    from .model import SketchNet
    from .training import load_checkpoint

    model = SketchNet(num_classes=len(CATEGORIES))
    return load_checkpoint(model, checkpoint, device).to(device).eval()


def classify(model, images, device: str = "cpu", top_k: int = 5):
    import torch

    from .data import eval_transform

    tf = eval_transform(IMG_SIZE)
    batch = torch.stack([tf(to_quickdraw_style(im).convert("RGB")) for im in images]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).cpu()
    top = probs.topk(top_k, dim=1)
    return [[(CATEGORIES[i], float(p)) for p, i in zip(ps, idx)] for ps, idx in zip(top.values, top.indices)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify sketches with SketchEffNet")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "sketchnet_best.pth")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train with `python -m experiments.day5_sketcheffnet` "
                         "or copy sketchnet_best.pth from your Kaggle output into models/.")
    model = load_model(args.checkpoint)
    for path, preds in zip(args.images, classify(model, [Image.open(p) for p in args.images], top_k=args.top)):
        print(f"\n{path}")
        for name, p in preds:
            print(f"  {p * 100:6.2f}%  {name}")


if __name__ == "__main__":
    main()
