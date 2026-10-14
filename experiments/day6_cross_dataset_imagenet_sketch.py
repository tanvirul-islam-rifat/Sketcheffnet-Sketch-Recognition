"""
Day 6 - Cross-dataset test: QuickDraw-trained SketchEffNet on ImageNet-Sketch.

Target data: the first 50 ImageNet-Sketch classes in WordNet-ID order
(``n01440764`` tench, ``n01443537`` goldfish, ... fish, birds, amphibians
and reptiles). About 50 images each, 2,548 in total; stratified 70/15/15 split.
These classes do **not** overlap with the 50 QuickDraw classes, so this is a
feature-transfer test and not a same-label test.

Three approaches, each scored on the same 383 test images:

1. "Zero-shot": the QuickDraw backbone with a freshly initialised 50-way head
   and no training. With no shared labels, this can only reach chance (2 %).
2. Fine-tuning: the same model trained for 10 epochs (Adam, lr 1e-4).
3. From scratch: a new SketchEffNet trained for 20 epochs (Adam, lr 1e-3).

Recorded: 2.09 % / 7.83 % / 8.62 %.

Unlike the original script, this one no longer runs ``from day5_custom_efficient_model
import SketchNet``. That import re-ran all of Day 5 (37 minutes) as a side
effect; the model now comes from the side-effect-free ``sketcheffnet.model``.

    python -m experiments.day6_cross_dataset_imagenet_sketch [--checkpoint PATH] [--smoke]
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from experiments._common import banner, loader_kwargs, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.data import ImageDataFrameDataset, get_transforms, image_folder_dataframe, set_seed
from sketcheffnet.model import SketchNet
from sketcheffnet.plotting import plt
from sketcheffnet.training import evaluate, fit


def find_class_folders(root: Path) -> tuple[Path, list[str]]:
    """Accept either the class folder root or a parent containing ``sketch/``."""
    for cand in (root, root / "sketch", root / "ImageNet-Sketch"):
        if cand.is_dir():
            classes = sorted(d.name for d in cand.iterdir() if d.is_dir() and d.name.startswith("n"))
            if classes:
                return cand, classes
    raise SystemExit(
        f"No ImageNet-Sketch class folders (n0xxxxxxx/) found under {root}.\n"
        "Run on Kaggle with the ImageNet-Sketch dataset attached, or set IMAGENET_SKETCH_DIR "
        "(see scripts/download_imagenet_sketch.py)."
    )


def main() -> None:
    args = parse_args(__doc__.splitlines()[1], extra=lambda p: p.add_argument(
        "--checkpoint", type=Path, default=None, help="SketchEffNet checkpoint from Day 5"))
    paths = run_paths(args.smoke)
    out = paths.stage("day6")
    device = config.get_device()
    ckpt = args.checkpoint or (paths.models / "sketchnet_best.pth")
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt} - run Day 5 first or pass --checkpoint.")
    n_classes = 5 if args.smoke else 50
    batch_size = 16 if args.smoke else 128
    ep_ft, ep_scratch = (1, 1) if args.smoke else (10, 20)
    kw = loader_kwargs(device, args.workers, args.smoke)

    set_seed(config.SEED)
    root, all_classes = find_class_folders(config.IMAGENET_SKETCH_DIR)
    classes = all_classes[:n_classes]
    df = image_folder_dataframe(root, classes)
    print(f"{len(all_classes)} classes found; using {len(classes)} ({classes[0]} ... {classes[-1]}), {len(df)} images")

    train_df, rest = train_test_split(df, test_size=0.3, random_state=config.SEED, stratify=df["label"])
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=config.SEED, stratify=rest["label"])
    print(f"Split: train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    train_tf, eval_tf = get_transforms(config.IMG_SIZE, augment=True)
    train_loader = DataLoader(ImageDataFrameDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(ImageDataFrameDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(ImageDataFrameDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)

    banner("EXPERIMENT 1: 'Zero-shot' (QuickDraw backbone + untrained head)")
    source = SketchNet(num_classes=config.NUM_CLASSES).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    source.load_state_dict(state.get("model_state_dict", state))
    target = SketchNet(num_classes=len(classes)).to(device)
    merged = target.state_dict()
    merged.update({k: v for k, v in source.state_dict().items() if not k.startswith("fc.")})
    target.load_state_dict(merged)
    zs_acc, zs_f1 = evaluate(target, test_loader, device)
    print(f"  accuracy {zs_acc:.2f}% | weighted F1 {zs_f1:.2f}% | chance {100 / len(classes):.2f}%")

    banner("EXPERIMENT 2: Fine-tuning")
    ft = SketchNet(num_classes=len(classes)).to(device)
    ft.load_state_dict(merged)
    t0 = time.time()
    fit(ft, train_loader, val_loader, nn.CrossEntropyLoss(), optim.Adam(ft.parameters(), lr=0.0001),
        ep_ft, device, name="fine-tune")
    ft_time = (time.time() - t0) / 60
    ft_acc, ft_f1 = evaluate(ft, test_loader, device)

    banner("EXPERIMENT 3: From scratch")
    sc = SketchNet(num_classes=len(classes)).to(device)
    t0 = time.time()
    fit(sc, train_loader, val_loader, nn.CrossEntropyLoss(), optim.Adam(sc.parameters(), lr=0.001),
        ep_scratch, device, name="scratch")
    sc_time = (time.time() - t0) / 60
    sc_acc, sc_f1 = evaluate(sc, test_loader, device)

    table = pd.DataFrame({
        "approach": ["Zero-shot (untrained head)", "Fine-tuning", "From scratch"],
        "accuracy": [zs_acc, ft_acc, sc_acc], "f1_weighted": [zs_f1, ft_f1, sc_f1],
        "minutes": [0.0, ft_time, sc_time], "train_images": [0, len(train_df), len(train_df)],
    })
    print("\n" + table.round(2).to_string(index=False))
    print(f"Chance level: {100 / len(classes):.2f}%")
    table.to_csv(out / "cross_dataset_results.csv", index=False)
    save_json({"smoke": args.smoke, "classes": classes, "images": len(df),
               "chance_accuracy": 100 / len(classes), "results": table.to_dict(orient="records")},
              out / "cross_dataset_results.json")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(table["approach"], table["accuracy"], color=["#9aa5b1", "#f59e0b", "#2563eb"])
    ax.axhline(100 / len(classes), ls="--", color="k", lw=1, label="chance")
    ax.set_ylabel("Test accuracy (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "cross_dataset_comparison.png", dpi=150); plt.close(fig)
    print("\nDAY 6 DONE")


if __name__ == "__main__":
    main()
