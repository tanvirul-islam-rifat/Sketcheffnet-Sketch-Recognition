"""
Extension - same-label cross-dataset test: QuickDraw -> TU-Berlin.

Day 6 found almost no transfer to ImageNet-Sketch, but those classes did not
overlap with QuickDraw at all. This experiment removes that confound. It keeps
only categories that exist in **both** datasets
(``config.TUBERLIN_SHARED_CATEGORIES``), so the QuickDraw head can be scored on
TU-Berlin directly, with no new classifier:

1. Zero-shot: train SketchEffNet on QuickDraw, then test on TU-Berlin as is.
2. Fine-tuned: continue training on a small TU-Berlin split.
3. From scratch: train on that TU-Berlin split alone.

TU-Berlin is fetched file by file through the Kaggle CLI
(``scripts/download_tuberlin_subset.py``), so only a few MB are downloaded.
This was built after the recorded run and has no reported numbers yet.

    python -m experiments.cross_dataset_tuberlin --smoke   # 5 classes, 1 epoch
    python -m experiments.cross_dataset_tuberlin           # 6 classes, 800 + 60 images each
"""
from __future__ import annotations

import time

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from experiments._common import banner, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.data import (ImageDataFrameDataset, QuickDrawDataset, get_transforms, image_folder_dataframe,
                               prepare_dataset_from_npy, set_seed, split_dataset)
from sketcheffnet.model import SketchNet, count_parameters
from sketcheffnet.training import evaluate, fit


def main() -> None:
    args = parse_args(__doc__.splitlines()[1], extra=lambda p: p.add_argument(
        "--no-download", action="store_true", help="use TU-Berlin images already on disk"))
    paths = run_paths(args.smoke)
    out = paths.stage("tuberlin")
    device = config.get_device()
    if args.smoke:
        cats, qd_n, tb_n, bs, ep_src, ep_ft, ep_sc, lr_ft = config.TUBERLIN_SHARED_CATEGORIES[:5], 40, 8, 8, 1, 1, 1, 5e-4
    else:
        cats, qd_n, tb_n, bs, ep_src, ep_ft, ep_sc, lr_ft = config.TUBERLIN_SHARED_CATEGORIES, 800, 60, 32, 12, 8, 12, 1e-4
    set_seed(config.SEED)
    print(f"Device {device} | categories: {cats}")

    banner("STEP 1: QuickDraw subset")
    qd = prepare_dataset_from_npy(config.QUICKDRAW_DIR, cats, qd_n, paths.root / "processed_tuberlin_qd",
                                  config.IMG_SIZE, config.SEED)
    qd_train, qd_val, _ = split_dataset(qd, 0.7, 0.15, config.SEED)

    banner("STEP 2: TU-Berlin subset")
    if not args.no_download:
        from scripts.download_tuberlin_subset import ensure_tuberlin_categories
        ensure_tuberlin_categories(cats, tb_n)
    tb = image_folder_dataframe(config.TUBERLIN_DIR, cats)
    if tb.empty:
        raise SystemExit(f"No TU-Berlin images under {config.TUBERLIN_DIR}.")
    tb_train, tb_val, tb_test = split_dataset(tb, 0.6, 0.2, config.SEED)
    print(f"QuickDraw train/val {len(qd_train)}/{len(qd_val)} | TU-Berlin train/val/test "
          f"{len(tb_train)}/{len(tb_val)}/{len(tb_test)}")

    train_tf, eval_tf = get_transforms(config.IMG_SIZE, augment=True)

    def loader(df, tf, shuffle, cls=ImageDataFrameDataset):
        return DataLoader(cls(df, tf), batch_size=min(bs, max(len(df), 1)), shuffle=shuffle, num_workers=0)

    banner("STEP 3: Source model on QuickDraw")
    source = SketchNet(num_classes=len(cats)).to(device)
    print(f"SketchEffNet parameters: {count_parameters(source):,}")
    run = fit(source, loader(qd_train, train_tf, True, QuickDrawDataset), loader(qd_val, eval_tf, False),
              nn.CrossEntropyLoss(), optim.Adam(source.parameters(), lr=1e-3), ep_src, device, name="quickdraw")

    tb_test_loader = loader(tb_test, eval_tf, False)
    zs = evaluate(source, tb_test_loader, device)

    ft = SketchNet(num_classes=len(cats)).to(device)
    ft.load_state_dict(source.state_dict())
    t0 = time.time()
    fit(ft, loader(tb_train, train_tf, True), loader(tb_val, eval_tf, False), nn.CrossEntropyLoss(),
        optim.Adam(ft.parameters(), lr=lr_ft), ep_ft, device, name="fine-tune")
    ft_time = (time.time() - t0) / 60
    ftr = evaluate(ft, tb_test_loader, device)

    sc = SketchNet(num_classes=len(cats)).to(device)
    t0 = time.time()
    fit(sc, loader(tb_train, train_tf, True), loader(tb_val, eval_tf, False), nn.CrossEntropyLoss(),
        optim.Adam(sc.parameters(), lr=1e-3), ep_sc, device, name="scratch")
    sc_time = (time.time() - t0) / 60
    scr = evaluate(sc, tb_test_loader, device)

    table = pd.DataFrame({"approach": ["Zero-shot", "Fine-tuned", "From scratch"],
                          "accuracy": [zs[0], ftr[0], scr[0]], "f1_weighted": [zs[1], ftr[1], scr[1]],
                          "minutes": [0.0, ft_time, sc_time]})
    print("\n" + table.round(2).to_string(index=False) + f"\nChance level: {100 / len(cats):.2f}%")
    table.to_csv(out / "cross_dataset_results.csv", index=False)
    save_json({"smoke": args.smoke, "categories": cats, "quickdraw_source_val_acc": run["best_val_acc"],
               "results": table.to_dict(orient="records")}, out / "cross_dataset_results.json")


if __name__ == "__main__":
    main()
