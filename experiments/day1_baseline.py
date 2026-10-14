"""
Day 1 - Dataset preparation, EDA and the ResNet18 baseline.

* Samples 2,000 sketches for each of the 50 categories (100,000 images), renders
  them to 64x64 PNGs and writes the stratified 70/15/15 split used by every
  later experiment (``processed_data/{train,val,test}.csv``).
* Fine-tunes an ImageNet-pretrained ResNet18: Adam (lr 1e-3), batch 128,
  20 epochs, ReduceLROnPlateau on validation accuracy, best-validation checkpoint.
* Evaluates on the held-out test set: accuracy, weighted F1, per-class report,
  confusion matrix.

Recorded result (Tesla T4): 83.49 % test accuracy, 83.33 % weighted F1, 14.37 min.

    python -m experiments.day1_baseline            # full run
    python -m experiments.day1_baseline --smoke    # pipeline check in seconds
"""
from __future__ import annotations

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from experiments._common import banner, loader_kwargs, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.baselines import create_model
from sketcheffnet.data import (QuickDrawDataset, get_transforms, prepare_dataset_from_npy, set_seed,
                               split_dataset)
from sketcheffnet.model import count_parameters
from sketcheffnet.plotting import (plot_class_distribution, plot_confusion_matrix, plot_samples,
                                   plot_training_curves)
from sketcheffnet.training import fit, load_checkpoint, predict, score


def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    paths = run_paths(args.smoke)
    out = paths.stage("day1")
    device = config.get_device()

    samples = 30 if args.smoke else config.SAMPLES_PER_CATEGORY
    epochs = 1 if args.smoke else 20
    batch_size = 16 if args.smoke else 128
    pretrained = not args.smoke

    set_seed(config.SEED)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    banner("STEP 1: Load and prepare the dataset")
    dataset_csv = paths.processed / "dataset.csv"
    if dataset_csv.exists():
        df = pd.read_csv(dataset_csv)
        print(f"Reusing {len(df)} processed images from {dataset_csv}")
    else:
        df = prepare_dataset_from_npy(config.QUICKDRAW_DIR, config.CATEGORIES, samples, paths.processed,
                                      config.IMG_SIZE, config.SEED)
    print(df["category"].value_counts().describe()[["count", "mean", "std"]].to_string())

    banner("STEP 2: Exploratory data analysis")
    plot_samples(df, config.CATEGORIES, 5, out / "sample_images.png")
    plot_class_distribution(df, out / "class_distribution.png")

    banner("STEP 3: Stratified 70/15/15 split")
    train_df, val_df, test_df = split_dataset(df, 0.7, 0.15, config.SEED)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(paths.processed / f"{name}.csv", index=False)
    print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    banner("STEP 4: DataLoaders")
    train_tf, eval_tf = get_transforms(config.IMG_SIZE, augment=True)
    kw = loader_kwargs(device, args.workers, args.smoke)
    train_loader = DataLoader(QuickDrawDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(QuickDrawDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(QuickDrawDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)

    banner("STEP 5: ResNet18 baseline")
    model = create_model("resnet18", config.NUM_CLASSES, pretrained=pretrained).to(device)
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    ckpt = paths.models / "baseline_best.pth"

    banner("STEP 6: Training")
    run = fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, epochs, device,
              scheduler=scheduler, scheduler_on_metric=True, checkpoint_path=ckpt, name="resnet18")
    print(f"Training: {run['time_minutes']:.2f} min | best val {run['best_val_acc']:.2f}% (epoch {run['best_epoch']})")
    plot_training_curves(run["history"], out / "training_curves.png", "ResNet18 baseline")

    banner("STEP 7: Test-set evaluation (best checkpoint)")
    load_checkpoint(model, ckpt, device)
    preds, labels = predict(model, test_loader, device)
    test_acc, test_f1 = score(labels, preds)
    present = sorted(set(labels) | set(preds))
    report = classification_report(labels, preds, labels=present, digits=4, zero_division=0,
                                   target_names=[config.CATEGORIES[i] for i in present])
    print(report)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_confusion_matrix(labels, preds, config.CATEGORIES, out / "confusion_matrix.png")

    save_json({
        "model": "ResNet18 (Baseline)", "parameters": params, "smoke": args.smoke,
        "dataset": {"total": len(df), "train": len(train_df), "val": len(val_df), "test": len(test_df),
                    "samples_per_category": samples, "num_categories": config.NUM_CLASSES},
        "training": {"epochs": epochs, "batch_size": batch_size, "learning_rate": 0.001,
                     "time_minutes": run["time_minutes"], "best_val_acc": run["best_val_acc"],
                     "best_epoch": run["best_epoch"]},
        "test_metrics": {"accuracy": test_acc, "f1_score": test_f1},
        "history": run["history"],
    }, out / "baseline_results.json")
    print(f"\nDAY 1 DONE - test accuracy {test_acc:.2f}%, weighted F1 {test_f1:.2f}%")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", str(config.SEED))
    main()
