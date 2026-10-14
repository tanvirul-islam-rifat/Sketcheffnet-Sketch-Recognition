"""
Day 5 - Train SketchEffNet (1.0 M parameters) from scratch.

Protocol (unchanged from the recorded run): AdamW (lr 1e-3, weight decay 0.01),
CosineAnnealingWarmRestarts (T_0=5, T_mult=2), batch 128, up to 25 epochs,
early stopping after 7 epochs without a validation gain, best-validation
checkpoint. Augmentation: rotation 20, translation 0.15, scale 0.85-1.15,
horizontal flip 0.3, elastic deformation (p=0.3).

Recorded result: 81.45 % test accuracy, 81.38 % weighted F1, best epoch 15 of 22,
36.49 min on a Tesla T4.

The parameter comparison uses the 50-class ResNet18 that was actually trained
(11,202,162 parameters). The original script compared against the 1,000-class
ImageNet ResNet18 (11,689,512), which overstated the reduction slightly
(91.4 % vs 91.1 %); see the README findings.

    python -m experiments.day5_sketcheffnet [--smoke]
"""
from __future__ import annotations

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from experiments._common import banner, loader_kwargs, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.data import (QuickDrawDataset, eval_transform, load_splits, set_seed,
                               sketcheffnet_train_transform)
from sketcheffnet.model import SketchNet, count_parameters, fp32_size_mb
from sketcheffnet.plotting import plot_training_curves
from sketcheffnet.training import fit, load_checkpoint, predict, score


def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    paths = run_paths(args.smoke)
    out = paths.stage("day5")
    device = config.get_device()
    epochs = 1 if args.smoke else 25
    batch_size = 16 if args.smoke else 128
    kw = loader_kwargs(device, args.workers, args.smoke)

    set_seed(config.SEED)
    train_df, val_df, test_df = load_splits(paths.processed)
    print(f"Device {device} | train {len(train_df)} val {len(val_df)} test {len(test_df)}")

    eval_tf = eval_transform(config.IMG_SIZE)
    train_loader = DataLoader(QuickDrawDataset(train_df, sketcheffnet_train_transform(config.IMG_SIZE)),
                              batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(QuickDrawDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(QuickDrawDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)

    banner("SketchEffNet")
    model = SketchNet(num_classes=config.NUM_CLASSES).to(device)
    params = count_parameters(model)
    ref = config.RESNET18_PARAMS_50_CLASS
    reduction = (ref - params) / ref * 100
    print(f"Parameters: {params:,} ({fp32_size_mb(model):.2f} MB fp32)")
    print(f"vs ResNet18 (50-class, {ref:,}): {reduction:.1f}% fewer parameters, "
          f"{(ref - params) * 4 / 1024 ** 2:.2f} MB smaller")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    ckpt = paths.models / "sketchnet_best.pth"
    run = fit(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, epochs, device,
              scheduler=scheduler, checkpoint_path=ckpt, patience=7, name="SketchEffNet")
    print(f"Training: {run['time_minutes']:.2f} min | best val {run['best_val_acc']:.2f}% (epoch {run['best_epoch']})")
    plot_training_curves(run["history"], out / "sketcheffnet_training.png", "SketchEffNet")

    banner("Test-set evaluation (best checkpoint)")
    load_checkpoint(model, ckpt, device)
    preds, labels = predict(model, test_loader, device)
    test_acc, test_f1 = score(labels, preds)
    print(f"Test accuracy {test_acc:.2f}% | weighted F1 {test_f1:.2f}%")

    save_json({
        "model": "SketchEffNet", "smoke": args.smoke,
        "architecture": {"total_parameters": params, "model_size_mb": fp32_size_mb(model),
                         "reference_resnet18_parameters": ref, "parameter_reduction_percent": reduction},
        "training": {"epochs_trained": run["epochs_run"], "best_epoch": run["best_epoch"],
                     "training_time_minutes": run["time_minutes"]},
        "performance": {"test_accuracy": test_acc, "test_f1_score": test_f1,
                        "best_val_accuracy": run["best_val_acc"]},
        "history": run["history"],
    }, out / "sketcheffnet_results.json")
    print(f"\nDAY 5 DONE - checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
