"""
Day 4 - Compare six CNNs and three classical classifiers on the same split.

Deep models (15 epochs, Adam lr 1e-3, batch 128, no scheduler; ImageNet-pretrained
except the Simple CNN): Simple CNN, MobileNetV2, ResNet18, ResNet34,
DenseNet121, EfficientNet-B0. As in the recorded run, test metrics come from
the **last** epoch, not from the best-validation checkpoint.

Classical models use grayscale 64x64 pixels (4,096 features) reduced with
PCA(100): Random Forest (100 trees, depth 20), k-NN (k=5), linear SVM
(max_iter=1000).

Recorded: EfficientNet-B0 82.63 % (best), ResNet18 81.49 %, k-NN 44.50 %,
linear SVM 10.79 % (it hit the iteration cap and did not converge).

    python -m experiments.day4_model_comparison [--smoke] [--with-sketcheffnet]

``--with-sketcheffnet`` adds SketchEffNet under this exact protocol, so it
can be compared on equal terms (not part of the recorded run).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision import transforms

from experiments._common import banner, loader_kwargs, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.baselines import DISPLAY_NAMES, create_model
from sketcheffnet.data import QuickDrawDataset, eval_transform, load_splits, set_seed
from sketcheffnet.model import count_parameters
from sketcheffnet.plotting import plt
from sketcheffnet.training import evaluate, fit

DEEP_MODELS = ["simple_cnn", "mobilenet_v2", "resnet18", "resnet34", "densenet121", "efficientnet_b0"]


def extract_pixels(loader):
    """Grayscale (channel mean of the normalised tensor), flattened to 4,096 values."""
    X, y = [], []
    for images, labels in loader:
        X.append(images.mean(dim=1).flatten(1).numpy())
        y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)


def main() -> None:
    args = parse_args(__doc__.splitlines()[1], extra=lambda p: p.add_argument(
        "--with-sketcheffnet", action="store_true", help="also train SketchEffNet under this protocol"))
    paths = run_paths(args.smoke)
    out = paths.stage("day4")
    device = config.get_device()
    epochs = 1 if args.smoke else 15
    batch_size = 16 if args.smoke else 128
    pretrained = not args.smoke
    pca_components = 10 if args.smoke else 100
    kw = loader_kwargs(device, args.workers, args.smoke)

    set_seed(config.SEED)
    train_df, val_df, test_df = load_splits(paths.processed)
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    eval_tf = eval_transform(config.IMG_SIZE)
    train_loader = DataLoader(QuickDrawDataset(train_df, train_tf), batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(QuickDrawDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(QuickDrawDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)

    rows, histories = [], {}
    deep = DEEP_MODELS + (["sketcheffnet"] if args.with_sketcheffnet else [])

    banner("EXPERIMENT 1: Deep-learning models")
    for key in deep:
        name = DISPLAY_NAMES[key]
        model = create_model(key, config.NUM_CLASSES, pretrained=pretrained and key not in ("simple_cnn", "sketcheffnet"))
        params = count_parameters(model)
        print(f"\n{name}: {params:,} parameters")
        run = fit(model.to(device), train_loader, val_loader, nn.CrossEntropyLoss(),
                  optim.Adam(model.parameters(), lr=0.001), epochs, device, name=name)
        test_acc, test_f1 = evaluate(model, test_loader, device)  # last-epoch weights, as recorded
        print(f"  {name}: best val {run['best_val_acc']:.2f}% | test {test_acc:.2f}% | F1 {test_f1:.2f}% | "
              f"{run['time_minutes']:.2f} min")
        rows.append({"model": name, "type": "Deep Learning", "parameters": params,
                     "best_val_acc": run["best_val_acc"], "test_acc": test_acc, "test_f1": test_f1,
                     "train_minutes": run["time_minutes"]})
        histories[name] = run["history"]
        torch.save(model.state_dict(), paths.models / f"{key}_day4.pth")

    banner("EXPERIMENT 2: Classical models on PCA pixel features")
    X_tr, y_tr = extract_pixels(train_loader)
    X_va, y_va = extract_pixels(val_loader)
    X_te, y_te = extract_pixels(test_loader)
    print(f"Feature matrix: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")
    classical = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=config.SEED, n_jobs=-1),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM (Linear)": SVC(kernel="linear", random_state=config.SEED, max_iter=1000),
    }
    for name, clf in classical.items():
        start = time.time()
        pca = PCA(n_components=pca_components, random_state=config.SEED)
        clf.fit(pca.fit_transform(X_tr), y_tr)
        minutes = (time.time() - start) / 60
        val_acc = (clf.predict(pca.transform(X_va)) == y_va).mean() * 100
        pred = clf.predict(pca.transform(X_te))
        test_acc, test_f1 = (pred == y_te).mean() * 100, f1_score(y_te, pred, average="weighted") * 100
        print(f"  {name}: PCA variance {pca.explained_variance_ratio_.sum():.2%} | val {val_acc:.2f}% | "
              f"test {test_acc:.2f}% | {minutes:.2f} min")
        rows.append({"model": name, "type": "Traditional ML", "parameters": None, "best_val_acc": val_acc,
                     "test_acc": test_acc, "test_f1": test_f1, "train_minutes": minutes})

    df = pd.DataFrame(rows).sort_values("test_acc", ascending=False)
    df.to_csv(out / "model_comparison.csv", index=False)
    print("\n" + df.round(2).to_string(index=False))
    save_json({"smoke": args.smoke, "models": rows, "histories": histories}, out / "model_comparison.json")

    colors = ["#2563eb" if t == "Deep Learning" else "#dc2626" for t in df["type"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["model"], df["test_acc"], color=colors)
    for y, v in enumerate(df["test_acc"]):
        ax.text(v + 0.5, y, f"{v:.2f}%", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("Test accuracy (%)"); ax.set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=150); plt.close(fig)
    print("\nDAY 4 DONE")


if __name__ == "__main__":
    main()
