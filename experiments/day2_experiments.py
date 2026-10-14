"""
Day 2 - Three controlled experiments with ResNet18 (ImageNet-pretrained,
Adam lr 1e-3, 15 epochs, no scheduler, same validation and test sets as Day 1).

1. Few-shot learning: 10 / 50 / 100 / 500 / 1,000 training sketches per class
   (standard augmentation).
2. Augmentation: none vs. standard (rotation, translation, flip) vs.
   sketch-specific (standard + scale + elastic + blur), with 500 sketches per class.
3. Transfer to unseen classes: train on 40 classes, extract 512-d penultimate
   features for the 10 held-out classes, then fit a k-NN (k=5) on 70 % of those
   features and score the other 30 %; t-SNE plot.

Recorded results: 49.65 -> 80.25 % (few-shot), 74.10 / 78.62 / 78.15 %
(augmentation), 83.08 % seen-class validation vs. 82.89 % unseen-class k-NN.

    python -m experiments.day2_experiments [--smoke]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from experiments._common import banner, loader_kwargs, parse_args, run_paths, save_json
from sketcheffnet import config
from sketcheffnet.baselines import create_model
from sketcheffnet.data import (QuickDrawDataset, augmentation_strategies, get_transforms, load_splits,
                               set_seed)
from sketcheffnet.plotting import plt
from sketcheffnet.training import evaluate, fit


def subsample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed)
             for _, g in ((c, df[df["category"] == c]) for c in sorted(df["category"].unique()))]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    paths = run_paths(args.smoke)
    out = paths.stage("day2")
    device = config.get_device()
    epochs = 1 if args.smoke else 15
    batch_size = 16 if args.smoke else 128
    pretrained = not args.smoke
    few_shot_sizes = [2, 5] if args.smoke else [10, 50, 100, 500, 1000]
    aug_subset = 5 if args.smoke else 500
    kw = loader_kwargs(device, args.workers, args.smoke)

    set_seed(config.SEED)
    train_df, val_df, test_df = load_splits(paths.processed)
    categories = sorted(train_df["category"].unique())
    print(f"Device {device} | train {len(train_df)} val {len(val_df)} test {len(test_df)} | {len(categories)} classes")

    _, eval_tf = get_transforms(config.IMG_SIZE, augment=False)
    val_loader = DataLoader(QuickDrawDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(QuickDrawDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False, **kw)

    def train_and_test(train_subset, transform, name, num_classes=config.NUM_CLASSES, vloader=val_loader):
        loader = DataLoader(QuickDrawDataset(train_subset, transform), batch_size=batch_size, shuffle=True, **kw)
        model = create_model("resnet18", num_classes, pretrained=pretrained).to(device)
        run = fit(model, loader, vloader, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001),
                  epochs, device, log_every=5, name=name)
        return model, run

    # ---------------------------------------------------------------- Experiment 1
    banner("EXPERIMENT 1: Few-shot learning")
    few = {"samples": [], "train_acc": [], "val_acc": [], "test_acc": [], "test_f1": []}
    std_tf, _ = get_transforms(config.IMG_SIZE, augment=True)
    for n in few_shot_sizes:
        subset = subsample(train_df, n, config.SEED)
        model, run = train_and_test(subset, std_tf, f"fewshot-{n}")
        acc, f1 = evaluate(model, test_loader, device)
        for k, v in zip(few, (n, run["history"]["train_acc"][-1], run["best_val_acc"], acc, f1)):
            few[k].append(v)
        print(f"  {n:>5}/class ({len(subset)} images): val {run['best_val_acc']:.2f}%  test {acc:.2f}%  F1 {f1:.2f}%")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(few["samples"], few["test_acc"], "o-")
    a2.plot(few["samples"], few["test_f1"], "s-", color="tab:orange")
    for a, lab in ((a1, "Test accuracy (%)"), (a2, "Test F1 (%)")):
        a.set_xscale("log"); a.set_xlabel("Training sketches per class"); a.set_ylabel(lab); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fewshot_results.png", dpi=150); plt.close(fig)

    # ---------------------------------------------------------------- Experiment 2
    banner("EXPERIMENT 2: Augmentation strategies")
    aug = {"strategy": [], "val_acc": [], "test_acc": [], "test_f1": []}
    subset = subsample(train_df, aug_subset, config.SEED)
    for name, tf in augmentation_strategies(config.IMG_SIZE).items():
        model, run = train_and_test(subset, tf, name)
        acc, f1 = evaluate(model, test_loader, device)
        for k, v in zip(aug, (name, run["best_val_acc"], acc, f1)):
            aug[k].append(v)
        print(f"  {name:<30} val {run['best_val_acc']:.2f}%  test {acc:.2f}%  F1 {f1:.2f}%")

    # ---------------------------------------------------------------- Experiment 3
    banner("EXPERIMENT 3: Transfer to unseen classes")
    np.random.seed(config.SEED)
    shuffled = np.array(categories)
    np.random.shuffle(shuffled)
    seen, unseen = shuffled[:40].tolist(), shuffled[40:].tolist()
    print(f"Unseen classes: {unseen}")
    remap = {c: i for i, c in enumerate(seen)}
    t_train = train_df[train_df["category"].isin(seen)].copy()
    t_val = val_df[val_df["category"].isin(seen)].copy()
    t_test = test_df[test_df["category"].isin(unseen)].copy()
    t_train["label"] = t_train["category"].map(remap)
    t_val["label"] = t_val["category"].map(remap)
    t_val_loader = DataLoader(QuickDrawDataset(t_val, eval_tf), batch_size=batch_size, shuffle=False, **kw)
    model, run = train_and_test(t_train, std_tf, "transfer-40", num_classes=40, vloader=t_val_loader)
    print(f"  best validation accuracy on the 40 seen classes: {run['best_val_acc']:.2f}%")

    extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in DataLoader(QuickDrawDataset(t_test, eval_tf), batch_size=batch_size, shuffle=False, **kw):
            feats.append(extractor(x.to(device)).flatten(1).cpu().numpy())
            labels.append(y.numpy())
    feats, labels = np.vstack(feats), np.concatenate(labels)
    X_tr, X_te, y_tr, y_te = train_test_split(feats, labels, test_size=0.3, random_state=config.SEED, stratify=labels)
    knn_acc = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr).score(X_te, y_te) * 100
    print(f"  k-NN (k=5) accuracy on the 10 unseen classes: {knn_acc:.2f}%  (chance = 10%)")

    n = min(1000, len(feats))
    idx = np.random.choice(len(feats), n, replace=False)
    emb = TSNE(n_components=2, random_state=config.SEED, perplexity=min(30, n - 1)).fit_transform(feats[idx])
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels[idx], cmap="tab10", s=12, alpha=0.7)
    fig.colorbar(sc, label="QuickDraw label")
    ax.set_title("t-SNE of transferred features (10 unseen classes)")
    fig.tight_layout(); fig.savefig(out / "transfer_tsne.png", dpi=150); plt.close(fig)

    save_json({
        "smoke": args.smoke,
        "experiment_1_fewshot": few,
        "experiment_2_augmentation": aug,
        "experiment_3_transfer": {"train_categories": seen, "test_categories": unseen,
                                  "best_val_acc_40_categories": run["best_val_acc"],
                                  "knn_acc_unseen_categories": knn_acc},
    }, out / "day2_all_results.json")
    print("\nDAY 2 DONE")


if __name__ == "__main__":
    main()
