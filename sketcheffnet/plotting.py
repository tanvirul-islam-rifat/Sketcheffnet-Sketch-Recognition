"""Plot helpers used by the experiment scripts (headless-safe)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_samples(df, categories, n_samples=5, save_path="sample_images.png", seed=42):
    cats = categories[:min(10, len(categories))]
    fig, axes = plt.subplots(len(cats), n_samples, figsize=(2.2 * n_samples, 2.2 * len(cats)), squeeze=False)
    for i, cat in enumerate(cats):
        sub = df[df["category"] == cat]
        sub = sub.sample(n=min(n_samples, len(sub)), random_state=seed)
        for j in range(n_samples):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(sub):
                ax.imshow(Image.open(sub.iloc[j]["image_path"]), cmap="gray")
            if j == 0:
                ax.set_title(cat, fontsize=9, loc="left")
    _save(fig, save_path)


def plot_class_distribution(df, save_path="class_distribution.png"):
    counts = df["category"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)), counts.index, rotation=90)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution")
    _save(fig, save_path)


def plot_confusion_matrix(y_true, y_pred, categories, save_path="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(categories)))
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(categories)), categories, rotation=90, fontsize=7)
    ax.set_yticks(range(len(categories)), categories, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, save_path)
    return cm


def plot_training_curves(history, save_path="training_curves.png", title=""):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ep = np.arange(1, len(history["train_acc"]) + 1)
    if history.get("train_loss"):
        a1.plot(ep, history["train_loss"], label="train")
        a1.plot(ep, history["val_loss"], label="validation")
        a1.set_ylabel("Loss")
        a1.legend()
    a2.plot(ep, history["train_acc"], label="train")
    a2.plot(ep, history["val_acc"], label="validation")
    a2.set_ylabel("Accuracy (%)")
    a2.legend()
    for a in (a1, a2):
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    _save(fig, save_path)
