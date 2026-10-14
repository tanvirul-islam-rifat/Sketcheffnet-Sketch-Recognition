"""
Regenerate every README figure from the recorded results in ``results/``.

    python -m tools.make_figures

Figures go to ``docs/images/``. No number is typed in here; everything is read
from the JSON that ``tools/extract_kaggle_results.py`` parsed out of the run log.
(The sample-sketch grid, the confusion matrix and the t-SNE plot are image
outputs of the run itself, cropped from the notebook.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.complexity import resnet18, sketcheffnet  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "images"

# Palette (validated reference palette, light mode) + neutral ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT_GRAY = "#8d8c87", "#c9c8c2"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": LIGHT_GRAY, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  docs/images/{name}")


def accuracy_vs_size(d1, d4, d5):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    deep = [m for m in d4["models"] if m["type"] == "Deep Learning"]
    for m in deep:
        ax.scatter(m["parameters"] / 1e6, m["test_acc"], s=70, color=GRAY, zorder=3, edgecolor="white", lw=1.5)
        dy = -0.55 if m["model"] in ("ResNet18",) else 0.3
        ax.annotate(m["model"], (m["parameters"] / 1e6, m["test_acc"]), xytext=(7, dy * 14),
                    textcoords="offset points", fontsize=9, color=INK_2)
    ax.scatter(d1["parameters"] / 1e6, d1["test"]["accuracy"], s=70, color=GRAY, marker="D", zorder=3,
               edgecolor="white", lw=1.5)
    ax.annotate("ResNet18 baseline\n(20 ep + LR schedule)", (d1["parameters"] / 1e6, d1["test"]["accuracy"]),
                xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK_2, va="top")
    ax.scatter(d5["parameters"] / 1e6, d5["test"]["accuracy"], s=170, color=BLUE, zorder=4, edgecolor="white", lw=2)
    ax.annotate(f"SketchEffNet  {d5['parameters'] / 1e6:.2f} M params, {d5['test']['accuracy']:.2f}%\n"
                "trained from scratch", (d5["parameters"] / 1e6, d5["test"]["accuracy"]),
                xytext=(-4, -40), textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 7, 11, 21], ["1M", "2M", "4M", "7M", "11M", "21M"])
    ax.set_xlim(0.8, 32)
    ax.set_ylim(74.5, 84.5)
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. model size - 50-class QuickDraw, 15,000 test sketches")
    save(fig, "accuracy_vs_size.png")


def model_comparison(d4, d5):
    rows = [(m["model"], m["test_acc"], m["type"]) for m in d4["models"]]
    rows.append(("SketchEffNet *", d5["test"]["accuracy"], "ours"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = [BLUE if t == "ours" else (GRAY if t == "Deep Learning" else LIGHT_GRAY) for _, _, t in rows]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=colors, height=0.66)
    for y, (_, v, _) in enumerate(rows):
        ax.text(v + 0.8, y, f"{v:.2f}%", va="center", fontsize=9, color=INK)
    ax.axvline(2, color=INK_2, lw=1, ls=":")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison (dark gray = CNNs, light gray = PCA-100 + classical)")
    fig.text(0.01, -0.02, "* SketchEffNet uses its own recipe (Day 5); the others share the Day 4 protocol. "
             "Dotted line: chance (2%).",
             fontsize=8.5, color=INK_2)
    save(fig, "model_comparison.png")


def training_dynamics(d1, d5):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    h5, h1 = d5["history"], d1["history"]
    a.plot(range(1, len(h5["val_acc"]) + 1), h5["val_acc"], color=BLUE, marker="o", ms=4, label="SketchEffNet (val)")
    a.plot(range(1, len(h1["val_acc"]) + 1), h1["val_acc"], color=ORANGE, marker="s", ms=4, label="ResNet18 baseline (val)")
    be = d5["training"]["best_epoch"]
    a.annotate(f"best {max(h5['val_acc']):.2f}% (epoch {be})\nearly stop at {d5['training']['epochs_run']}",
               (be, max(h5["val_acc"])), xytext=(22, -82), textcoords="offset points", fontsize=8.5,
               color=INK_2, arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    for x in (5, 15):
        a.axvline(x + 0.5, color=LIGHT_GRAY, lw=1, ls="--")
    a.text(5.7, 50, "LR restart", fontsize=8, color=INK_2)
    a.text(15.7, 50, "LR restart", fontsize=8, color=INK_2)
    a.set_xlabel("Epoch"); a.set_ylabel("Validation accuracy (%)"); a.legend(loc="lower center")
    a.set_title("Validation accuracy per epoch")
    b.plot(range(1, len(h5["lr"]) + 1), [x * 1000 for x in h5["lr"]], color=BLUE, marker="o", ms=3.5)
    b.set_xlabel("Epoch"); b.set_ylabel("Learning rate (x 1e-3)")
    b.set_title("SketchEffNet LR (cosine, warm restarts)")
    save(fig, "training_dynamics.png")


def fewshot(d1, d2):
    fs = d2["few_shot"]
    full_n = d1["dataset"]["train"] // 50
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(fs["samples_per_class"], fs["test_acc"], color=BLUE, marker="o", ms=7, label="ResNet18, 15 epochs")
    for x, y in zip(fs["samples_per_class"], fs["test_acc"]):
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.scatter([full_n], [d1["test"]["accuracy"]], color=ORANGE, marker="D", s=60, zorder=4,
               label=f"full training set ({full_n:,}/class, 20 epochs + LR schedule)")
    ax.annotate(f"{d1['test']['accuracy']:.1f}%", (full_n, d1["test"]["accuracy"]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([10, 50, 100, 500, 1000, full_n], ["10", "50", "100", "500", "1k", f"{full_n:,}"])
    ax.set_ylim(45, 90)
    ax.set_xlabel("Training sketches per class (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_title("Few-shot: accuracy grows roughly linearly in log(samples)")
    save(fig, "fewshot.png")


def augmentation(d2):
    ag = d2["augmentation"]
    names = ["None", "Standard\n(rotate, shift, flip)", "Sketch-specific\n(+ scale, elastic, blur)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(names, ag["test_acc"], color=[LIGHT_GRAY, BLUE, AQUA], width=0.6)
    base = ag["test_acc"][0]
    for x, v in enumerate(ag["test_acc"]):
        label = f"{v:.2f}%" + (f"\n({v - base:+.2f})" if x else "")
        ax.text(x, v + 0.25, label, ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(70, 81)
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Augmentation (ResNet18, 500 sketches/class)")
    save(fig, "augmentation.png")


def complexity():
    se, rn = sketcheffnet(), resnet18()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, unit, title in ((a, "params", 1e6, "Parameters (millions)"), (b, "macs", 1e6, "MACs per image (millions)")):
        vals = [rn[key] / unit, se[key] / unit]
        ax.bar(["ResNet18", "SketchEffNet"], vals, color=[GRAY, BLUE], width=0.55)
        for x, v in enumerate(vals):
            ax.text(x, v * 1.02, f"{v:,.1f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_title(f"{title}: {rn[key] / se[key]:.1f}x smaller")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, "64x64 input, 50 classes. Analytic counts (tools/complexity.py) that reproduce the "
             "logged parameter totals exactly.", fontsize=8.5, color=INK_2)
    save(fig, "complexity.png")


def per_class(d1):
    pc = sorted(d1["per_class"].items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(11, 4.0))
    vals = [v["f1"] * 100 for _, v in pc]
    colors = [ORANGE if i < 5 else (BLUE if i >= len(pc) - 5 else LIGHT_GRAY) for i in range(len(pc))]
    ax.bar(range(len(pc)), vals, color=colors, width=0.75)
    ax.set_xticks(range(len(pc)), [k for k, _ in pc], rotation=90, fontsize=8)
    ax.set_ylim(40, 100)
    ax.set_ylabel("F1 (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("ResNet18 baseline - per-class F1 on the test set (5 hardest in orange, 5 easiest in blue)")
    save(fig, "per_class_f1.png")


def cross_dataset(d6):
    a = d6["approaches"]
    names = ["Zero-shot\n(untrained head)", "Fine-tuned\n(10 epochs)", "From scratch\n(20 epochs)"]
    vals = [a["Zero-Shot"]["accuracy"], a["Fine-Tuning"]["accuracy"], a["From Scratch"]["accuracy"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(names, vals, color=[LIGHT_GRAY, GRAY, BLUE], width=0.58)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(d6["chance_accuracy"], color=INK_2, lw=1, ls=":")
    ax.text(0.34, d6["chance_accuracy"] + 0.2, "chance 2%", fontsize=8.5, color=INK_2, ha="left")
    ax.set_ylim(0, 12)
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"ImageNet-Sketch, 50 classes, {d6['split']['train']:,} training images")
    save(fig, "cross_dataset.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d1, d2, d4, d5, d6 = (load(n) for n in ("day1_baseline", "day2_experiments", "day4_model_comparison",
                                            "day5_sketcheffnet", "day6_cross_dataset"))
    accuracy_vs_size(d1, d4, d5)
    model_comparison(d4, d5)
    training_dynamics(d1, d5)
    fewshot(d1, d2)
    augmentation(d2)
    complexity()
    per_class(d1)
    cross_dataset(d6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
