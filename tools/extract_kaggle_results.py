"""
Extract the recorded results of the full Kaggle run into versioned files.

The experiments were executed once, end to end, in a Kaggle notebook
(Tesla T4). That notebook - with all of its printed output - is kept in
``notebooks/kaggle_full_run.ipynb``. This tool parses those printed logs
and writes:

    results/kaggle_logs/*.txt          cleaned per-stage logs (progress bars removed)
    results/day1_baseline.json         ResNet18 baseline: config, history, test metrics, per-class report
    results/day2_experiments.json      few-shot, augmentation and transfer-learning experiments
    results/day4_model_comparison.json 6 CNNs + 3 classical models (with per-epoch history)
    results/day4_model_comparison.csv  same table, flat
    results/day5_sketcheffnet.json     SketchEffNet: size, history (incl. LR), test metrics, re-run check
    results/day6_cross_dataset.json    QuickDraw -> ImageNet-Sketch transfer
    results/tf_keras_baseline.json     archived TensorFlow/Keras exploration (notebooks/archive)

Nothing is typed in by hand: every number in results/ (and therefore in the
README tables and figures) is parsed from the recorded run.

Usage:
    python -m tools.extract_kaggle_results
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_NOTEBOOK = ROOT / "notebooks" / "kaggle_full_run.ipynb"
TF_NOTEBOOK = ROOT / "notebooks" / "archive" / "tf_keras_baseline.ipynb"
RESULTS = ROOT / "results"
LOGS = RESULTS / "kaggle_logs"

# Stage name -> a snippet that identifies the notebook cell that ran it.
STAGES = {
    "00_check_dataset": "check_dataset.py",
    "01_day1_baseline": "day1_setup_baseline.py",
    "02_day2_experiments": "day2_experiments.py",
    "03_day3_analysis": "day3_analysis_and_paper.py",
    "04_day4_model_comparison": "day4_other_models_test.py",
    "05_resnet18_parameter_count": "RESNET18 PARAMETER ANALYSIS",
    "06_day5_sketcheffnet": "day5_custom_efficient_model.py",
    "07_day6_cross_dataset": "DAY 6: CROSS-DATASET VALIDATION",
}

PROGRESS = re.compile(r"\d+%\|")
FLOAT = r"(-?\d+(?:\.\d+)?)"


# --------------------------------------------------------------------------
# Notebook helpers
# --------------------------------------------------------------------------

def _cell_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append("".join(out["text"]))
    raw = "".join(parts)
    lines = re.split(r"[\r\n]", raw)
    keep = [l.rstrip() for l in lines if l.strip() and not PROGRESS.search(l)]
    return "\n".join(keep) + "\n"


def load_stage_logs(notebook: Path = RUN_NOTEBOOK) -> dict[str, str]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    logs: dict[str, str] = {}
    for name, marker in STAGES.items():
        for cell in nb["cells"]:
            src = "".join(cell.get("source", ""))
            if cell.get("cell_type") != "code":
                continue
            # "!python day1_setup_baseline.py" style cells, or an inline-source marker
            if (f"!python {marker}" in src) or (marker in src and not src.strip().startswith("# 1.")
                                                and "nbconvert" not in src):
                text = _cell_text(cell)
                if text.strip():
                    logs[name] = text
                    break
        if name not in logs:
            raise SystemExit(f"Could not locate the '{name}' stage in {notebook}")
    return logs


def _one(pattern: str, text: str, cast=float, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"pattern not found: {pattern!r}")
    return cast(m.group(1))


# --------------------------------------------------------------------------
# Stage parsers
# --------------------------------------------------------------------------

def parse_classification_report(text: str) -> dict:
    per_class = {}
    block = text.split("Classification Report:")[1]
    for line in block.splitlines():
        m = re.match(r"^\s*(.+?)\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d\.\d{4})\s+(\d+)\s*$", line)
        if m and m.group(1) not in ("macro avg", "weighted avg"):
            per_class[m.group(1).strip()] = {
                "precision": float(m.group(2)), "recall": float(m.group(3)),
                "f1": float(m.group(4)), "support": int(m.group(5)),
            }
    return per_class


def parse_day1(text: str) -> dict:
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for m in re.finditer(r"Train Loss: " + FLOAT + r", Train Acc: " + FLOAT + r"%\s*\n"
                         r"Val Loss: " + FLOAT + r", Val Acc: " + FLOAT + "%", text):
        hist["train_loss"].append(float(m.group(1)))
        hist["train_acc"].append(float(m.group(2)))
        hist["val_loss"].append(float(m.group(3)))
        hist["val_acc"].append(float(m.group(4)))
    per_class = parse_classification_report(text)
    return {
        "model": "ResNet18 (ImageNet-pretrained, fc -> 50 classes)",
        "device": "Tesla T4 (Kaggle)",
        "parameters": _one(r"Model created with (\d+) parameters", text, int),
        "dataset": {
            "categories": len(per_class),
            "total_images": _one(r"Total images: (\d+)", text, int),
            "train": _one(r"Train set: (\d+)", text, int),
            "val": _one(r"Validation set: (\d+)", text, int),
            "test": _one(r"Test set: (\d+)", text, int),
        },
        "training": {
            "epochs": len(hist["val_acc"]), "batch_size": 128, "optimizer": "Adam",
            "learning_rate": 0.001, "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=3)",
            "checkpoint": "best validation accuracy",
            "time_minutes": _one(r"Training completed in " + FLOAT + " minutes", text),
            "best_val_acc": _one(r"Best validation accuracy: " + FLOAT + "%", text),
        },
        "test": {
            "accuracy": _one(r"Test Accuracy: " + FLOAT + "%", text),
            "f1_weighted": _one(r"Test F1-Score: " + FLOAT + "%", text),
            "macro_f1": round(_one(r"macro avg\s+\S+\s+\S+\s+" + FLOAT, text) * 100, 2),
        },
        "history": hist,
        "per_class": per_class,
    }


def parse_day2(text: str) -> dict:
    few = {"samples_per_class": [], "train_images": [], "val_acc": [], "test_acc": [], "test_f1": []}
    blocks = re.split(r"--- Training with ", text)
    aug = {"strategy": [], "val_acc": [], "test_acc": [], "test_f1": []}
    for b in blocks[1:]:
        res = re.search(r"Results - Val Acc: " + FLOAT + "%, Test Acc: " + FLOAT + "%, Test F1: " + FLOAT + "%", b)
        if b.split()[0].isdigit():
            few["samples_per_class"].append(int(b.split()[0]))
            few["train_images"].append(_one(r"Few-shot training samples: (\d+)", b, int))
            few["val_acc"].append(float(res.group(1)))
            few["test_acc"].append(float(res.group(2)))
            few["test_f1"].append(float(res.group(3)))
        else:
            aug["strategy"].append(b.split()[0])
            aug["val_acc"].append(float(res.group(1)))
            aug["test_acc"].append(float(res.group(2)))
            aug["test_f1"].append(float(res.group(3)))
    test_cats = re.search(r"Test categories \(10\): \[(.*?)\]", text).group(1)
    return {
        "model": "ResNet18 (ImageNet-pretrained), Adam lr=0.001, 15 epochs, no scheduler",
        "few_shot": few,
        "augmentation": {**aug, "samples_per_class": 500},
        "transfer": {
            "train_categories": 40,
            "unseen_categories": [c.strip().strip("'") for c in test_cats.split(",")],
            "best_val_acc_seen_40": _one(r"Best validation accuracy on 40 categories: " + FLOAT + "%", text),
            "knn_acc_unseen_10": _one(r"k-NN accuracy on unseen categories.*?: " + FLOAT + "%", text),
            "feature_dim": _one(r"Extracted features shape: \(\d+, (\d+)\)", text, int),
            "unseen_test_images": _one(r"Transfer test: (\d+)", text, int),
            "knn": "k=5, 70/30 split of the unseen-category features",
        },
    }


def parse_day4(text: str) -> dict:
    models = []
    # deep-learning models: blocks start at "Testing: <name>"
    sections = re.split(r"\*{20,}\nTesting: ", text)
    for sec in sections[1:]:
        name = sec.splitlines()[0].strip()
        entry = {"model": name}
        if "Total parameters:" in sec:
            entry["type"] = "Deep Learning"
            entry["parameters"] = _one(r"Total parameters: ([\d,]+)", sec, lambda s: int(s.replace(",", "")))
            entry["pretrained"] = name != "Simple CNN"
            entry["history"] = {
                "train_acc": [float(x) for x in re.findall(r"Epoch \d+: Train Acc=" + FLOAT, sec)],
                "val_acc": [float(x) for x in re.findall(r"Val Acc=" + FLOAT, sec)],
            }
            entry["best_val_acc"] = _one(r"Best Val Acc: " + FLOAT, sec)
        else:
            entry["type"] = "Traditional ML"
            entry["features"] = "64x64 grayscale pixels -> PCA(100)"
            entry["pca_explained_variance"] = _one(r"Explained variance: " + FLOAT + "%", sec)
            entry["best_val_acc"] = _one(r"Val Acc: " + FLOAT, sec)
        entry["test_acc"] = _one(r"Test Acc: " + FLOAT, sec)
        entry["test_f1"] = _one(r"Test F1: " + FLOAT, sec)
        entry["train_minutes"] = _one(r"Training Time: " + FLOAT, sec)
        models.append(entry)
    return {
        "protocol": "15 epochs, Adam lr=0.001, no scheduler, test metrics from the LAST epoch "
                    "(not the best-validation checkpoint); classical models on PCA-100 pixel features",
        "models": models,
    }


def parse_resnet18_params(text: str) -> dict:
    return {
        "resnet18_50_class_total": _one(r"Total Parameters: ([\d,]+)", text, lambda s: int(s.replace(",", ""))),
        "resnet18_50_class_fp32_mb": _one(r"Model Size: " + FLOAT + " MB", text),
    }


def parse_day5(text: str, rerun_text: str | None = None) -> dict:
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    for m in re.finditer(r"Train Loss: " + FLOAT + r", Train Acc: " + FLOAT + r"%\s*\n"
                         r"Val Loss: " + FLOAT + r", Val Acc: " + FLOAT + r"%\s*\n"
                         r"Learning Rate: " + FLOAT, text):
        for k, g in zip(hist, range(1, 6)):
            hist[k].append(float(m.group(g)))
    out = {
        "model": "SketchEffNet (code name SketchNet) - trained from scratch, no ImageNet pretraining",
        "parameters": _one(r"Total Parameters: ([\d,]+)", text, lambda s: int(s.replace(",", ""))),
        "fp32_size_mb": _one(r"Model Size: " + FLOAT + " MB", text),
        "reference_resnet18_used_in_script": _one(r"ResNet18: ([\d,]+) parameters", text,
                                                  lambda s: int(s.replace(",", ""))),
        "printed_reduction_percent": _one(r"Reduction: " + FLOAT + "%", text),
        "printed_memory_saved_mb": _one(r"Memory Saved: " + FLOAT + " MB", text),
        "training": {
            "max_epochs": 25, "epochs_run": len(hist["val_acc"]), "batch_size": 128,
            "optimizer": "AdamW(lr=0.001, weight_decay=0.01)",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=5, T_mult=2)",
            "early_stopping_patience": 7,
            "best_epoch": _one(r"Best validation accuracy: [\d.]+% \(Epoch (\d+)\)", text, int),
            "best_val_acc": _one(r"Best validation accuracy: " + FLOAT + "%", text),
            "time_minutes": _one(r"Training completed in " + FLOAT + " minutes", text),
            "augmentation": "rotation 20, translate 0.15, scale 0.85-1.15, hflip 0.3, elastic(p=0.3)",
        },
        "test": {
            "accuracy": _one(r"Test Accuracy: " + FLOAT + "%", text),
            "f1_weighted": _one(r"Test F1-Score: " + FLOAT + "%", text),
        },
        "history": hist,
    }
    if rerun_text is not None:
        # Day 6 imported the Day 5 script, which re-ran the whole training as an import side effect.
        rerun = parse_day5(rerun_text)
        out["accidental_rerun"] = {
            "source": "Day 6 cell (`from day5_custom_efficient_model import SketchNet` re-executed Day 5)",
            "test_accuracy": rerun["test"]["accuracy"],
            "test_f1_weighted": rerun["test"]["f1_weighted"],
            "history_identical": rerun["history"] == hist,
            "time_minutes": rerun["training"]["time_minutes"],
        }
    return out


def parse_day6(text: str) -> dict:
    def epochs(label):
        sec = text.split(label + ":")[1].split("🎯")[0]
        return {
            "train_acc": [float(x) for x in re.findall(r"Train=" + FLOAT + "%", sec)],
            "val_acc": [float(x) for x in re.findall(r"Val=" + FLOAT + "%", sec)],
        }
    rows = {}
    for m in re.finditer(r"^\s*(Zero-Shot|Fine-Tuning|From Scratch)\s+" + FLOAT + r"\s+" + FLOAT + r"\s+"
                         + FLOAT + r"\s+(\d+)\s*$", text, re.M):
        rows[m.group(1)] = {"accuracy": round(float(m.group(2)), 2), "f1_weighted": round(float(m.group(3)), 2),
                            "time_minutes": round(float(m.group(4)), 2), "train_images": int(m.group(5))}
    return {
        "target_dataset": "ImageNet-Sketch, first 50 WordNet IDs (n01440764 ...) - no label overlap with QuickDraw",
        "images": _one(r"Loaded ([\d,]+) images", text, lambda s: int(s.replace(",", ""))),
        "split": {
            "train": _one(r"Train=([\d,]+)", text, lambda s: int(s.replace(",", ""))),
            "val": _one(r"Val=([\d,]+) \|", text, lambda s: int(s.replace(",", ""))),
            "test": _one(r"Test=([\d,]+)", text, lambda s: int(s.replace(",", ""))),
        },
        "chance_accuracy": 2.0,
        "approaches": rows,
        "zero_shot_note": "the QuickDraw classifier head is discarded and a freshly initialised 50-way head is "
                          "used, so this number measures a random head on top of QuickDraw features",
        "history": {"fine_tuning": epochs("Fine-Tuning"), "from_scratch": epochs("From Scratch")},
    }


def parse_tf_baseline(notebook: Path = TF_NOTEBOOK) -> dict:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    rows, robust = [], {}
    for cell in nb["cells"]:
        txt = "".join("".join(o.get("text", "")) for o in cell.get("outputs", []) if o.get("output_type") == "stream")
        for m in re.finditer(r"^\s*(\S+)\s+(0\.\d+)\s+(\d+\.\d+)\s*$", txt, re.M):
            rows.append({"model": m.group(1), "final_val_accuracy": round(float(m.group(2)) * 100, 2),
                         "parameters_millions": float(m.group(3))})
        for m in re.finditer(r"(Vanilla|Attention) MobileNet Accuracy on Broken Data:\s+(0\.\d+)", txt):
            robust[m.group(1).lower()] = round(float(m.group(2)) * 100, 2)
    return {
        "status": "archived exploration - superseded by the PyTorch pipeline",
        "protocol": "TensorFlow/Keras, 28x28 -> resized 128 (224 for Swin), 80/20 split used as validation, 7 epochs",
        "models": rows,
        "robustness_40pct_pixels_dropped": robust,
    }


# --------------------------------------------------------------------------

def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    logs = load_stage_logs()
    for name, text in logs.items():
        (LOGS / f"{name}.txt").write_text(text, encoding="utf-8")

    day6_text = logs["07_day6_cross_dataset"]
    rerun_part = day6_text.split("DAY 5: CUSTOM EFFICIENT MODEL")[1].split("DAY 6: CROSS-DATASET VALIDATION OF")[0]

    outputs = {
        "day1_baseline.json": parse_day1(logs["01_day1_baseline"]),
        "day2_experiments.json": parse_day2(logs["02_day2_experiments"]),
        "day4_model_comparison.json": {**parse_day4(logs["04_day4_model_comparison"]),
                                       **parse_resnet18_params(logs["05_resnet18_parameter_count"])},
        "day5_sketcheffnet.json": parse_day5(logs["06_day5_sketcheffnet"], rerun_part),
        "day6_cross_dataset.json": parse_day6(day6_text.split("DAY 6: CROSS-DATASET VALIDATION OF")[1]),
        "tf_keras_baseline.json": parse_tf_baseline(),
    }
    for fname, data in outputs.items():
        (RESULTS / fname).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote results/{fname}")

    with open(RESULTS / "day4_model_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "type", "parameters", "best_val_acc", "test_acc", "test_f1", "train_minutes"])
        for m in outputs["day4_model_comparison.json"]["models"]:
            w.writerow([m["model"], m["type"], m.get("parameters", ""), m["best_val_acc"],
                        m["test_acc"], m["test_f1"], m["train_minutes"]])
    print("wrote results/day4_model_comparison.csv")
    print(f"wrote {len(logs)} cleaned logs to results/kaggle_logs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
