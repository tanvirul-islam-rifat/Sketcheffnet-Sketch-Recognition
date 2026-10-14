"""
Verify the recorded results and every quantitative claim made about them.

Training takes about 2.5 GPU-hours, so CI cannot rerun the experiments. What it
can do is check, on every push, that:

1. ``results/`` is exactly what the run log in ``notebooks/kaggle_full_run.ipynb`` says;
2. the recorded run is internally consistent: split sizes, the per-class report
   against the headline accuracy, early stopping against its patience, the logged
   learning rates against the cosine-restart formula, and both runs of Day 5
   against each other;
3. the architecture accounts for the logged parameter counts (analytic model in ``tools/complexity.py``);
4. each number the paper (``docs/SketchEffNet_Paper.pdf``) reports, recomputed from the results.

Statuses: PASS = holds; WARN = arithmetic holds but the claim needs a caveat, or the
paper and the run disagree (a *finding*, documented in the README); FAIL = the results
contradict themselves or the log (exit status 1).

    python -m tools.verify_results            # summary
    python -m tools.verify_results -v         # with details
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Numbers as printed in docs/SketchEffNet_Paper.pdf (the claims under test).
PAPER = {
    "parameter_reduction_pct": 91.4,
    "memory_saved_mb": 40.77,
    "resnet18_memory_mb": 44.59,
    "resnet18_params_m": 11.20,
    "sketcheffnet_params": 1_001_586,
    "sketcheffnet_memory_mb": 3.82,
    "accuracy_drop": 2.04,
    "baseline_acc": 83.49, "baseline_f1": 83.33, "baseline_minutes": 14.37,
    "sketcheffnet_acc": 81.45, "sketcheffnet_f1": 81.38, "sketcheffnet_minutes": 36.49,
    "fewshot_test_acc": [49.65, 65.73, 69.57, 78.10, 80.25],
    "fewshot_500_share_of_full_pct": 93.5,
    "fewshot_full_row": {"samples_per_class": 2000, "train_images": 100_000},
    "aug_test_acc": [74.10, 78.62, 78.15],
    "aug_gain_attributed_to_sketch_specific_in_abstract": 4.52,
    "transfer_seen": 83.08, "transfer_unseen": 82.89, "transfer_retention_pct": 99.8,
    "day4_test_acc": {"Simple CNN": 75.70, "MobileNetV2": 81.07, "ResNet18": 81.49, "ResNet34": 81.78,
                      "DenseNet121": 82.08, "EfficientNet-B0": 82.63, "Random Forest": 43.93,
                      "k-NN (k=5)": 44.50, "SVM (Linear)": 10.79},
    "param_efficiency_ratio": 11,
    "zero_shot": 2.09, "fine_tuning": 7.83, "from_scratch": 8.62, "finetune_share_of_scratch_pct": 90.9,
    "imagenet_sketch_split": (1783, 382, 383),
}


@dataclass
class Check:
    status: str  # PASS | WARN | FAIL
    name: str
    detail: str


def load_results(root: Path = RESULTS) -> dict:
    names = ["day1_baseline", "day2_experiments", "day4_model_comparison", "day5_sketcheffnet",
             "day6_cross_dataset", "tf_keras_baseline"]
    return {n: json.loads((root / f"{n}.json").read_text(encoding="utf-8")) for n in names}


def _close(a, b, tol=0.011):
    return abs(a - b) <= tol


def cosine_restart_lr(step: int, base=1e-3, t0=5, mult=2) -> float:
    t_i, t_cur = t0, step
    while t_cur >= t_i:
        t_cur -= t_i
        t_i *= mult
    return base * (1 + math.cos(math.pi * t_cur / t_i)) / 2


def run_checks(r: dict) -> list[Check]:
    out: list[Check] = []

    def check(cond, name, detail, warn=False):
        out.append(Check("PASS" if cond else ("WARN" if warn else "FAIL"), name, detail))

    d1, d2, d4, d5, d6 = (r["day1_baseline"], r["day2_experiments"], r["day4_model_comparison"],
                          r["day5_sketcheffnet"], r["day6_cross_dataset"])
    ds = d1["dataset"]
    per = d1["per_class"]

    # ---------------------------------------------------------------- data
    check(ds["train"] + ds["val"] + ds["test"] == ds["total_images"] == 100_000
          and (ds["train"], ds["val"], ds["test"]) == (70_000, 15_000, 15_000),
          "Split is 70/15/15 of 100,000 sketches", f"{ds['train']}/{ds['val']}/{ds['test']}")
    supports = {v["support"] for v in per.values()}
    check(len(per) == 50 and supports == {300}, "Test set is balanced: 50 classes x 300 sketches",
          f"{len(per)} classes, supports {sorted(supports)}")
    check(d2["few_shot"]["train_images"] == [n * 50 for n in d2["few_shot"]["samples_per_class"]],
          "Few-shot training sets hold samples x 50 images", str(d2["few_shot"]["train_images"]))

    # ---------------------------------------------------------------- internal consistency
    mean_recall = sum(v["recall"] for v in per.values()) / 50 * 100
    check(_close(mean_recall, d1["test"]["accuracy"], 0.01),
          "Per-class recalls average to the reported test accuracy (balanced set)",
          f"mean recall {mean_recall:.2f}% vs accuracy {d1['test']['accuracy']:.2f}%")
    mean_f1 = sum(v["f1"] for v in per.values()) / 50 * 100
    check(_close(mean_f1, d1["test"]["f1_weighted"], 0.02), "Per-class F1 averages to the reported weighted F1",
          f"{mean_f1:.2f}% vs {d1['test']['f1_weighted']:.2f}%")
    h1 = d1["history"]
    check(len(h1["val_acc"]) == 20 and _close(max(h1["val_acc"]), d1["training"]["best_val_acc"]),
          "Day 1 checkpoint = best of the 20 logged validation epochs",
          f"max {max(h1['val_acc'])}% at epoch {h1['val_acc'].index(max(h1['val_acc'])) + 1}")
    h5 = d5["history"]
    best_ep = h5["val_acc"].index(max(h5["val_acc"])) + 1
    check(best_ep == d5["training"]["best_epoch"] and d5["training"]["epochs_run"] == best_ep + 7,
          "Day 5 early stopping fired exactly 7 epochs after the best epoch",
          f"best epoch {best_ep}, stopped after {d5['training']['epochs_run']}")
    lr_ok = all(_close(cosine_restart_lr(i + 1), lr, 1e-6) for i, lr in enumerate(h5["lr"]))
    check(lr_ok, "Logged learning rates follow CosineAnnealingWarmRestarts(T_0=5, T_mult=2)",
          "restarts after epochs 5 and 15 (cycles of 5, 10, 20)")
    rr = d5.get("accidental_rerun", {})
    check(rr.get("history_identical") and rr.get("test_accuracy") == d5["test"]["accuracy"],
          "Training is deterministic: the accidental second Day 5 run reproduced every epoch",
          f"test {rr.get('test_accuracy')}% both times, {d5['training']['time_minutes']} vs {rr.get('time_minutes')} min")
    day4_ok = all(_close(max(m["history"]["val_acc"]), m["best_val_acc"]) for m in d4["models"] if "history" in m)
    check(day4_ok, "Day 4 best validation accuracy = max of each model's 15 logged epochs", "6 CNNs")

    # ---------------------------------------------------------------- architecture
    sys.path.insert(0, str(ROOT))
    from tools.complexity import resnet18, sketcheffnet
    se, rn = sketcheffnet(), resnet18()
    check(se["params"] == d5["parameters"] == PAPER["sketcheffnet_params"],
          "Architecture accounts for SketchEffNet's 1,001,586 parameters", f"analytic {se['params']:,}")
    check(rn["params"] == d4["resnet18_50_class_total"] == d1["parameters"],
          "Architecture accounts for ResNet18's 11,202,162 parameters (50-class head)", f"analytic {rn['params']:,}")
    check(resnet18(1000)["params"] == d5["reference_resnet18_used_in_script"],
          "Day 5's reference count 11,689,512 is ResNet18 with the 1,000-class ImageNet head",
          "not the 50-class model that was trained")
    check(_close(d5["parameters"] * 4 / 1024 ** 2, PAPER["sketcheffnet_memory_mb"], 0.005),
          "SketchEffNet fp32 size = 3.82 MB", f"{d5['parameters'] * 4 / 1024 ** 2:.2f} MB")

    # ---------------------------------------------------------------- paper claims
    true_red = (rn["params"] - se["params"]) / rn["params"] * 100
    check(_close(true_red, PAPER["parameter_reduction_pct"], 0.05),
          "Paper: '91.4% parameter reduction'",
          f"vs the 50-class ResNet18 actually trained it is {true_red:.1f}% "
          f"(91.4% used the 1,000-class count)", warn=True)
    saved = (rn["params"] - se["params"]) * 4 / 1024 ** 2
    check(_close(saved, PAPER["memory_saved_mb"], 0.05), "Paper: '40.77 MB memory saved'",
          f"vs the trained ResNet18 it is {saved:.2f} MB", warn=True)
    rn_mb = rn["params"] * 4 / 1024 ** 2
    check(_close(rn_mb, PAPER["resnet18_memory_mb"], 0.05), "Paper Table I: ResNet18 '11.20M params, 44.59 MB'",
          f"11.20M params is {rn_mb:.2f} MB; 44.59 MB is the 1,000-class model", warn=True)
    drop = d1["test"]["accuracy"] - d5["test"]["accuracy"]
    check(_close(drop, PAPER["accuracy_drop"]), "Paper: SketchEffNet is 2.04 points below the ResNet18 baseline",
          f"{d1['test']['accuracy']} - {d5['test']['accuracy']} = {drop:.2f}")
    for k, v in (("baseline_acc", d1["test"]["accuracy"]), ("baseline_f1", d1["test"]["f1_weighted"]),
                 ("baseline_minutes", d1["training"]["time_minutes"]), ("sketcheffnet_acc", d5["test"]["accuracy"]),
                 ("sketcheffnet_f1", d5["test"]["f1_weighted"]),
                 ("sketcheffnet_minutes", d5["training"]["time_minutes"])):
        check(_close(v, PAPER[k]), f"Paper Table I: {k.replace('_', ' ')} = {PAPER[k]}", f"run log: {v}")
    check(all(_close(a, b) for a, b in zip(d2["few_shot"]["test_acc"], PAPER["fewshot_test_acc"])),
          "Paper Table II: five few-shot accuracies", str(d2["few_shot"]["test_acc"]))
    share = d2["few_shot"]["test_acc"][3] / d1["test"]["accuracy"] * 100
    check(_close(share, PAPER["fewshot_500_share_of_full_pct"], 0.05),
          "Paper: '500 samples per class reach 93.5% of full-data accuracy'", f"{share:.1f}%")
    full_per_class = ds["train"] // 50
    check(PAPER["fewshot_full_row"]["samples_per_class"] == full_per_class,
          "Paper Table II: full-data row '2,000 per class, 100,000 training images'",
          f"the baseline trained on {full_per_class:,} per class ({ds['train']:,} images); 2,000 includes "
          "the validation and test sketches", warn=True)
    check(all(_close(a, b) for a, b in zip(d2["augmentation"]["test_acc"], PAPER["aug_test_acc"])),
          "Paper Table III: three augmentation accuracies", str(d2["augmentation"]["test_acc"]))
    aug = dict(zip(d2["augmentation"]["strategy"], d2["augmentation"]["test_acc"]))
    sk_gain = aug["sketch_specific_augmentation"] - aug["no_augmentation"]
    st_gain = aug["standard_augmentation"] - aug["no_augmentation"]
    check(_close(sk_gain, PAPER["aug_gain_attributed_to_sketch_specific_in_abstract"]),
          "Paper abstract/contributions: 'sketch-specific augmentations provided a 4.52% improvement'",
          f"+4.52 is STANDARD augmentation ({st_gain:+.2f}); sketch-specific is {sk_gain:+.2f} "
          "(the paper's Table III and conclusion state this correctly)", warn=True)
    t = d2["transfer"]
    check(_close(t["best_val_acc_seen_40"], PAPER["transfer_seen"]) and _close(t["knn_acc_unseen_10"], PAPER["transfer_unseen"]),
          "Paper: transfer 83.08% (40 seen) / 82.89% (10 unseen)", "matches the log")
    check(False, "Paper: '99.8% performance retention' on unseen classes",
          f"{t['knn_acc_unseen_10'] / t['best_val_acc_seen_40'] * 100:.1f}% is a ratio of a 10-way k-NN score "
          "(chance 10%) to a 40-way softmax score (chance 2.5%); different tasks, not a retention measure", warn=True)
    d4map = {m["model"]: m for m in d4["models"]}
    check(all(_close(d4map[k]["test_acc"], v) for k, v in PAPER["day4_test_acc"].items()),
          "Paper Tables IV-V: nine model-comparison accuracies", "all match the log")
    check(False, "Paper: 'SketchEffNet nearly matches ResNet18 (81.45% vs 81.49%)'",
          "different protocols: ResNet18 81.49% = Day 4 (pretrained, Adam, 15 epochs, last epoch); SketchEffNet = "
          "Day 5 (from scratch, AdamW + cosine, 22 epochs, best checkpoint). Re-run with "
          "`day4_model_comparison --with-sketcheffnet` for a like-for-like number", warn=True)
    svm = d4map["SVM (Linear)"]["test_acc"]
    check(False, "Linear SVM (10.79%) is not a converged linear model",
          f"SVC(kernel='linear', max_iter=1000) on 70,000 samples is capped at 1,000 solver iterations "
          f"(the run suppressed warnings, so no ConvergenceWarning is logged); {svm}% is far below k-NN "
          "(44.50%) on the same features", warn=True)
    eff = (d5["test"]["accuracy"] / (d5["parameters"] / 1e6)) / (d1["test"]["accuracy"] / (d1["parameters"] / 1e6))
    check(round(eff) == PAPER["param_efficiency_ratio"], "Paper: '11x better parameter efficiency'",
          f"accuracy per million parameters: {eff:.1f}x")
    a = d6["approaches"]
    check(_close(a["Zero-Shot"]["accuracy"], PAPER["zero_shot"]) and _close(a["Fine-Tuning"]["accuracy"], PAPER["fine_tuning"])
          and _close(a["From Scratch"]["accuracy"], PAPER["from_scratch"]),
          "Paper Table VI: 2.09 / 7.83 / 8.62%", "matches the log")
    check(tuple(d6["split"].values()) == PAPER["imagenet_sketch_split"] and sum(d6["split"].values()) == d6["images"],
          "ImageNet-Sketch split 1,783 / 382 / 383 of 2,548 images", str(d6["split"]))
    check(False, "Paper: zero-shot 2.09% shows 'QuickDraw features don't transfer'",
          "the zero-shot model uses a randomly initialised head on classes QuickDraw never saw; 2.09% equals chance "
          "(2.00%) by construction and says nothing about the features", warn=True)
    share_ft = a["Fine-Tuning"]["accuracy"] / a["From Scratch"]["accuracy"] * 100
    check(_close(share_ft, PAPER["finetune_share_of_scratch_pct"], 0.15),
          "Paper: fine-tuning reaches 90.9% of from-scratch", f"{share_ft:.1f}%")
    return out


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    checks = run_checks(load_results())

    # results/ must be exactly what the run log says
    try:
        from tools.extract_kaggle_results import (RUN_NOTEBOOK, load_stage_logs, parse_day1, parse_day2,
                                                  parse_day6)
        logs = load_stage_logs(RUN_NOTEBOOK)
        r = load_results()
        same = (parse_day1(logs["01_day1_baseline"]) == r["day1_baseline"]
                and parse_day2(logs["02_day2_experiments"]) == r["day2_experiments"]
                and parse_day6(logs["07_day6_cross_dataset"].split("DAY 6: CROSS-DATASET VALIDATION OF")[1])
                == r["day6_cross_dataset"])
        checks.insert(0, Check("PASS" if same else "FAIL", "results/ matches the recorded Kaggle run log",
                               "re-parsed from notebooks/kaggle_full_run.ipynb"))
    except Exception as e:  # pragma: no cover
        checks.insert(0, Check("FAIL", "results/ matches the recorded Kaggle run log", repr(e)))

    width = max(len(c.name) for c in checks)
    for c in checks:
        line = f"[{c.status}] {c.name:<{width}}"
        print(line + (f"  {c.detail}" if verbose or c.status != "PASS" else ""))
    n = {s: sum(c.status == s for c in checks) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} finding(s), {n['FAIL']} failed")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
