"""
Day 3 - Analysis of Day 1 + Day 2: comparison tables and summary figures.

Reads the JSON written by Day 1 and Day 2 (of *this* machine's run, under
``outputs/results``) and writes CSV tables, a Markdown summary and plots to
``outputs/results/day3``.

The original Day 3 also generated a template research paper from these numbers.
That generator is retired here: the finished paper is ``docs/SketchEffNet_Paper.pdf``,
and the template made some claims the numbers did not support (for example,
"sketch-specific augmentation is superior", when standard augmentation scored higher).

    python -m experiments.day3_analysis [--smoke]
"""
from __future__ import annotations

import json

import pandas as pd

from experiments._common import banner, parse_args, run_paths
from sketcheffnet.plotting import plt


def main() -> None:
    args = parse_args(__doc__.splitlines()[1])
    paths = run_paths(args.smoke)
    out = paths.stage("day3")
    try:
        d1 = json.loads((paths.results / "day1" / "baseline_results.json").read_text())
        d2 = json.loads((paths.results / "day2" / "day2_all_results.json").read_text())
    except FileNotFoundError as e:
        raise SystemExit(f"{e.filename} not found - run Day 1 and Day 2 first.") from None

    banner("Few-shot table")
    fs = d2["experiment_1_fewshot"]
    base = d1["test_metrics"]["accuracy"]
    few = pd.DataFrame({
        "samples_per_class": fs["samples"],
        "test_acc": fs["test_acc"],
        "test_f1": fs["test_f1"],
        "val_acc": fs["val_acc"],
        "share_of_full_data_baseline_%": [a / base * 100 for a in fs["test_acc"]],
    }).round(2)
    print(few.to_string(index=False))
    few.to_csv(out / "fewshot_results_table.csv", index=False)

    banner("Augmentation table")
    ag = d2["experiment_2_augmentation"]
    aug = pd.DataFrame({"strategy": ag["strategy"], "test_acc": ag["test_acc"], "test_f1": ag["test_f1"]})
    aug["gain_vs_none"] = aug["test_acc"] - aug["test_acc"].iloc[0]
    aug = aug.round(2)
    print(aug.to_string(index=False))
    aug.to_csv(out / "augmentation_results_table.csv", index=False)
    best = aug.loc[aug["test_acc"].idxmax(), "strategy"]

    tr = d2["experiment_3_transfer"]
    summary = {
        "baseline_test_accuracy": base,
        "baseline_test_f1": d1["test_metrics"]["f1_score"],
        "baseline_training_minutes": d1["training"]["time_minutes"],
        "fewshot_min_acc": min(fs["test_acc"]),
        "fewshot_max_acc": max(fs["test_acc"]),
        "best_augmentation": best,
        "best_augmentation_gain": float(aug["gain_vs_none"].max()),
        "transfer_seen_40_val": tr["best_val_acc_40_categories"],
        "transfer_unseen_10_knn": tr["knn_acc_unseen_categories"],
    }
    pd.Series(summary).to_csv(out / "overall_summary_table.csv", header=["value"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(fs["samples"], fs["test_acc"], "o-", label="ResNet18, k sketches/class")
    a1.axhline(base, ls="--", color="tab:green", label=f"full data ({base:.2f}%)")
    a1.set_xscale("log"); a1.set_xlabel("Training sketches per class"); a1.set_ylabel("Test accuracy (%)")
    a1.legend(); a1.grid(alpha=0.3)
    a2.bar(aug["strategy"].str.replace("_", " "), aug["test_acc"], color=["#9aa5b1", "#3b82f6", "#f59e0b"])
    a2.set_ylabel("Test accuracy (%)"); a2.set_ylim(min(aug["test_acc"]) - 5, max(aug["test_acc"]) + 2)
    a2.tick_params(axis="x", rotation=10)
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=150); plt.close(fig)

    md = ["# Day 3 summary", "", "## Few-shot", "", few.to_markdown(index=False), "",
          "## Augmentation", "", aug.to_markdown(index=False), "",
          f"Best strategy: **{best}**", "",
          "## Transfer", "",
          f"- validation accuracy on the 40 seen classes: {tr['best_val_acc_40_categories']:.2f}%",
          f"- k-NN accuracy on the 10 unseen classes: {tr['knn_acc_unseen_categories']:.2f}% (chance 10%)",
          "- these are different tasks (40-way vs 10-way), so the two numbers are not a retention ratio"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nDAY 3 DONE - tables and summary in {out}")


if __name__ == "__main__":
    main()
