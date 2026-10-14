# Original scripts (archived, unmodified)

These are the scripts as they stood when the recorded results were produced, kept for
provenance so the refactor can be diffed against them. **Do not run them.** Use the
maintained code in `sketcheffnet/` and `experiments/` instead.

| Original | Replaced by |
|---|---|
| `utils.py` | `sketcheffnet/data.py`, `sketcheffnet/plotting.py` |
| `local_config.py` | `sketcheffnet/config.py` (also auto-detects Kaggle) |
| `sketchnet_model.py` | `sketcheffnet/model.py` (same layers and state-dict keys) |
| `day1_setup_baseline.py` | `experiments/day1_baseline.py` |
| `day2_experiments.py` | `experiments/day2_experiments.py` |
| `day3_analysis_and_paper.py` | `experiments/day3_analysis.py` (the template-paper generator is retired) |
| `day4_other_models_test.py` | `experiments/day4_model_comparison.py` |
| `day5_custom_efficient_model.py` | `experiments/day5_sketcheffnet.py` |
| `day6_cross_dataset_validation.py` | `experiments/day6_cross_dataset_imagenet_sketch.py` |
| `master_pipeline_v2.py` | `experiments/cross_dataset_tuberlin.py` |
| `check_dataset.py`, `download_dataset.py` | `scripts/check_dataset.py`, `scripts/download_quickdraw.py` |
| `local_orchestrator.ipynb` | `notebooks/run_pipeline.ipynb` |

Days 1-5 here are the versions adapted for running locally (paths from `local_config.py`).
The Kaggle run used the same code with `/kaggle/...` paths. Day 6 is the Kaggle version.
The run itself, with all output, is `notebooks/kaggle_full_run.ipynb`.

The refactor keeps every hyperparameter, transform, split, seed and evaluation rule. The
behavioural changes are listed in [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md#what-changed-in-the-refactor).
