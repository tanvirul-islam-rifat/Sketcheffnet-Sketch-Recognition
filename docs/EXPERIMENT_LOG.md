# Experiment Log

How the project was run, what each stage produced, and how the code changed from the
original Kaggle notebooks to this repository.

## How the recorded run was produced

Each stage was first written as its own Kaggle notebook (`utils`, `check-dataset`,
`day1-setup-baseline` … `day6-cross-dataset-validation`). A single orchestrator notebook
attached all of them as inputs, converted each to a `.py` file with `jupyter nbconvert`,
and executed them in order on one **Tesla T4** session, along with the two datasets
(`quickdraw20` and ImageNet-Sketch). That orchestrator, with every line of output, is
[`notebooks/kaggle_full_run.ipynb`](../notebooks/kaggle_full_run.ipynb). It is the source
of truth for every number in this repository (parsed by `tools/extract_kaggle_results.py`).

Afterwards the pipeline was ported to run locally on an Apple Silicon Mac (MPS backend)
through a central `local_config.py`, and a same-label cross-dataset extension (TU-Berlin)
was added. Those versions are in [`archive/original_scripts/`](../archive/original_scripts/).

## Timeline

| Stage | What it did | Key result | Time (T4) |
|---|---|---|---|
| Check | 50 `.npy` files, shape (N, 784), uint8 | all present | — |
| **Day 1** | Sample 2,000/class → 64×64 PNG, 70/15/15 split; ResNet18 baseline | **83.49 %** test, best val 83.65 % (epoch 19 of 20) | 14.37 min |
| **Day 2** | Few-shot ×5, augmentation ×3, 40→10 class transfer (ResNet18) | 49.65 → 80.25 %; 74.10 / 78.62 / 78.15 %; 82.89 % k-NN on unseen | ~45 min |
| Day 3 | Tables, plots and a *template* paper draft | — | seconds |
| **Day 4** | 6 CNNs + 3 classical models, one protocol | EfficientNet-B0 82.63 % best; k-NN 44.50 % best classical | ~80 min |
| — | ResNet18 parameter audit | 11,202,162 (50-class head) | — |
| **Day 5** | SketchEffNet from scratch | **81.45 %** test, 1,001,586 params | 36.49 min |
| **Day 6** | QuickDraw → ImageNet-Sketch (first 50 classes) | 2.09 / 7.83 / 8.62 % | ~6 min + 37 min accidental Day 5 rerun |

### Day 5 in detail

The cosine schedule restarts at the end of epochs 5 and 15, and validation accuracy dips right
after each restart (77.0 → 71.0 %, 82.2 → 74.1 %). The best checkpoint (82.16 %) is epoch 15,
the low-learning-rate end of the second cycle. The third cycle (20 epochs long) never got
back above it, so early stopping (patience 7) ended training at epoch 22. A longer run,
or a schedule without a restart after epoch 15, might gain a little. This is untested.

### The accidental rerun (Day 6)

`day6_cross_dataset_validation.py` imported `SketchNet` from `day5_custom_efficient_model.py`,
a flat script with no `if __name__ == "__main__"` guard, so the import re-executed the whole
Day 5 training. The second run matched the first in all 22 epochs (losses, accuracies,
learning rates) and in the final 81.45 % / 81.38 %; only the wall time differed (37.01 vs
36.49 min). Seeded, deterministic cuDNN made the pipeline bit-for-bit reproducible on the same
hardware. The verifier keeps that as a check.

## Archived exploration: TensorFlow/Keras baseline

Before the PyTorch pipeline, [`notebooks/archive/tf_keras_baseline.ipynb`](../notebooks/archive/tf_keras_baseline.ipynb)
tried Keras applications on 28×28 inputs resized to 128 (224 for Swin), 7 epochs, with an
80/20 split used as validation:

| Model | Final val. accuracy | Params |
|---|---|---|
| ResNet50 | 80.47 % | 24.1 M |
| MobileNetV3-Small | 67.66 % | 1.10 M |
| EfficientNet-B0 | 9.53 % | 4.39 M |
| MobileNetV3 + CBAM | 2.19 % | 1.10 M |
| Swin-Tiny | 1.82 % | 27.7 M |

Several models collapsed: training accuracy rose to 83–91 % while validation accuracy swung
between 2 % and 75 %. Swin-Tiny's loss sat at 3.912 = ln(50), meaning it never left the uniform
prediction. A likely cause (untested) is input scaling. Keras' MobileNetV3 and EfficientNet
models include their own rescaling layer and expect pixels in 0–255, but the notebook fed 0–1.
Combined with BatchNorm statistics in train vs inference mode, that is a known recipe for this
train/val split. The PyTorch pipeline normalises explicitly and does not show the problem. The
notebook's "robustness" test (40 % of pixels dropped) gave 4.0 % and 1.6 %, so it measured broken
models, not robustness. It is kept only as a record.

## What changed in the refactor

Behaviour that **did not change**: every hyperparameter, transform, split, seed, model
definition (layer names and state-dict keys included, so the original `sketchnet_best.pth`
loads), checkpoint rule and evaluation rule of the recorded run. Day 4 still evaluates
last-epoch weights, and Day 6 still uses a fresh head for "zero-shot", so a rerun measures the
same thing as the recorded one.

| Change | Why |
|---|---|
| Scripts → `sketcheffnet/` package + `experiments/` entry points with `main()` | No code runs on import (the Day 6 bug); one shared `fit()` loop replaces four copies |
| `utils.py` no longer reseeds RNGs at import; sampling uses `RandomState(seed)` | Same indices as before (a unit test proves it), without global side effects |
| `config.py` auto-detects Kaggle, else local paths, else env vars | The same code runs on Kaggle, a Mac or CI without edits |
| `pretrained=True` → `weights=*_Weights.IMAGENET1K_V1` | Same weights, no deprecation warning |
| `torch.load(..., weights_only=True, map_location=...)` | Safe loading; works on CPU/MPS for checkpoints saved on CUDA |
| Day 5 reports the reduction against the 50-class ResNet18 | Finding 1 |
| Day 4 `--with-sketcheffnet` | Like-for-like comparison (Finding 4) |
| Day 6 imports the model module; image lists are sorted | No 37-minute side effect; deterministic file order (the original used unsorted `os.listdir`, so a rerun's ImageNet-Sketch split can differ from the recorded one) |
| Day 3's template paper generator retired | It held placeholders and the claim "sketch-specific augmentation is superior", which the data contradicts; the real paper is `docs/SketchEffNet_Paper.pdf` |
| `--smoke` on every stage + `scripts/make_synthetic_data.py` | CI runs the full pipeline end to end in minutes |
| `sketcheffnet/predict.py` | Use the model on real drawings |
| `tools/` (extract, verify, complexity, figures, benchmark) | Results as data; claims checked automatically |

## Paper notes

The paper ([`SketchEffNet_Paper.pdf`](SketchEffNet_Paper.pdf)) is kept as submitted. Where it
and the run disagree, the README and `results/` hold the corrected figures (Findings 1–7). One
more minor mismatch: the paper describes SketchEffNet's augmentation as "rotation ±15°,
translation ±10 %". The Day 5 code used rotation ±20°, translation ±15 %, scale 0.85–1.15, flip
0.3 and elastic deformation (p = 0.3).
