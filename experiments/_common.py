"""
Shared plumbing for the experiment scripts: CLI flags, output folders, device.

Every experiment accepts ``--smoke``: the same code path with tiny settings
(30 images per class, 1 epoch, no pretrained-weight downloads), writing to
``outputs/smoke/``. CI uses it on synthetic data to prove the whole pipeline
runs end to end. Without ``--smoke``, every hyperparameter is exactly the one
used for the recorded Kaggle run.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sketcheffnet import config


@dataclass
class RunPaths:
    root: Path
    processed: Path
    models: Path
    results: Path

    def stage(self, name: str) -> Path:
        p = self.results / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def parse_args(description: str, extra=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny, fast configuration to check the pipeline end to end")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader worker processes")
    if extra:
        extra(parser)
    return parser.parse_args()


def run_paths(smoke: bool) -> RunPaths:
    root = config.OUTPUT_DIR / "smoke" if smoke else config.OUTPUT_DIR
    paths = RunPaths(root, root / "processed_data", root / "models", root / "results")
    config.ensure_dirs(paths.processed, paths.models, paths.results)
    return paths


def loader_kwargs(device: str, workers: int | None, smoke: bool) -> dict:
    n = workers if workers is not None else (0 if smoke else 2)
    return {"num_workers": n, "pin_memory": device == "cuda"}


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"  saved {path}")
