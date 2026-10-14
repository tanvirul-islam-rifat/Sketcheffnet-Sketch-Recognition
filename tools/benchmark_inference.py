"""
Measure inference cost on this machine: parameters, MACs, checkpoint size and latency.

The paper lists "no inference latency" as a limitation. This tool measures it for
SketchEffNet and the main baselines at the 64x64 input size used throughout.
Weights are random, which does not affect speed.

    python -m tools.benchmark_inference                 # CPU, batch 1 and 64
    python -m tools.benchmark_inference --device mps    # Apple Silicon GPU

Writes ``results/benchmark_<device>.json``; the numbers depend on the hardware,
so the file records which machine and settings produced them.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from sketcheffnet.baselines import DISPLAY_NAMES, create_model
from sketcheffnet.model import count_parameters

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["sketcheffnet", "mobilenet_v2", "efficientnet_b0", "resnet18"]


def macs(model, x) -> int:
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() // 2


def latency_ms(model, x, device, warmup=10, runs=50) -> float:
    times = []
    with torch.inference_mode():
        for i in range(warmup + runs):
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            t = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    rows = []
    for key in MODELS:
        model = create_model(key, 50, pretrained=False).eval().to(args.device)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        x1 = torch.randn(1, 3, 64, 64, device=args.device)
        x64 = torch.randn(64, 3, 64, 64, device=args.device)
        row = {"model": DISPLAY_NAMES[key], "parameters": count_parameters(model),
               "macs_millions": macs(model.cpu(), x1.cpu()) / 1e6, "state_dict_mb": buf.tell() / 1024 ** 2}
        model.to(args.device)
        row["latency_ms_batch1"] = latency_ms(model, x1, args.device)
        row["latency_ms_batch64"] = latency_ms(model, x64, args.device, runs=20)
        row["throughput_img_per_s"] = 64 / row["latency_ms_batch64"] * 1000
        rows.append(row)
        print(f"{row['model']:<16} {row['parameters']:>11,} params  {row['macs_millions']:7.1f} M MACs  "
              f"{row['state_dict_mb']:6.2f} MB  {row['latency_ms_batch1']:7.2f} ms (b=1)  "
              f"{row['throughput_img_per_s']:8.0f} img/s (b=64)")
    out = ROOT / "results" / f"benchmark_{args.device}.json"
    out.write_text(json.dumps({
        "device": args.device, "platform": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "threads": torch.get_num_threads(), "input": "64x64 RGB", "models": rows,
    }, indent=2) + "\n")
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
