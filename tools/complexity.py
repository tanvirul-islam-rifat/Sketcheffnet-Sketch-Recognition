"""
Analytic model complexity: parameters and multiply-accumulates (MACs) per layer.

Pure Python (no PyTorch), so it runs anywhere, CI included. The formulas are
checked three ways:

* they reproduce the parameter counts printed in the recorded Kaggle run:
  SketchEffNet 1,001,586, ResNet18 (50-class head) 11,202,162 and ResNet18
  (1,000-class head) 11,689,512;
* ``tests/test_complexity.py`` compares them with ``torch`` parameter counts;
* the same test compares the MACs with ``torch.utils.flop_counter`` when PyTorch is installed.

MACs cover convolutions and linear layers at a 64x64 input. BatchNorm, ReLU,
pooling and the SE multiply are elementwise and left out, as is usual.

    python -m tools.complexity
"""
from __future__ import annotations

import json
import sys

SKETCHEFFNET_BLOCKS = [  # (in, out, stride) for stage1..stage4
    ("stage1", [(32, 64, 1), (64, 64, 1)]),
    ("stage2", [(64, 128, 2), (128, 128, 1), (128, 128, 1)]),
    ("stage3", [(128, 256, 2), (256, 256, 1), (256, 256, 1), (256, 256, 1)]),
    ("stage4", [(256, 512, 2), (512, 512, 1)]),
]


def sketcheffnet(num_classes: int = 50, size: int = 64) -> dict:
    stages = {}
    h = size // 2
    stages["stem"] = {"params": 3 * 32 * 9 + 2 * 32, "macs": h * h * 32 * 3 * 9, "resolution": h}
    for name, blocks in SKETCHEFFNET_BLOCKS:
        p = m = 0
        for cin, cout, s in blocks:
            ho = h // s
            p += cin * 9 + cin * cout + 2 * cout            # depthwise + pointwise + BN
            m += ho * ho * cin * 9 + ho * ho * cin * cout
            r = cout // 16
            p += 2 * cout * r                                 # SE (two bias-free linears)
            m += 2 * cout * r
            if s != 1 or cin != cout:                         # projection shortcut
                p += cin * cout + 2 * cout
                m += ho * ho * cin * cout
            h = ho
        stages[name] = {"params": p, "macs": m, "resolution": h}
    stages["head"] = {"params": 512 * num_classes + num_classes, "macs": 512 * num_classes, "resolution": 1}
    return _total("SketchEffNet", stages)


def resnet18(num_classes: int = 50, size: int = 64) -> dict:
    stages = {}
    h = size // 2
    stages["stem"] = {"params": 3 * 64 * 49 + 2 * 64, "macs": h * h * 64 * 3 * 49, "resolution": h}
    h //= 2  # max-pool
    cin = 64
    for i, (cout, s) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], start=1):
        p = m = 0
        for b in range(2):
            st = s if b == 0 else 1
            ho = h // st
            p += cin * cout * 9 + cout * cout * 9 + 4 * cout
            m += ho * ho * cin * cout * 9 + ho * ho * cout * cout * 9
            if st != 1 or cin != cout:
                p += cin * cout + 2 * cout
                m += ho * ho * cin * cout
            cin, h = cout, ho
        stages[f"layer{i}"] = {"params": p, "macs": m, "resolution": h}
    stages["head"] = {"params": 512 * num_classes + num_classes, "macs": 512 * num_classes, "resolution": 1}
    return _total("ResNet18", stages)


def _total(name: str, stages: dict) -> dict:
    return {"model": name, "stages": stages,
            "params": sum(s["params"] for s in stages.values()),
            "macs": sum(s["macs"] for s in stages.values())}


def main() -> int:
    a, b = sketcheffnet(), resnet18()
    for r in (a, b):
        print(f"\n{r['model']}: {r['params']:,} parameters, {r['macs'] / 1e6:.1f} M MACs @ 64x64")
        for k, s in r["stages"].items():
            print(f"  {k:<7} {s['resolution']:>3}x{s['resolution']:<3} params {s['params']:>10,}   "
                  f"MACs {s['macs'] / 1e6:7.2f} M  ({s['macs'] / r['macs'] * 100:4.1f}%)")
    print(f"\nParameters: {(1 - a['params'] / b['params']) * 100:.1f}% fewer "
          f"({b['params'] / a['params']:.1f}x)   MACs: {(1 - a['macs'] / b['macs']) * 100:.1f}% fewer "
          f"({b['macs'] / a['macs']:.1f}x)")
    if "--json" in sys.argv:
        print(json.dumps({"sketcheffnet": a, "resnet18": b}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
