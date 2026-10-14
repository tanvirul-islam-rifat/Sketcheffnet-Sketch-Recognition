"""SketchEffNet and baselines: shapes, parameter counts, MACs and checkpoint compatibility."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from tools.complexity import resnet18 as analytic_resnet18
from tools.complexity import sketcheffnet as analytic_sketcheffnet

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


class TestAnalyticComplexity(unittest.TestCase):
    """Runs without PyTorch: the formulas reproduce the counts printed in the Kaggle run."""

    def test_parameter_counts_match_run_log(self):
        self.assertEqual(analytic_sketcheffnet()["params"], 1_001_586)
        self.assertEqual(analytic_resnet18()["params"], 11_202_162)
        self.assertEqual(analytic_resnet18(1000)["params"], 11_689_512)

    def test_macs(self):
        self.assertEqual(analytic_sketcheffnet()["macs"], 49_401_856)
        self.assertEqual(analytic_resnet18()["macs"], 148_071_424)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestSketchEffNet(unittest.TestCase):
    def setUp(self):
        import torch

        from sketcheffnet.model import SketchEffNet, SketchNet, count_parameters
        self.torch, self.SketchNet, self.count = torch, SketchNet, count_parameters
        self.assertIs(SketchEffNet, SketchNet)
        self.model = SketchNet(50).eval()

    def test_parameters(self):
        self.assertEqual(self.count(self.model), 1_001_586)
        self.assertEqual(self.count(self.model), analytic_sketcheffnet()["params"])

    def test_output_and_feature_shapes(self):
        x = self.torch.randn(2, 3, 64, 64)
        self.assertEqual(tuple(self.model(x).shape), (2, 50))
        self.assertEqual(tuple(self.model.features(x).shape), (2, 512))

    def test_stage_resolutions(self):
        seen = {}
        for name in ("stem", "stage1", "stage2", "stage3", "stage4"):
            getattr(self.model, name).register_forward_hook(lambda m, i, o, n=name: seen.__setitem__(n, tuple(o.shape[1:])))
        self.model(self.torch.randn(1, 3, 64, 64))
        self.assertEqual(seen, {"stem": (32, 32, 32), "stage1": (64, 32, 32), "stage2": (128, 16, 16),
                                "stage3": (256, 8, 8), "stage4": (512, 4, 4)})

    def test_macs_match_analytic_model(self):
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            self.model(self.torch.randn(1, 3, 64, 64))
        self.assertEqual(fc.get_total_flops() // 2, analytic_sketcheffnet()["macs"])

    def test_state_dict_keys_match_original_checkpoints(self):
        keys = set(self.model.state_dict())
        for k in ("stem.0.weight", "stage1.0.conv.depthwise.weight", "stage1.0.se.excitation.0.weight",
                  "stage1.0.skip.0.weight", "stage4.1.conv.pointwise.weight", "fc.weight", "fc.bias"):
            self.assertIn(k, keys)

    def test_seeded_initialisation_is_deterministic(self):
        self.torch.manual_seed(42)
        a = self.SketchNet(50).state_dict()
        self.torch.manual_seed(42)
        b = self.SketchNet(50).state_dict()
        self.assertTrue(all(self.torch.equal(a[k], b[k]) for k in a))

    def test_checkpoint_round_trip(self):
        from sketcheffnet.training import load_checkpoint
        path = Path(tempfile.mkdtemp()) / "ckpt.pth"
        self.torch.save({"epoch": 0, "model_state_dict": self.model.state_dict(), "val_acc": 1.0}, path)
        fresh = load_checkpoint(self.SketchNet(50), path)
        x = self.torch.randn(1, 3, 64, 64)
        self.assertTrue(self.torch.allclose(fresh.eval()(x), self.model(x)))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestBaselines(unittest.TestCase):
    EXPECTED = {  # parameter counts printed in the recorded Day 4 run
        "simple_cnn": 2_203_506, "mobilenet_v2": 2_287_922, "resnet18": 11_202_162,
        "resnet34": 21_310_322, "densenet121": 7_005_106, "efficientnet_b0": 4_071_598,
    }

    def test_parameter_counts_match_run_log(self):
        from sketcheffnet.baselines import create_model
        from sketcheffnet.model import count_parameters
        for name, n in self.EXPECTED.items():
            with self.subTest(model=name):
                self.assertEqual(count_parameters(create_model(name, 50, pretrained=False)), n)

    def test_resnet18_macs_match_analytic_model(self):
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        from sketcheffnet.baselines import create_model
        m = create_model("resnet18", 50, pretrained=False).eval()
        with FlopCounterMode(display=False) as fc:
            m(torch.randn(1, 3, 64, 64))
        self.assertEqual(fc.get_total_flops() // 2, analytic_resnet18()["macs"])

    def test_unknown_model(self):
        from sketcheffnet.baselines import create_model
        with self.assertRaises(ValueError):
            create_model("alexnet")


if __name__ == "__main__":
    unittest.main()
