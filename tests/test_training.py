"""Training loop behaviour: it learns, checkpoints the best epoch and stops early."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestFit(unittest.TestCase):
    def setUp(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from sketcheffnet.data import set_seed
        set_seed(0)
        self.torch = torch
        # two linearly separable "classes": bright vs dark images
        x = torch.cat([torch.randn(32, 3, 16, 16) + 2, torch.randn(32, 3, 16, 16) - 2])
        y = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
        self.loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        self.val = DataLoader(TensorDataset(x, y), batch_size=16)
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 2))

    def test_learns_and_checkpoints(self):
        from sketcheffnet.training import evaluate, fit
        ckpt = Path(tempfile.mkdtemp()) / "best.pth"
        opt = self.torch.optim.Adam(self.model.parameters(), lr=1e-2)
        run = fit(self.model, self.loader, self.val, self.torch.nn.CrossEntropyLoss(), opt, 3, "cpu",
                  checkpoint_path=ckpt)
        self.assertTrue(ckpt.exists())
        self.assertEqual(run["epochs_run"], 3)
        self.assertGreaterEqual(run["best_val_acc"], 95.0)
        acc, f1 = evaluate(self.model, self.val, "cpu")
        self.assertGreaterEqual(acc, 95.0)
        self.assertEqual(set(self.torch.load(ckpt, weights_only=True)), {"epoch", "model_state_dict",
                                                                          "optimizer_state_dict", "val_acc", "train_acc"})

    def test_early_stopping(self):
        from sketcheffnet.training import fit
        for p in self.model.parameters():          # all-zero logits -> always class 0 -> constant 50%
            self.torch.nn.init.zeros_(p)
        opt = self.torch.optim.SGD(self.model.parameters(), lr=0.0)   # never improves after epoch 1
        run = fit(self.model, self.loader, self.val, self.torch.nn.CrossEntropyLoss(), opt, 20, "cpu", patience=2)
        self.assertEqual(run["epochs_run"], 3)
        self.assertEqual(run["best_epoch"], 1)

    def test_plateau_scheduler_receives_metric(self):
        from sketcheffnet.training import fit
        opt = self.torch.optim.SGD(self.model.parameters(), lr=0.1)
        sched = self.torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=0)
        run = fit(self.model, self.loader, self.val, self.torch.nn.CrossEntropyLoss(), opt, 4, "cpu",
                  scheduler=sched, scheduler_on_metric=True)
        self.assertEqual(len(run["history"]["lr"]), 4)


if __name__ == "__main__":
    unittest.main()
