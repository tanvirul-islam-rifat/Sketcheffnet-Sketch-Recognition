"""The recorded results are self-consistent, and the verifier catches tampering."""
import copy
import json
import unittest
from pathlib import Path

from tools import extract_kaggle_results as ex
from tools.verify_results import cosine_restart_lr, load_results, run_checks

ROOT = Path(__file__).resolve().parents[1]


def statuses(results):
    return {c.name: c.status for c in run_checks(results)}


class TestRecordedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = load_results()

    def test_no_check_fails_on_the_recorded_run(self):
        failed = [c for c in run_checks(self.r) if c.status == "FAIL"]
        self.assertEqual(failed, [], "\n".join(f"{c.name}: {c.detail}" for c in failed))

    def test_findings_are_the_documented_nine(self):
        self.assertEqual(sum(c.status == "WARN" for c in run_checks(self.r)), 9)

    def test_extractor_reproduces_committed_results(self):
        logs = ex.load_stage_logs()
        self.assertEqual(ex.parse_day1(logs["01_day1_baseline"]), self.r["day1_baseline"])
        self.assertEqual(ex.parse_day2(logs["02_day2_experiments"]), self.r["day2_experiments"])
        day4 = {**ex.parse_day4(logs["04_day4_model_comparison"]),
                **ex.parse_resnet18_params(logs["05_resnet18_parameter_count"])}
        self.assertEqual(json.loads(json.dumps(day4)), self.r["day4_model_comparison"])

    def test_headline_numbers(self):
        self.assertEqual(self.r["day1_baseline"]["test"]["accuracy"], 83.49)
        self.assertEqual(self.r["day5_sketcheffnet"]["test"]["accuracy"], 81.45)
        self.assertEqual(self.r["day5_sketcheffnet"]["parameters"], 1_001_586)

    def test_cosine_restart_formula(self):
        self.assertAlmostEqual(cosine_restart_lr(5), 1e-3)     # first restart
        self.assertAlmostEqual(cosine_restart_lr(15), 1e-3)    # second restart (5 + 10)
        self.assertAlmostEqual(cosine_restart_lr(10), 5e-4)    # middle of the 10-epoch cycle


class TestVerifierCatchesMistakes(unittest.TestCase):
    """Negative tests: corrupt one recorded number and the matching check must FAIL."""

    def setUp(self):
        self.r = copy.deepcopy(load_results())

    def assertFails(self, check_name):
        st = statuses(self.r)
        self.assertEqual(st[check_name], "FAIL", f"{check_name} did not catch the change")

    def test_wrong_headline_accuracy(self):
        self.r["day1_baseline"]["test"]["accuracy"] = 85.0
        self.assertFails("Per-class recalls average to the reported test accuracy (balanced set)")

    def test_unbalanced_test_set(self):
        self.r["day1_baseline"]["per_class"]["cake"]["support"] = 299
        self.assertFails("Test set is balanced: 50 classes x 300 sketches")

    def test_wrong_split(self):
        self.r["day1_baseline"]["dataset"]["train"] = 69_000
        self.assertFails("Split is 70/15/15 of 100,000 sketches")

    def test_learning_rate_log_edited(self):
        self.r["day5_sketcheffnet"]["history"]["lr"][4] = 0.0004
        self.assertFails("Logged learning rates follow CosineAnnealingWarmRestarts(T_0=5, T_mult=2)")

    def test_early_stopping_inconsistent(self):
        self.r["day5_sketcheffnet"]["training"]["epochs_run"] = 25
        self.assertFails("Day 5 early stopping fired exactly 7 epochs after the best epoch")

    def test_parameter_count_does_not_match_architecture(self):
        self.r["day5_sketcheffnet"]["parameters"] = 1_000_000
        self.assertFails("Architecture accounts for SketchEffNet's 1,001,586 parameters")

    def test_nondeterministic_rerun(self):
        self.r["day5_sketcheffnet"]["accidental_rerun"]["history_identical"] = False
        self.assertFails("Training is deterministic: the accidental second Day 5 run reproduced every epoch")

    def test_paper_table_mismatch(self):
        self.r["day4_model_comparison"]["models"][0]["test_acc"] = 70.0
        self.assertFails("Paper Tables IV-V: nine model-comparison accuracies")


if __name__ == "__main__":
    unittest.main()
