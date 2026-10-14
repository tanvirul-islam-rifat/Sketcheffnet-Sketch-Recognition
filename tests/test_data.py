"""Data preparation: sampling, rendering, splitting and inference preprocessing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from sketcheffnet.config import CATEGORIES
from sketcheffnet.data import (find_npy_file, image_folder_dataframe, load_npy_file, npy_to_image,
                               prepare_dataset_from_npy, split_dataset)
from sketcheffnet.predict import to_quickdraw_style


def fake_df(n_per_class=100, classes=5):
    return pd.DataFrame([{"image_path": f"{c}/{i}.png", "category": f"c{c}", "label": c}
                         for c in range(classes) for i in range(n_per_class)])


class TestSplit(unittest.TestCase):
    def test_ratios_per_class(self):
        tr, va, te = split_dataset(fake_df(2000, 3))
        for part, n in ((tr, 1400), (va, 300), (te, 300)):
            self.assertEqual(set(part["category"].value_counts()), {n})

    def test_disjoint_and_complete(self):
        df = fake_df()
        tr, va, te = split_dataset(df)
        sets = [set(p["image_path"]) for p in (tr, va, te)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set().union(*sets), set(df["image_path"]))

    def test_deterministic(self):
        a, b = split_dataset(fake_df(), seed=42), split_dataset(fake_df(), seed=42)
        for x, y in zip(a, b):
            pd.testing.assert_frame_equal(x, y)
        self.assertFalse(split_dataset(fake_df(), seed=1)[0].equals(a[0]))


class TestNpy(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = np.random.RandomState(0).randint(0, 256, size=(500, 784)).astype(np.uint8)

    def test_sampling_matches_original_global_seed_behaviour(self):
        np.save(self.tmp / "apple.npy", self.data)
        np.random.seed(42)                               # what the original utils.py did
        legacy = self.data[np.random.choice(500, 100, replace=False)]
        np.testing.assert_array_equal(load_npy_file(self.tmp / "apple.npy", 100, seed=42), legacy)

    def test_no_sampling_when_small(self):
        np.save(self.tmp / "apple.npy", self.data)
        self.assertEqual(len(load_npy_file(self.tmp / "apple.npy", 1000)), 500)

    def test_file_name_conventions(self):
        for name in ("alarm clock.npy", "alarm_clock.npy", "full_numpy_bitmap_alarm clock.npy"):
            d = Path(tempfile.mkdtemp())
            np.save(d / name, self.data[:2])
            self.assertEqual(find_npy_file(d, "alarm clock").name, name)
        self.assertIsNone(find_npy_file(self.tmp, "airplane"))

    def test_render(self):
        img = npy_to_image(self.data[0])
        self.assertEqual((img.size, img.mode), ((64, 64), "L"))

    def test_prepare_labels_follow_category_order(self):
        cats = CATEGORIES[:3]
        for c in cats:
            np.save(self.tmp / f"{c}.npy", self.data[:10])
        df = prepare_dataset_from_npy(self.tmp, cats, 5, self.tmp / "out")
        self.assertEqual(len(df), 15)
        self.assertEqual(dict(zip(df["category"], df["label"])), {c: i for i, c in enumerate(cats)})
        self.assertTrue(all(Path(p).exists() for p in df["image_path"]))

    def test_missing_directory_has_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            prepare_dataset_from_npy(self.tmp / "nope", ["apple"], 5, self.tmp / "out")

    def test_image_folder_dataframe(self):
        for cls in ("n001", "n002"):
            (self.tmp / cls).mkdir()
            Image.new("RGB", (8, 8)).save(self.tmp / cls / "a.JPEG")
        (self.tmp / "n002" / "notes.txt").write_text("x")
        df = image_folder_dataframe(self.tmp, ["n001", "n002"])
        self.assertEqual(list(df["label"]), [0, 1])


class TestInferencePreprocessing(unittest.TestCase):
    def test_dark_on_light_drawing_is_inverted_and_framed(self):
        im = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(im).ellipse((150, 100, 250, 200), outline="black", width=8)
        out = np.asarray(to_quickdraw_style(im))
        self.assertEqual(out.shape, (64, 64))
        self.assertLess(out.mean(), 100)          # mostly black background
        self.assertGreater(out.max(), 150)        # bright strokes
        rows = np.nonzero(out.max(axis=1) > 64)[0]
        self.assertGreater(np.ptp(rows), 40)  # cropped to the strokes -> fills the frame

    def test_blank_image_does_not_crash(self):
        self.assertEqual(to_quickdraw_style(Image.new("L", (50, 50), 255)).size, (64, 64))


if __name__ == "__main__":
    unittest.main()
