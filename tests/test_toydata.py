"""The synthetic quick-start dataset is valid input for the whole pipeline."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sensorline.config import load_config, load_data_description
from sensorline.data import build_dataset
from sensorline.toydata import make_toy_dataset

ROOT = Path(__file__).resolve().parents[1]


class ToyDatasetTests(unittest.TestCase):
    def test_dataset_reads_like_the_real_one_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            first = make_toy_dataset(Path(folder) / "a", train=12, val=4, seed=5)
            second = make_toy_dataset(Path(folder) / "b", train=12, val=4, seed=5)
            data = load_data_description(first)
            self.assertEqual((data["sensor_axis"], data["scale"], data["offset"]), ("x", 255.0, 0.0))
            for split, frames in (("train", 12), ("val", 4)):
                dataset = build_dataset(data, split)
                self.assertEqual(len(dataset), frames)
                sample = dataset[0]
                self.assertEqual(tuple(sample["signal"].shape), (1, 40, 40))
                self.assertGreaterEqual(float(sample["signal"].min()), 0.0)
                self.assertLessEqual(float(sample["signal"].max()), 1.0)
                for boxes in dataset.boxes:
                    sizes = (boxes[:, 2:] - boxes[:, :2]).numpy() * 40
                    # [time, sensor]: objects are about 4-10 steps long and 3-5 sensors wide.
                    self.assertTrue(np.all((sizes[:, 0] > 3.5) & (sizes[:, 0] < 11)))
                    self.assertTrue(np.all((sizes[:, 1] > 3) & (sizes[:, 1] < 5.5)))
            for path in sorted(first.rglob("*.*")):
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
            with self.assertRaisesRegex(ValueError, "not empty"):
                make_toy_dataset(first)

    def test_length_and_brightness_ranges_give_another_distribution(self):
        with tempfile.TemporaryDirectory() as folder:
            data = load_data_description(make_toy_dataset(Path(folder) / "long", train=10, val=2, seed=5,
                                                          lengths=(14.0, 18.0), brightness=(0.3, 0.4)))
            dataset = build_dataset(data, "train")
            lengths = np.concatenate([(boxes[:, 2] - boxes[:, 0]).numpy() * 40 for boxes in dataset.boxes])
            self.assertTrue(np.all((lengths > 13.9) & (lengths < 18.1)))
            # Peaks stay below the brightest object plus the background and its noise.
            self.assertLess(max(float(dataset[index]["signal"].max()) for index in range(len(dataset))), 0.5)
            for options, message in (({"lengths": (5.0, 41.0)}, "lengths"), ({"lengths": (6.0, 5.0)}, "lengths"),
                                     ({"brightness": (0.0, 0.5)}, "brightness"),
                                     ({"brightness": (0.5, 1.5)}, "brightness")):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                    make_toy_dataset(Path(folder) / "bad", **options)

    def test_quick_start_config_names_the_toy_dataset(self):
        config = load_config(ROOT / "configs" / "toy.yaml")
        self.assertEqual(Path(config["data"]["train"]), (ROOT / "toy_data" / "train").resolve())
        self.assertEqual(config["model"], load_config(ROOT / "configs" / "theory_00_control.yaml")["model"])


if __name__ == "__main__":
    unittest.main()
