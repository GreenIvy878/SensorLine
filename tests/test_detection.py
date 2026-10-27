import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from model_fixtures import model_config
from PIL import Image

from sensorline.config import load_config
from sensorline.data import (
    YoloImageDataset,
    audit_sources,
    collate_samples,
    dataset_fingerprint,
)
from sensorline.evaluation import (
    confidence_metrics,
    decode_predictions,
    detection_metrics,
    prepare_candidates,
    suppress_sample,
)
from sensorline.losses import DetectionLoss, bbox_loss
from sensorline.models import Detector, model_info
from sensorline.models.builder import validate_model_config
from sensorline.toydata import make_toy_dataset

ROOT = Path(__file__).resolve().parents[1]


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_axis_transpose_and_yolo_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            values = np.arange(10 * 40, dtype=np.uint16).reshape(10, 40) % 256
            Image.fromarray(values.astype(np.uint8)).save(root / "images/a.png")
            (root / "labels/a.txt").write_text("0 0.5 0.3 0.2 0.4\n")

            sample = YoloImageDataset(root)[0]
            self.assertEqual(tuple(sample["signal"].shape), (1, 40, 10))
            self.assertAlmostEqual(float(sample["signal"][0, 7, 3]), values[3, 7] / 255)
            expected_boxes = torch.tensor([[0.1, 0.4, 0.5, 0.6]])
            torch.testing.assert_close(sample["boxes"], expected_boxes)

    def test_empty_missing_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            values = np.zeros((40, 40), dtype=np.uint8)
            Image.fromarray(values).save(root / "images/a.png")
            with self.assertRaises(FileNotFoundError):
                YoloImageDataset(root)

            unlabeled = YoloImageDataset(root, require_labels=False)
            self.assertEqual(len(unlabeled[0]["boxes"]), 0)
            invalid_labels = (
                "1 0.5 0.5 0.2 0.2",
                "0 nan 0.5 0.1 0.1",
                "0 0 0 0.2 0.2",
                "0 1 2",
            )
            for text in invalid_labels:
                (root / "labels/a.txt").write_text(text)
                with self.assertRaises(ValueError):
                    YoloImageDataset(root)

            (root / "labels/a.txt").write_text("")
            sample = YoloImageDataset(root)[0]
            self.assertEqual(tuple(sample["boxes"].shape), (0, 4))

    def test_all_configs_dummy_forward_and_real_backward(self):
        # Frames in the format of the real data, including a crowded one.
        with tempfile.TemporaryDirectory() as folder:
            dataset = YoloImageDataset(make_toy_dataset(Path(folder) / "toy", train=40, val=1, seed=3) / "train")
            crowded = max(range(len(dataset)), key=lambda index: len(dataset.boxes[index]))
            self.assertGreaterEqual(len(dataset.boxes[crowded]), 8)
            signal, targets = collate_samples([dataset[0], dataset[crowded]])

        for name in ("baseline", "example_tcn", "example_fpn"):
            with self.subTest(name=name):
                config = model_config(ROOT, name)
                model = Detector(config["model"])
                info = model_info(model, (2, 1, 40, 53))
                self.assertEqual(info["prediction_shape"][0], 2)
                self.assertEqual(info["backbone"][0]["shape"][-2], 40)

                for kind in ("iou", "giou", "diou", "ciou"):
                    model.zero_grad(set_to_none=True)
                    output = model(signal)
                    losses = DetectionLoss({"bbox": kind})(
                        output, targets, signal.shape[-2:]
                    )
                    self.assertTrue(torch.isfinite(losses["total"]))
                    losses["total"].backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_iou_losses_exact_boxes_and_disjoint_gradients(self):
        target = torch.tensor([[0.1, 0.2, 0.4, 0.6]])
        for kind in ("iou", "giou", "diou", "ciou"):
            self.assertAlmostEqual(
                float(bbox_loss(target, target, kind)), 0.0, places=6
            )
            predicted = torch.tensor([[0.6, 0.6, 0.9, 0.9]], requires_grad=True)
            value = bbox_loss(predicted, target, kind)
            value.sum().backward()
            self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_invalid_architectures(self):
        model_config = load_config(ROOT / "tests/fixtures/baseline.yaml")["model"]
        for change in ("channels", "kernel", "neck", "resolution"):
            config = copy.deepcopy(model_config)
            if change == "channels":
                config["backbone"][1]["channels"] = 0
            elif change == "kernel":
                config["backbone"][0]["kernel"] = [2, 4]
            elif change == "neck":
                config["neck"] = {"type": "fpn", "features": [0, 99]}
            else:
                config["neck"] = {"type": "fpn", "features": [1, 2]}
            with self.assertRaises(ValueError):
                validate_model_config(config)

    def test_metrics_duplicates_empty_and_wrong_localization(self):
        truth = [
            {"id": "a", "boxes": torch.tensor([[0.1, 0.1, 0.4, 0.4]])},
            {"id": "b", "boxes": torch.empty(0, 4)},
        ]
        predictions = [
            {"boxes": truth[0]["boxes"].clone(), "scores": torch.tensor([0.9])},
            {"boxes": torch.empty(0, 4), "scores": torch.empty(0)},
        ]
        metrics, _ = detection_metrics(predictions, truth, 0.3, 0.5)
        self.assertEqual(metrics["map50_95"], 1.0)
        self.assertEqual(metrics["count_mae"], 0.0)

        predictions[0] = {
            "boxes": truth[0]["boxes"].repeat(2, 1),
            "scores": torch.tensor([0.9, 0.8]),
        }
        metrics, _ = detection_metrics(predictions, truth, 0.3, 0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["count_mae"], 0.5)

        predictions[0] = {
            "boxes": torch.tensor([[0.6, 0.6, 0.9, 0.9]]),
            "scores": torch.tensor([0.9]),
        }
        metrics, _ = detection_metrics(predictions, truth, 0.3, 0.5)
        self.assertEqual(metrics["count_mae"], 0)
        self.assertEqual(metrics["map50"], 0)

    def test_nms_removes_duplicates(self):
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.1, 0.1, 0.4, 0.4],
                [0.6, 0.6, 0.9, 0.9],
            ]
        )
        kept = suppress_sample({"boxes": boxes, "scores": torch.tensor([0.9, 0.8, 0.7])}, 0.5, 10)
        torch.testing.assert_close(kept["scores"], torch.tensor([0.9, 0.7]))
        empty = suppress_sample({"boxes": torch.empty(0, 4), "scores": torch.empty(0)}, 0.5, 10)
        self.assertEqual(len(empty["boxes"]), 0)

    def test_confidence_sweep_matches_ordinary_metrics_with_score_ties(self):
        # More than sixteen equal scores exercises torch's shortened-array tie ordering.
        boxes = torch.tensor(
            [[0.1, 0.1, 0.3, 0.3], [0.2, 0.1, 0.4, 0.3], [0.7, 0.7, 0.9, 0.9]] * 7
        )
        predictions = [
            {"boxes": boxes, "scores": torch.tensor([0.8] * 17 + [0.6, 0.6, 0.2, 0.1])},
            {"boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3]]), "scores": torch.tensor([0.6])},
            {"boxes": torch.empty(0, 4), "scores": torch.empty(0)},
        ]
        truth = [
            {"id": "a", "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.1, 0.4, 0.3]])},
            {"id": "b", "boxes": torch.empty(0, 4)},
            {"id": "c", "boxes": torch.tensor([[0.7, 0.7, 0.9, 0.9]])},
        ]
        confidences = [0.1, 0.6, 0.8, 1.0]
        summaries = confidence_metrics(predictions, truth, confidences, 0.3)
        self.assertEqual(list(summaries), confidences)
        for confidence, summary in summaries.items():
            ordinary, _ = detection_metrics(predictions, truth, confidence, 0.3)
            for key, value in summary.items():
                self.assertAlmostEqual(value, ordinary[key], places=12, msg=key)

    def test_candidates_clipping_padding_anchor_filter_topk_and_nms_cap(self):
        evaluation = {"ap_min_confidence": 0.01, "pre_nms_topk": 2}
        boxes = torch.tensor(
            [
                [-0.2, -0.2, 0.3, 0.3],
                [0.2, 0.2, 0.3, 0.3],
                [2.0, 0.2, 3.0, 0.3],
                [0.1, 0.1, 0.2, 0.2],
                [0.4, 0.4, 0.5, 0.5],
            ]
        )
        scores = torch.tensor([0.7, 0.6, 0.9, 0.99, 0.8])
        anchors = torch.tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.8, 0.1], [0.1, 0.1]])
        candidates = prepare_candidates(boxes, scores, anchors, 0.5, evaluation)
        torch.testing.assert_close(candidates["scores"], torch.tensor([0.8, 0.7]))
        torch.testing.assert_close(
            candidates["boxes"], torch.tensor([[0.8, 0.4, 1.0, 0.5], [0.0, 0.0, 0.6, 0.3]])
        )
        capped = suppress_sample(candidates, 0.3, 1)
        torch.testing.assert_close(capped["scores"], torch.tensor([0.8]))

    def test_variable_time_padding(self):
        samples = []
        for index, time_size in enumerate((20, 40)):
            samples.append(
                {
                    "id": str(index),
                    "signal": torch.zeros(1, 40, time_size),
                    "boxes": torch.tensor([[0.2, 0.3, 0.6, 0.7]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                }
            )
        signal, targets = collate_samples(samples)
        self.assertEqual(tuple(signal.shape), (2, 1, 40, 40))

        output = {
            "boxes": torch.tensor(
                [
                    [[0.1, 0.3, 0.3, 0.7], [0.6, 0.2, 0.9, 0.8]],
                    [[0.2, 0.3, 0.6, 0.7], [0.6, 0.2, 0.9, 0.8]],
                ]
            ),
            "objectness": torch.full((2, 2), 10.0),
            "anchors": torch.tensor([[0.2, 0.5], [0.8, 0.5]]),
        }
        config = load_config(ROOT / "tests/fixtures/baseline.yaml")
        decoded = decode_predictions(
            output, targets, signal.shape[-2:], config["evaluation"]
        )
        self.assertEqual(len(decoded[0]["boxes"]), 1)
        torch.testing.assert_close(decoded[0]["boxes"], samples[0]["boxes"])

        model = Detector(config["model"])
        losses = DetectionLoss({})(model(signal), targets, signal.shape[-2:])
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_source_independent_sample_audit_and_fingerprint(self):
        sample = {
            "id": "table-frame",
            "signal": torch.zeros(1, 40, 11),
            "boxes": torch.empty(0, 4),
            "labels": torch.empty(0, dtype=torch.long),
        }
        audit = audit_sources([sample], [sample])
        self.assertEqual(audit["train_samples"], 1)
        self.assertEqual(len(audit["cross_split_duplicates"]), 1)
        fingerprint = dataset_fingerprint([sample])
        self.assertEqual(fingerprint, dataset_fingerprint([sample]))
        sample["signal"][0, 0, 0] = 1
        self.assertNotEqual(fingerprint, dataset_fingerprint([sample]))


if __name__ == "__main__":
    unittest.main()
