"""The test split: monitored during training, run from the dashboard, and never used to choose anything."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from sensorline.analysis.web import AnalysisState, make_server
from sensorline.config import load_config, load_data_description
from sensorline.toydata import make_toy_dataset
from sensorline.trainer import train


def quiet_train(config, name):
    with redirect_stdout(io.StringIO()):
        return train(config, name=name)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_losses(directory):
    return [row["train_loss_total"] for row in read(directory / "history.json")]


class TestSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="sensorline_test_split_")
        root = cls.root = Path(cls.temporary.name).resolve()
        config = load_config(ROOT / "tests/fixtures/baseline.yaml")
        config["data"] = load_data_description(make_toy_dataset(root / "toy", train=8, val=4, test=4, seed=3))
        config["model"] = {"backbone": [{"type": "conv", "channels": 2, "kernel": [3, 3]}],
                           "neck": {"type": "none"}, "head": {"type": "bbox"}}
        config["training"].update(epochs=2, batch_size=4, device="cpu", cpu_threads=2, num_workers=0)
        config["evaluation"].update(max_detections=5, pre_nms_topk=10)
        cls.config = config
        # The same run twice: with its test split monitored, and without one.
        tested, plain = deepcopy(config), deepcopy(config)
        del plain["data"]["test"]
        tested["training"]["output_dir"] = str(root / "runs" / "tested")
        plain["training"]["output_dir"] = str(root / "runs" / "plain")
        cls.tested = quiet_train(tested, "tested")
        cls.plain = quiet_train(plain, "plain")

        configs = root / "configs"
        configs.mkdir()
        (configs / "model.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        cls.state = AnalysisState(config["data"], root / "analysis", 17, runs_root=root / "runs",
                                  config_path=configs / "model.yaml")
        cls.server = make_server(cls.state, 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.temporary.cleanup()

    def get(self, path):
        with urlopen(self.base + path) as response:
            return json.load(response)

    def post(self, path, body):
        request = Request(self.base + path, data=json.dumps(body).encode(), method="POST")
        with urlopen(request) as response:
            return json.load(response)

    def infer(self, run_id, source):
        self.post("/api/inference/start", {"source": source, "checkpoint": run_id, "device": "cpu"})
        for _ in range(600):
            state = self.get("/api/inference/status")["inference"]
            if state["status"] != "running":
                return state
            time.sleep(0.2)
        self.fail("Inference did not finish")

    def test_every_epoch_reports_the_test_split_and_it_changes_nothing_else(self):
        history = read(self.tested / "history.json")
        self.assertTrue(all("test_f1" in row and "test_count_mae" in row for row in history))
        self.assertTrue((self.tested / "test_metrics.json").is_file())
        self.assertIn("best_test_metrics", read(self.tested / "summary.json"))
        # A run without a test split leaves no trace of one.
        self.assertFalse(any(key.startswith("test_") for row in read(self.plain / "history.json") for key in row))
        self.assertNotIn("best_test_metrics", read(self.plain / "summary.json"))
        # Training itself is the same: the test pass takes no part in it.
        self.assertEqual(train_losses(self.tested), train_losses(self.plain))

    def test_a_run_is_evaluated_on_its_own_test_split(self):
        run = read(self.tested / "summary.json")["id"]
        row = next(row for row in self.get("/api/inference/checkpoints") if row["id"] == run)
        self.assertEqual(row["splits"]["test"]["origin"], "run")
        state = self.infer(run, "test")
        self.assertEqual(state["status"], "completed", state["error"])
        self.assertEqual(state["metrics"]["samples"], 4)
        # The numbers the training reported for its best epoch are those of this evaluation.
        self.assertAlmostEqual(state["metrics"]["f1"], read(self.tested / "test_metrics.json")["metrics"]["f1"])

    def test_a_run_without_a_test_split_takes_the_test_folder_of_the_model_yaml(self):
        run = read(self.plain / "summary.json")["id"]
        row = next(row for row in self.get("/api/inference/checkpoints") if row["id"] == run)
        self.assertEqual(row["splits"]["test"]["origin"], "model.yaml")
        state = self.infer(run, "test")
        self.assertEqual(state["status"], "completed", state["error"])
        self.assertEqual(Path(state["data_root"]), self.root / "toy" / "test")


if __name__ == "__main__":
    unittest.main()
