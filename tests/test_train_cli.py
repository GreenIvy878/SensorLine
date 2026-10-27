"""train.py modes on TEMP data: ordinary training, an Optuna search and fine-tuning of one model."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensorline.config import load_config


def load_cli():
    spec = importlib.util.spec_from_file_location("sensorline_train_cli", ROOT / "train.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class TrainCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sensorline_train_cli_")
        self.addCleanup(self.temporary.cleanup)
        # The full path: a temporary folder may be given by its short 8.3 name on Windows,
        # while the runs report resolved paths.
        self.directory = Path(self.temporary.name).resolve()
        self.config_path = self.directory / "model.yaml"
        self.output = self.directory / "runs" / "model"
        self.cli = load_cli()
        self.configure_tiny_data()

    def configure_tiny_data(self):
        import numpy as np
        from PIL import Image
        import torch
        torch.set_num_threads(2)
        config = load_config(ROOT / "tests/fixtures/baseline.yaml")
        config["model"] = {"backbone": [{"type": "conv", "channels": 2, "kernel": [3, 3]}],
                           "neck": {"type": "none"}, "head": {"type": "bbox"}}
        config["training"].update(epochs=1, batch_size=2, device="cpu", cpu_threads=2, num_workers=0)
        config["evaluation"].update(max_detections=5, pre_nms_topk=10)
        for split, intensity in (("train", 160), ("val", 180)):
            directory = self.directory / split
            (directory / "images").mkdir(parents=True)
            (directory / "labels").mkdir()
            config["data"][split] = str(directory)
            for index in range(2):
                image = np.zeros((12 + index * 4, 40), dtype=np.uint8)
                image[3:8, 10:16] = intensity
                name = f"{split}_{index}"
                Image.fromarray(image).save(directory / "images" / (name + ".png"))
                (directory / "labels" / (name + ".txt")).write_text(
                    f"0 .325 {5.5 / len(image)} .15 {5 / len(image)}\n", encoding="utf-8",
                )
        self.config = config
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def arguments(self, *flags):
        return ["train.py", "--config", str(self.config_path), "--output-dir", str(self.output), *flags]

    def invoke(self, *flags):
        stream = io.StringIO()
        with patch.object(sys, "argv", self.arguments(*flags)), redirect_stdout(stream):
            result = self.cli.main()
        return result, stream.getvalue()

    def search_path(self):
        path = self.directory / "search.json"
        path.write_text(json.dumps({
            "schema_version": 1, "study_name": "geometry", "output_dir": "ignored",
            "n_trials": 1, "sampler_seed": 17,
            "objective": {"metric": "f1", "direction": "maximize"},
            "parameters": {
                "training.augmentation.rotation_degrees": {"type": "float", "low": 0, "high": 5},
                "training.augmentation.shear_limit": {"type": "float", "low": 0, "high": .1},
            },
        }), encoding="utf-8")
        return path

    def test_dry_run_and_invalid_config_never_start_training(self):
        with patch.object(self.cli, "train") as train:
            _, printed = self.invoke("--dry-run", "--epochs", "3")
            self.assertIn(f"model.yaml -> {self.output.resolve()} (epochs=3)", printed)
            _, printed = self.invoke("--dry-run", "--optuna", str(self.search_path()), "--trials", "4")
            self.assertIn("trials=4, resume=False", printed)
            self.config_path.write_text("invalid: true", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.invoke()
            train.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_yaml_must_write_thresholds_and_loss_instead_of_taking_code_defaults(self):
        written = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        for section, key in (("evaluation", "confidence"), ("evaluation", "nms_iou"),
                             ("evaluation", "match_iou"), ("evaluation", "pre_nms_topk"),
                             ("loss", "objectness"), ("loss", "bbox"), ("loss", "focal_gamma"),
                             ("loss", "hard_negative_ratio")):
            config = yaml.safe_load(yaml.safe_dump(written))
            del config[section][key]
            self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Missing {section}.*{key}"):
                load_config(self.config_path)
        # The focal settings belong to the focal loss only.
        config = yaml.safe_load(yaml.safe_dump(written))
        config["loss"]["objectness"] = "bce"
        del config["loss"]["focal_gamma"], config["loss"]["focal_alpha"]
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.assertEqual(load_config(self.config_path)["loss"]["objectness"], "bce")

    def test_config_is_required_and_sequential_series_options_do_not_exist(self):
        for arguments in (["train.py"],
                          ["train.py", "--all-configs", str(self.directory)],
                          [*self.arguments(), "--continue-on-error"]):
            with self.subTest(arguments=arguments[1:2]), patch.object(sys, "argv", arguments), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                self.cli.parse_arguments()
            self.assertEqual(error.exception.code, 2)

    def test_ordinary_mode_passes_name_and_overrides_to_one_training_run(self):
        with patch.object(self.cli, "train", return_value=self.output) as train, \
                patch.dict(sys.modules, {"optuna": None}):
            result, _ = self.invoke("--epochs", "2", "--device", "cpu", "--name", "test")
        self.assertEqual(result, self.output)
        train.assert_called_once()
        config = train.call_args.args[0]
        self.assertEqual(config["training"]["epochs"], 2)
        self.assertEqual(config["training"]["output_dir"], str(self.output.resolve()))
        self.assertEqual(config["model"], self.config["model"])
        self.assertEqual(train.call_args.kwargs, {"name": "test", "selection": None})

    def test_real_training_writes_one_run_and_retains_source_files(self):
        before = {path: path.read_bytes() for split in ("train", "val")
                  for path in (self.directory / split).rglob("*") if path.is_file()}
        directory, _ = self.invoke()
        self.assertEqual(directory, self.output)
        self.assertTrue((directory / "best.pt").is_file())
        metrics = json.loads((directory / "metrics.json").read_text())["metrics"]
        self.assertTrue({"f1", "map50", "count_mae"} <= set(metrics))
        self.assertEqual(json.loads((directory / "summary.json").read_text())["status"], "completed")
        self.assertFalse(list(self.output.parent.glob("research_*.json")))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_optuna_mode_receives_search_and_occupied_study_requires_resume(self):
        search = self.search_path()
        with patch("sensorline.optuna_search.run_search", return_value=self.output) as run_search:
            self.invoke("--optuna", str(search), "--trials", "3", "--resume", "--name", "study")
        config, spec = run_search.call_args.args
        self.assertEqual(config["model"], self.config["model"])
        self.assertEqual(spec["study_name"], "geometry")
        self.assertEqual(spec["output_dir"], str(self.output.resolve()))
        self.assertEqual(run_search.call_args.kwargs, {"n_trials": 3, "resume": True, "name": "study"})

    def test_finetune_flags_need_their_mode_and_exclude_a_search(self):
        for flags in (["--lr-scale", "0.5"], ["--replay-ratio", "2"], ["--max-f1-drop", "0"],
                      ["--max-count-mae-rise", "0"],
                      ["--finetune", "best.pt", "--optuna", str(self.search_path())]):
            with self.subTest(flags=flags), patch.object(sys, "argv", self.arguments(*flags)), \
                    patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                self.cli.parse_arguments()
            self.assertEqual(error.exception.code, 2)

    def test_finetune_mode_checks_the_base_checkpoint_and_trains_from_it(self):
        base, _ = self.invoke()
        checkpoint = base / "best.pt"
        tuned = self.directory / "runs" / "tuned"
        arguments = ["train.py", "--config", str(self.config_path), "--output-dir", str(tuned),
                     "--finetune", str(checkpoint)]

        def invoke(*flags):
            stream = io.StringIO()
            with patch.object(sys, "argv", [*arguments, *flags]), redirect_stdout(stream):
                return self.cli.main(), stream.getvalue()

        with patch.object(self.cli, "finetune", return_value=tuned) as finetune, \
                patch.object(self.cli, "train") as train:
            _, printed = invoke("--dry-run", "--lr-scale", "0.5")
            self.assertIn(f"finetune=model/best.pt, learning_rate={self.config['training']['learning_rate'] * 0.5:g}", printed)
            finetune.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Base checkpoint not found"):
                invoke("--finetune", str(self.directory / "absent.pt"))
            other = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            other["model"]["backbone"][0]["channels"] = 3
            other_path = self.directory / "other.yaml"
            other_path.write_text(yaml.safe_dump(other), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                invoke("--config", str(other_path))
            finetune.assert_not_called()

            invoke("--replay-ratio", "2", "--max-f1-drop", "0.02", "--name", "tuned", "--epochs", "2")
            train.assert_not_called()
        config, options = finetune.call_args.args
        self.assertEqual(config["training"]["epochs"], 2)
        # The rate is scaled by the run itself, so the saved config shows what was used.
        self.assertEqual(config["training"]["learning_rate"], self.config["training"]["learning_rate"])
        self.assertEqual(options, {"checkpoint": str(checkpoint.resolve()), "learning_rate_scale": 0.1,
                                   "replay_ratio": 2.0, "max_f1_drop": 0.02, "max_count_mae_rise": 0.01})
        self.assertEqual(finetune.call_args.kwargs, {"name": "tuned", "selection": None})
        self.assertFalse(tuned.exists())

        # A YAML with other hyperparameters than the base run: the base run wins, visibly.
        edited = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        edited["training"].update(learning_rate=0.5, batch_size=1, epochs=4)
        edited["loss"]["bbox_weight"] = 9.0
        edited["evaluation"]["confidence"] = 0.9
        edited_path = self.directory / "edited.yaml"
        edited_path.write_text(yaml.safe_dump(edited), encoding="utf-8")
        best = self.directory / "best.json"
        best.write_text(json.dumps({"schema_version": 1, "params": {"loss.bbox_weight": 3.0},
                                    "objective": {"metric": "f1", "direction": "maximize"}}), encoding="utf-8")
        with patch.object(self.cli, "finetune", return_value=tuned) as finetune:
            _, printed = invoke("--config", str(edited_path))
            inherited = finetune.call_args.args[0]
            _, same = invoke()
            invoke("--config", str(edited_path), "--best-params", str(best))
            explicit = finetune.call_args.args[0]
        self.assertIn("Hyperparameters of the base checkpoint replace those of the YAML:", printed)
        self.assertIn(f"  training.learning_rate: 0.5 -> {self.config['training']['learning_rate']!r}", printed)
        self.assertIn(f"  loss.bbox_weight: 9.0 -> {self.config['loss']['bbox_weight']!r}", printed)
        self.assertIn("Hyperparameters of the base checkpoint are those of the YAML.", same)
        for section in ("loss", "evaluation"):
            self.assertEqual(inherited[section], self.config[section])
        self.assertEqual(inherited["training"]["learning_rate"], self.config["training"]["learning_rate"])
        self.assertEqual(inherited["training"]["batch_size"], self.config["training"]["batch_size"])
        # The length of the run is the YAML's; an explicit parameter file still applies on top.
        self.assertEqual(inherited["training"]["epochs"], 4)
        self.assertEqual(explicit["loss"]["bbox_weight"], 3.0)
        self.assertEqual(explicit["evaluation"], self.config["evaluation"])

        directory, _ = invoke()
        self.assertEqual(directory, tuned)
        report = json.loads((directory / "finetune.json").read_text(encoding="utf-8"))
        self.assertEqual(report["base"]["checkpoint"], str(checkpoint.resolve()))
        self.assertIn("result", report)
        self.assertEqual(json.loads((directory / "summary.json").read_text())["status"], "completed")

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "optional Optuna not installed")
    def test_real_optuna_search_and_resume_add_trials_to_one_study(self):
        search = self.search_path()
        self.invoke("--optuna", str(search))
        with self.assertRaisesRegex(ValueError, "use --resume"):
            self.invoke("--optuna", str(search))
        self.invoke("--optuna", str(search), "--resume", "--trials", "1")
        import optuna
        from sqlalchemy.pool import NullPool
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{(self.output / 'study.db').as_posix()}",
            engine_kwargs={"poolclass": NullPool},
        )
        try:
            study = optuna.load_study(study_name="geometry", storage=storage)
            self.assertEqual(len(study.trials), 2)
            self.assertTrue(all(trial.state.name == "COMPLETE" for trial in study.trials))
            self.assertTrue((self.output / "best_params.json").is_file())
            self.assertTrue((self.output / "trials" / "trial_00001" / "best.pt").is_file())
        finally:
            storage.remove_session()
            storage.engine.dispose()


if __name__ == "__main__":
    unittest.main()
