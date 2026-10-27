"""Dashboard adapter of the Model section.

A model is a YAML of the configs folder (its architecture, untrained weights) or a
trained checkpoint of the runs folder. Size and cost are computed on request and
take well under a second. Speed is measured only when asked, in a worker thread,
on the validation frames of the model's own data.
"""

import platform
import threading
import traceback
from uuid import uuid4

import torch

from ...checkpoint import load_checkpoint, load_model
from ...config import sensor_count
from ...data import build_dataset
from ...evaluation import evaluation_settings
from ...models import Detector
from ...runtime import choose_device
from ..experiments.experiment_analysis import object_extents
from ..experiments.model_inspector import parameter_count
from ..workspace import device_description, model_description, requested_device
from .complexity import model_profile
from .timing import inference_timing, training_timing

SOURCES = ("config", "checkpoint")
MAX_TIME_STEPS = 4096


def integer(value, name, minimum, maximum) -> int:
    """A whole number from a query string or a JSON body."""
    try:
        number = int(str(value).strip())
    except ValueError:
        number = None
    if number is None or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return number


class ModelViews:
    def __init__(self, workspace):
        self.workspace = workspace
        self.lock = threading.Lock()
        self.extents = {}
        self.state = None

    def options(self) -> dict:
        configs = [{"name": name, **model_description(config["model"])}
                   for name, config in self.workspace.valid_configs().items()]
        checkpoints = [{key: row[key] for key in ("id", "name", "timestamp", "f1", "count_mae")}
                       for row in self.workspace.checkpoints()]
        return {
            "configs": configs,
            "checkpoints": checkpoints,
            "default_config": self.workspace.config_path.stem if self.workspace.config_path else None,
            **device_description(),
        }

    def load(self, source, name, device="cpu") -> dict:
        """The model, its config and where both come from."""
        if source == "config":
            files = self.workspace.config_files()
            configs = self.workspace.valid_configs()
            if name not in configs:
                raise ValueError("Choose a valid model configuration")
            config = configs[name]
            model = Detector(config["model"]).to(device)
            return {"model": model, "config": config, "trained": False, "label": f"{files[name].name} (untrained weights)"}
        if source == "checkpoint":
            path = self.workspace.checkpoint_path(name)
            checkpoint = load_checkpoint(path, device)
            model = load_model(checkpoint, device)
            return {"model": model, "config": checkpoint["config"], "trained": True,
                    "label": f"{path.parent.name}/best.pt (epoch {checkpoint['epoch']})"}
        raise ValueError("Choose a model YAML or a trained checkpoint")

    def object_extents(self, data) -> tuple[dict | None, str | None]:
        """Annotated object sizes of the validation data, read once per folder."""
        key = (data["val"], data["sensor_axis"], data["scale"], data["offset"], sensor_count(data))
        with self.lock:
            if key in self.extents:
                return self.extents[key]
        try:
            dataset = build_dataset(data, "val")
            frames = ((dataset.boxes[index].numpy(), tuple(dataset[index]["signal"].shape[-2:]))
                      for index in range(len(dataset)))
            result = object_extents(frames), None
        except (OSError, ValueError) as error:
            result = None, f"Validation data cannot be read: {error}"
        with self.lock:
            self.extents[key] = result
        return result

    def summary(self, query) -> dict:
        source = query.get("source", [""])[0]
        name = query.get("name", [""])[0]
        time_steps = integer(query.get("time", ["40"])[0], "Signal length", 1, MAX_TIME_STEPS)
        loaded = self.load(source, name)
        extents, extents_error = self.object_extents(loaded["config"]["data"])
        profile = model_profile(loaded["model"], sensor_count(loaded["config"]["data"]), time_steps, extents)
        return {**profile, "source": source, "name": name, "label": loaded["label"],
                "trained": loaded["trained"], "extents_error": extents_error,
                "evaluation": loaded["config"]["evaluation"],
                "data_val": loaded["config"]["data"]["val"],
                "batch_size": loaded["config"]["training"]["batch_size"]}

    def start(self, body) -> dict:
        source, name = body.get("source"), body.get("name")
        if source not in SOURCES:
            raise ValueError("Choose a model YAML or a trained checkpoint")
        device = requested_device(body)
        batch_size = integer(body.get("batch_size"), "Batch size", 1, 1024)
        # A wrong model name is reported now, not as a failed measurement a moment later.
        if source == "config" and name not in self.workspace.config_files():
            raise ValueError("Choose a valid model configuration")
        if source == "checkpoint":
            self.workspace.checkpoint_path(name)
        with self.lock:
            if self.state and self.state["status"] == "running":
                raise ValueError("A measurement is already running")
            self.state = {"id": uuid4().hex[:8], "status": "running", "source": source, "name": name,
                          "device": device, "batch_size": batch_size, "stage": "Loading the model",
                          "done": 0, "total": 0, "error": None, "result": None}
            state = self.state
        threading.Thread(target=self.run, args=(state,), daemon=True).start()
        return self.status()

    def run(self, state) -> None:
        def report(stage, done, total):
            with self.lock:
                state.update(stage=stage, done=done, total=total)

        try:
            device = choose_device(state["device"])
            loaded = self.load(state["source"], state["name"], device)
            config = loaded["config"]
            report("Reading the validation frames", 0, 0)
            dataset = build_dataset(config["data"], "val")
            samples = [dataset[index] for index in range(len(dataset))]
            evaluation = evaluation_settings(config["evaluation"], config["data"])
            inference = inference_timing(loaded["model"], samples, device, evaluation,
                                         state["batch_size"], report)
            training = training_timing(loaded["model"], samples, device, config["loss"],
                                       state["batch_size"], report)
            try:
                train_frames = len(build_dataset(config["data"], "train"))
            except (OSError, ValueError):
                train_frames = None
            result = {
                **inference, **training,
                "label": loaded["label"], "trained": loaded["trained"],
                "frames": len(samples), "data_val": config["data"]["val"],
                "frame_shape_sensor_time": list(samples[0]["signal"].shape[-2:]),
                "time_steps_per_frame": float(sum(s["signal"].shape[-1] for s in samples) / len(samples)),
                "train_frames": train_frames,
                "parameters": parameter_count(loaded["model"]),
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU, {torch.get_num_threads()} threads",
                "torch": torch.__version__, "python": platform.python_version(),
                "evaluation": config["evaluation"],
            }
            with self.lock:
                state.update(status="completed", result=result)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            with self.lock:
                state.update(status="failed", error=str(error))
        except Exception as error:
            traceback.print_exc()
            with self.lock:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")

    def status(self) -> dict | None:
        with self.lock:
            return None if self.state is None else dict(self.state)

    def get(self, path, query):
        if path == "/api/model/options":
            return self.options()
        if path == "/api/model/summary":
            return self.summary(query)
        if path == "/api/model/benchmark":
            return {"benchmark": self.status()}
        raise ValueError("Unknown model endpoint")

    def post(self, path, body):
        if path == "/api/model/benchmark":
            return {"benchmark": self.start(body)}
        raise ValueError("Unknown model endpoint")
