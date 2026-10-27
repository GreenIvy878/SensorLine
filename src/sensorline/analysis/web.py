"""Local dashboard server. Computation runs once in a worker thread, not in views."""

import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import sensor_count
from ..data import image_boxes, image_to_signal, read_yolo_boxes, validate_image
from ..local_page import LocalHandler, LocalServer
from .experiments.experiment_web import ExperimentViews
from .inference.views import InferenceViews
from .model.views import ModelViews
from .dataset.runner import load_or_run
from .workspace import Workspace


STATIC = Path(__file__).parent / "static"
EXPERIMENT_PATHS = {"/api/experiments", "/api/experiment", "/api/experiment-analysis", "/api/experiment-model"}
# GET responses offered as files to save, with their file names.
DOWNLOADS = {"/analysis.json": "analysis.json", "/api/experiment-analysis": "experiment_analysis.json"}
REQUEST_ERRORS = (ValueError, OSError, RuntimeError, KeyError, TypeError)


class AnalysisState:
    def __init__(self, data, output, seed, runs_root="runs", config_path=None):
        self.data = data
        self.output = Path(output)
        self.seed = seed
        self.report = None
        self.running = False
        self.message = "Ready to run analysis"
        self.error = None
        self.lock = threading.Lock()
        self.experiments = ExperimentViews(runs_root, self.output / "experiments")
        # The model YAMLs next to config_path and the trained runs; without config_path
        # the Model section has no YAMLs, but checkpoints can still be run and inspected.
        self.workspace = Workspace(config_path, runs_root)
        self.inference = InferenceViews(self.workspace, self.output)
        self.model = ModelViews(self.workspace)

    def start(self, refresh=False):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.error = None
        threading.Thread(target=self._run, args=(refresh,), daemon=True).start()
        return True

    def _run(self, refresh):
        def progress(message):
            with self.lock:
                self.message = message

        try:
            report = load_or_run(self.data, self.output, self.seed, refresh, progress)
            with self.lock:
                self.report = report
                self.message = "Analysis ready"
        except (OSError, ValueError) as error:
            with self.lock:
                self.error = str(error)
                self.message = "Analysis failed; previous result retained"
        finally:
            with self.lock:
                self.running = False

    def status(self):
        with self.lock:
            return {
                "running": self.running,
                "message": self.message,
                "error": self.error,
                "available": self.report is not None,
                "generated_at": self.report["generated_at"] if self.report else None,
                "signature": self.report["source_signature"] if self.report else None,
            }

    def sample(self, split, identifier):
        report = self.report
        if not report or split not in ("train", "val"):
            raise ValueError("Run analysis and select train or val")
        records = report["splits"][split]["samples"]
        if identifier not in {row["id"] for row in records}:
            raise ValueError("Sample is not in the analyzed population")
        path = Path(self.data[split]) / "images" / identifier
        validate_image(path, self.data["sensor_axis"], sensor_count(self.data))
        matrix = image_to_signal(path, self.data["sensor_axis"], 1, 0)[0].numpy()
        boxes = read_yolo_boxes(
            Path(self.data[split]) / "labels" / (path.stem + ".txt"),
            self.data["sensor_axis"],
        ).numpy()
        display = matrix.T if self.data["sensor_axis"] == "x" else matrix
        display_boxes = image_boxes(boxes, self.data["sensor_axis"])
        return {
            "id": identifier,
            "split": split,
            "matrix": display.tolist(),
            "sensor_axis": self.data["sensor_axis"],
            "boxes_image_xyxy": display_boxes.tolist(),
            "min": float(matrix.min()),
            "max": float(matrix.max()),
            "profiles": {
                "sensor_mean": matrix.mean(axis=1).tolist(),
                "sensor_max": matrix.max(axis=1).tolist(),
                "time_mean": matrix.mean(axis=0).tolist(),
                "time_max": matrix.max(axis=0).tolist(),
            },
        }


def make_server(state, port: int):
    def api_views(path):
        """The views object that answers an API path, or None; it dispatches on the full path."""
        if path.startswith("/api/inference/"):
            return state.inference
        if path.startswith("/api/model/"):
            return state.model
        if path in EXPERIMENT_PATHS:
            return state.experiments
        return None

    class Handler(LocalHandler):
        static = STATIC
        index = "dashboard.html"
        body_limit = 512 * 1024

        def do_GET(self):
            if self.refuse_foreign_request():
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                views = api_views(url.path)
                if views is not None:
                    self.send_json(views.get(url.path, query), download=DOWNLOADS.get(url.path))
                elif url.path == "/api/status":
                    self.send_json(state.status())
                elif url.path in ("/api/analysis", "/analysis.json"):
                    report = state.report
                    if report is None:
                        self.send_json({"error": "Analysis not ready"}, 409)
                    else:
                        self.send_json(report, download=DOWNLOADS.get(url.path))
                elif url.path == "/api/sample":
                    self.send_json(state.sample(query.get("split", [""])[0], query.get("id", [""])[0]))
                elif not self.send_page_file(url.path):
                    self.send_json({"error": "Not found"}, 404)
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

        def do_POST(self):
            if self.refuse_foreign_request():
                return
            url = urlparse(self.path)
            if url.path in ("/api/run", "/api/refresh"):
                started = state.start(refresh=url.path == "/api/refresh")
                self.send_json({"started": started}, 202 if started else 409)
                return
            try:
                if url.path == "/api/experiment-prepare":
                    self.send_json(state.experiments.post(url.path, parse_qs(url.query)))
                    return
                views = api_views(url.path)
                if views is None or views is state.experiments:
                    self.send_json({"error": "Not found"}, 404)
                    return
                self.send_json(views.post(url.path, self.json_body()))
            except REQUEST_ERRORS as error:
                self.send_json({"error": str(error)}, 400)

    return LocalServer(("127.0.0.1", port), Handler)
