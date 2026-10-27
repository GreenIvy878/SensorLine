"""Small dashboard adapter for the run list and the prepared run files."""

import threading
from pathlib import Path

from ...artifacts import read_json
from .experiment_analysis import artifact_signature, prepare_experiment
from .records import discover_experiments


class ExperimentViews:
    def __init__(self, runs_root, cache):
        self.runs_root = Path(runs_root).resolve()
        self.cache = Path(cache)
        self.lock = threading.Lock()

    def resolve(self, identifier):
        for row in discover_experiments(self.runs_root):
            if row.get("id") == identifier and identifier:
                return row
        raise ValueError("Experiment is not in the configured runs root")

    def analysis_directory(self, experiment):
        for directory in (Path(experiment["directory"]), self.cache / experiment["id"]):
            summary = read_json(directory / "experiment_analysis.json")
            if summary and summary.get("artifact_signature") == artifact_signature(
                experiment["directory"]
            ):
                return directory
        return None

    def prepared(self, experiment, filename):
        directory = self.analysis_directory(experiment)
        if directory is None:
            raise ValueError("Prepare analysis once for this experiment")
        return read_json(directory / filename)

    def detail(self, identifier):
        experiment = self.resolve(identifier)
        experiment["analysis_ready"] = self.analysis_directory(experiment) is not None
        return experiment

    def prepare(self, identifier):
        with self.lock:
            experiment = self.resolve(identifier)
            directory = self.analysis_directory(experiment)
            if directory:
                return read_json(directory / "experiment_analysis.json")
            return prepare_experiment(experiment["directory"], self.cache / identifier)

    def get(self, path, query):
        identifier = query.get("id", [""])[0]
        if path == "/api/experiments":
            return discover_experiments(self.runs_root)
        if path == "/api/experiment":
            return self.detail(identifier)
        files = {
            "/api/experiment-analysis": "experiment_analysis.json",
            "/api/experiment-model": "model_summary.json",
        }
        return self.prepared(self.resolve(identifier), files[path])

    def post(self, path, query):
        return self.prepare(query.get("id", [""])[0])
