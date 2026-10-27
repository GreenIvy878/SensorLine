"""The model YAMLs and trained runs that the dashboard sections work with."""

from pathlib import Path

import torch

from ..config import DEVICES, load_config
from .experiments.records import architecture_text, discover_experiments


def model_description(model: dict) -> dict:
    """How the dashboard lists an architecture: backbone block types, neck and head."""
    return {
        "architecture": architecture_text(model),
        "neck": model.get("neck", {}).get("type", "none"),
        "head": model.get("head", {}).get("type", "bbox"),
    }


def requested_device(body: dict) -> str:
    device = body.get("device", "auto")
    if device not in DEVICES:
        raise ValueError("device must be auto, cuda or cpu")
    return device


def device_description() -> dict:
    cuda = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name(0) if cuda else "CPU", "cuda": cuda}


def checkpoint_row(run: dict) -> dict:
    """How a section lists a trained run to choose from."""
    metrics = run.get("best_metrics") or {}
    return {"id": run["id"], "name": run["name"], "directory": run["directory"],
            "timestamp": run.get("timestamp"), "status": run.get("status"),
            "f1": metrics.get("f1"), "count_mae": metrics.get("count_mae")}


class Workspace:
    """Model YAMLs next to config_path and the runs with a checkpoint under runs_root."""

    def __init__(self, config_path, runs_root):
        self.config_path = Path(config_path).resolve() if config_path else None
        self.runs_root = Path(runs_root).resolve()

    def config_files(self) -> dict:
        if self.config_path is None:
            return {}
        directory = self.config_path.parent
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".yaml", ".yml"))
        return {path.stem: path for path in paths}

    def valid_configs(self) -> dict:
        """Loaded model YAMLs by name; one that cannot be read is left out."""
        loaded = {}
        for name, path in self.config_files().items():
            try:
                loaded[name] = load_config(path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return loaded

    def trained_runs(self) -> list:
        """Experiment records of the runs that saved a checkpoint, newest first."""
        runs = [row for row in discover_experiments(self.runs_root)
                if row.get("id") and (Path(row["directory"]) / "best.pt").is_file()]
        runs.sort(key=lambda row: row.get("timestamp") or "", reverse=True)
        return runs

    def checkpoints(self) -> list:
        return [checkpoint_row(run) for run in self.trained_runs()]

    def checkpoint_path(self, experiment_id) -> Path:
        directory = next((row["directory"] for row in self.trained_runs()
                          if row["id"] == experiment_id), None)
        if directory is None:
            raise ValueError("Choose a checkpoint from the runs folder")
        return Path(directory) / "best.pt"
