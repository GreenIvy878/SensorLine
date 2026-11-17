"""Dashboard adapter of the Inference section: which checkpoint runs on which data."""

from pathlib import Path

from ...config import validate_border_ignore
from ..workspace import requested_device
from .runner import SOURCES, InferenceRunner


class InferenceViews:
    def __init__(self, workspace, output):
        self.workspace = workspace
        self.runner = InferenceRunner(Path(output) / "inference")

    def start(self, body: dict) -> dict:
        source = body.get("source")
        if source not in SOURCES:
            raise ValueError("Choose the data to run on")
        checkpoint = self.workspace.checkpoint_path(body.get("checkpoint"))
        folder, origin = None, "run"
        if source == "folder":
            folder = Path(str(body.get("folder") or "").strip())
            if not str(folder) or not (folder / "images").is_dir():
                raise ValueError("Folder must contain an images/ directory")
            origin = "folder"
        elif source == "test":
            place = self.workspace.test_folder(self.workspace.run_record(body["checkpoint"]))
            if place is None:
                raise ValueError("This run has no test split, and no model YAML next to --config names a test "
                                 "folder with the same signal; run on a folder instead")
            folder, origin = place
            if not (folder / "images").is_dir():
                raise ValueError(f"The test folder {folder} has no images/ directory")
            # A test folder written in the run's own config is read exactly like its validation split.
            if origin == "run":
                folder = None
        device = requested_device(body)
        # Empty keeps the checkpoint's own setting; a number replaces it.
        border_ignore = body.get("border_ignore")
        if border_ignore is not None:
            validate_border_ignore(border_ignore, "Cut-object length")
        self.runner.start({
            "checkpoint": str(checkpoint),
            "checkpoint_id": body["checkpoint"],
            "source": source,
            "folder": str(folder) if folder else None,
            "origin": origin,
            # The splits of a run are always labeled; a folder may be new, unlabeled data.
            "labeled": bool(body.get("labeled", True)) if source == "folder" else True,
            "device": device,
            "border_ignore": border_ignore,
        })
        return self.runner.status()

    def get(self, path: str, query: dict):
        if path == "/api/inference/checkpoints":
            return self.workspace.checkpoints()
        if path == "/api/inference/status":
            return {"inference": self.runner.status()}
        if path == "/api/inference/sample":
            return self.runner.sample(query.get("id", [""])[0])
        raise ValueError("Unknown inference endpoint")

    def post(self, path: str, body: dict):
        if path == "/api/inference/start":
            return {"inference": self.start(body)}
        raise ValueError("Unknown inference endpoint")
