"""Dashboard adapter of the Inference section: which checkpoint runs on which data."""

from pathlib import Path

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
        folder = None
        if source == "folder":
            folder = Path(str(body.get("folder") or "").strip())
            if not str(folder) or not (folder / "images").is_dir():
                raise ValueError("Folder must contain an images/ directory")
        device = requested_device(body)
        self.runner.start({
            "checkpoint": str(checkpoint),
            "checkpoint_id": body["checkpoint"],
            "source": source,
            "folder": str(folder) if folder else None,
            # The splits of a run are always labeled; a folder may be new, unlabeled data.
            "labeled": bool(body.get("labeled", True)) if source == "folder" else True,
            "device": device,
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
