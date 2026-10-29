"""Saved training runs as the dashboard lists them; the runs write these files (see artifacts)."""

from pathlib import Path

from ...artifacts import read_json


def architecture_text(model: dict) -> str:
    return " → ".join(block["type"] for block in model["backbone"])


def load_experiment(directory):
    directory = Path(directory).resolve()
    config = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    if config is None or summary is None:
        raise ValueError("Experiment has no config.json or summary.json")
    return {
        **summary,
        "directory": str(directory),
        "config": config,
        "history": read_json(directory / "history.json", []),
        "model_info": read_json(directory / "model_info.json", {}),
        "dataset": read_json(directory / "data_audit.json", {}),
        "architecture": architecture_text(config["model"]),
    }


# A run folder of the tracker (track.py) has a summary.json too, but it is no training run.
TRACKING_RUN_MARKER = "config.resolved.yaml"


def discover_experiments(root):
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("summary.json")):
        if (path.parent / TRACKING_RUN_MARKER).is_file():
            continue
        try:
            rows.append(load_experiment(path.parent))
        except (ValueError, OSError, KeyError, TypeError) as error:
            rows.append(
                {
                    "id": None,
                    "directory": str(path.parent),
                    "status": "failed",
                    "error": str(error),
                }
            )
    return rows

