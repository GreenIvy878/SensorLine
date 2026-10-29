"""Files produced by one training run."""

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


REPLACE_ATTEMPTS = 100
REPLACE_DELAY_SECONDS = 0.02


def replace_file(source: str | Path, target: str | Path) -> None:
    """os.replace that waits out a reader briefly holding the target open.

    Windows refuses to replace a file that another handle has open, and the
    dashboard reads run files while training rewrites them.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAY_SECONDS)


def write_text(path: str | Path, text: str, newline: str | None = None) -> None:
    """Replace a text file in one step: a reader sees the old file or the new one, never a part.

    `newline` as in open(): None writes the platform's line ends, "\\n" keeps them as given.
    """
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline=newline)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value, indent=2) -> None:
    write_text(path, json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False))


def read_json(path: str | Path, default=None):
    """The content of a JSON file, or `default` when the file does not exist."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# summary.json records the state of a run while it trains; the dashboard lists runs by it.
def start_experiment(directory, config, name=None):
    summary = {
        "id": uuid4().hex,
        "name": name or Path(directory).name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "seed": config["training"]["seed"],
        "best_metrics": {},
        "final_metrics": {},
        "best_epoch": None,
        "training_seconds": None,
        "parameters": None,
        "checkpoint": "best.pt",
        "provenance": None,
    }
    write_json(Path(directory) / "config.json", config)
    write_json(Path(directory) / "summary.json", summary)
    return summary


def update_experiment(directory, **changes):
    path = Path(directory) / "summary.json"
    summary = read_json(path, {})
    summary.update(changes)
    write_json(path, summary)
    return summary


class RunArtifacts:
    """Own the run directory and keep report writing out of the epoch loop."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.directory = self.directory / timestamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.directory / "best.pt"

    def save_metadata(self, config, model_info, data_audit, environment) -> None:
        write_json(self.directory / "config.json", config)
        write_json(self.directory / "model_info.json", model_info)
        write_json(self.directory / "data_audit.json", data_audit)
        write_json(self.directory / "environment.json", environment)

    def save_history(self, history: list[dict]) -> None:
        write_rows(self.directory / "history.csv", history)
        write_json(self.directory / "history.json", history)

    def save_validation(self, report, sample_rows, predictions) -> None:
        write_json(self.directory / "metrics.json", report)
        write_rows(self.directory / "validation_samples.csv", sample_rows)
        write_json(self.directory / "validation_predictions.json", predictions)
