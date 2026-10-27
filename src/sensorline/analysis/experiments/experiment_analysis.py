"""Model summary of a finished run; the dashboard only reads these files."""

from pathlib import Path

import numpy as np

from ...artifacts import read_json, write_json
from ...models import Detector
from ..dataset.numerics import bbox_geometry
from .model_inspector import inspect_model
from .records import load_experiment


def artifact_signature(directory):
    result = {}
    for name in ("config.json", "metrics.json", "validation_predictions.json", "best.pt"):
        path = Path(directory) / name
        if path.is_file():
            stat = path.stat()
            result[name] = [stat.st_size, stat.st_mtime_ns]
    return result


def extent_quantiles(extents: dict) -> dict:
    """Median, p90 and p95 of each list of object extents; empty for a list without objects."""
    return {
        key: {str(q): float(np.percentile(values, q)) for q in (50, 90, 95)} if values else {}
        for key, values in extents.items()
    }


def object_extents(frames) -> dict:
    """Quantiles of annotated object width (sensors) and length (time steps), in original cells.

    `frames` yields (boxes, size): the normalized signal boxes and the [sensors, time] size of a frame.
    """
    extents = {"sensor_width": [], "temporal_length": []}
    for boxes, size in frames:
        geometry = bbox_geometry(boxes, size)
        for key, values in extents.items():
            values.extend(geometry[key])
    return extent_quantiles(extents)


def save_analysis(experiment, predictions, model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    extents = object_extents((row["target_boxes_signal_xyxy"], row["size_sensor_time"]) for row in predictions)
    inspection = inspect_model(model, (1, 1, *predictions[0]["size_sensor_time"]), extents)
    summary = {
        "schema_version": 2,
        "artifact_signature": artifact_signature(experiment["directory"]),
        "experiment": {
            key: experiment.get(key)
            for key in ("id", "name", "timestamp", "status", "seed", "best_epoch",
                        "training_seconds", "provenance", "directory", "checkpoint", "parameters")
        },
        "metrics": {"best": experiment["best_metrics"], "final": experiment["final_metrics"]},
        "architecture": experiment["config"]["model"],
        "dataset": {"inventory": experiment["dataset"], "validation_object_extents": extents},
        "model_inspection": inspection,
        "config": experiment["config"],
    }
    write_json(output / "model_summary.json", inspection)
    write_json(output / "experiment_analysis.json", summary)
    return summary


def prepare_experiment(directory, output=None):
    experiment = load_experiment(directory)
    if experiment["status"] != "completed":
        raise ValueError("Select a completed experiment")
    predictions = read_json(Path(directory) / "validation_predictions.json")
    if predictions is None:
        raise ValueError("No saved validation predictions; evaluate the checkpoint explicitly first")
    # Shapes, parameter counts and receptive fields depend on the architecture, not the weights.
    model = Detector(experiment["config"]["model"])
    return save_analysis(experiment, predictions, model, output or directory)
