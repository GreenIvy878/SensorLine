"""Run a saved detector on a dataset in a worker thread and keep the result for the dashboard."""

from datetime import datetime
from pathlib import Path
import threading
import traceback
from uuid import uuid4

import numpy as np
import torch

from ...checkpoint import load_checkpoint
from ...data import image_box, image_to_signal
from ...evaluation import confidence_metrics
from ...inference import evaluate_checkpoint, save_report
from ...runtime import choose_device
from ..detections import below_threshold_detections, box_tensor, classify_sample


SOURCES = ("val", "train", "folder")
SWEEP_STEP = 0.05


def confidence_sweep(predictions: list, evaluation: dict) -> list:
    """Precision, recall and F1 of the saved post-NMS detections at other confidence thresholds."""
    detections = [
        {"boxes": box_tensor(row["all_boxes_signal_xyxy"]),
         "scores": torch.tensor(row["all_scores"], dtype=torch.float32)}
        for row in predictions
    ]
    targets = [
        {"id": row["id"],
         "boxes": box_tensor(row["target_boxes_signal_xyxy"]),
         "size": tuple(row["size_sensor_time"])}
        for row in predictions
    ]
    floor = evaluation["ap_min_confidence"]
    grid = {round(float(value), 2) for value in np.arange(SWEEP_STEP, 1.0, SWEEP_STEP) if value >= floor}
    grid.add(evaluation["confidence"])
    summaries = confidence_metrics(detections, targets, sorted(grid), evaluation["match_iou"],
                                   evaluation.get("border_ignore", 0))
    return [
        {"confidence": confidence,
         **{key: summary[key] for key in ("precision", "recall", "f1", "tp", "fp", "fn", "count_mae")}}
        for confidence, summary in summaries.items()
    ]


def run_inference(request: dict, output_root: Path, report) -> dict:
    """Evaluate one checkpoint with its own data, thresholds and loader settings."""
    device = choose_device(request["device"])
    checkpoint = load_checkpoint(request["checkpoint"], device)
    saved = checkpoint["config"]
    labeled = request["labeled"]
    result = evaluate_checkpoint(
        checkpoint, request["checkpoint"], device, saved["data"], request["source"],
        runtime=saved["training"], folder=request["folder"] if request["source"] == "folder" else None,
        labeled=labeled, progress=report, border_ignore=request.get("border_ignore"),
    )
    rows, predictions, summary = result["rows"], result["predictions"], result["report"]
    if labeled:
        frames = [{key: row[key] for key in ("id", "true_count", "predicted_count", "tp", "fp", "fn")}
                  for row in rows]
    else:
        frames = [{"id": row["id"], "predicted_count": row["count"]} for row in predictions]
    directory = output_root / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    save_report(directory, result)
    return {
        "labeled": labeled,
        "metrics": summary["metrics"],
        "frames": frames,
        "sweep": confidence_sweep(predictions, summary["evaluation"]) if labeled else [],
        "evaluation": summary["evaluation"],
        "preprocessing": summary["preprocessing"],
        "checkpoint_epoch": summary["checkpoint_epoch"],
        "data_root": summary["data_root"],
        "device": str(device),
        "output_directory": str(directory.resolve()),
        # Kept in memory only, for the frame viewer.
        "predictions": {row["id"]: row for row in predictions},
        "paths": {path.name: path for path in result["dataset"].paths},
        "sensor_axis": summary["preprocessing"]["sensor_axis"],
    }


class InferenceRunner:
    """Runs one inference at a time and serves its last result."""

    PRIVATE = ("predictions", "paths", "sensor_axis")

    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.lock = threading.Lock()
        self.state = None

    def start(self, request: dict) -> None:
        with self.lock:
            if self.state and self.state["status"] == "running":
                raise ValueError("Inference is already running")
            self.state = {"id": uuid4().hex[:8], "status": "running", "request": request,
                          "done": 0, "total": 0, "error": None}
            state = self.state
        threading.Thread(target=self._run, args=(state,), daemon=True).start()

    def _run(self, state: dict) -> None:
        def report(done, total=None):
            with self.lock:
                state["done"] = done
                if total is not None:
                    state["total"] = total

        try:
            result = run_inference(state["request"], self.output_root, report)
            with self.lock:
                state.update(result, status="completed")
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            with self.lock:
                state.update(status="failed", error=str(error))
        except Exception as error:
            # Not a known input problem (a damaged best.pt raises EOFError or IndexError),
            # but the run is over: left "running", it would block every later one.
            traceback.print_exc()
            with self.lock:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")

    def status(self) -> dict | None:
        with self.lock:
            if self.state is None:
                return None
            return {key: value for key, value in self.state.items() if key not in self.PRIVATE}

    def sample(self, identifier: str) -> dict:
        """Matrix and boxes of one frame in the original image orientation."""
        with self.lock:
            state = self.state
            if not state or state["status"] != "completed" or identifier not in state["predictions"]:
                raise ValueError("Frame is not in the current inference result")
            row, path = state["predictions"][identifier], state["paths"][identifier]
            evaluation, labeled = state["evaluation"], state["labeled"]
            axis = state["sensor_axis"]
        matrix = image_to_signal(path, axis, 1, 0)[0].numpy()

        if labeled:
            classified = classify_sample(row, evaluation)
            targets = [{"box": image_box(obj["box"], axis), "detected": obj["detected"],
                        "ignored": obj["ignored"]}
                       for obj in classified["objects"]]
            detections = classified["detections"]
        else:
            targets = []
            detections = [{"box": box, "score": score, "classification": "prediction"}
                          for box, score in zip(row["boxes_signal_xyxy"], row["scores"])]
        below = below_threshold_detections(row, evaluation)
        return {
            "id": identifier,
            "matrix": (matrix.T if axis == "x" else matrix).tolist(),
            "sensor_axis": axis,
            "labeled": labeled,
            "targets": targets,
            "detections": [{"box": image_box(d["box"], axis), "score": d["score"],
                            "classification": d["classification"]} for d in detections],
            "below_threshold": [{"box": image_box(d["box"], axis), "score": d["score"]}
                                for d in below],
        }
