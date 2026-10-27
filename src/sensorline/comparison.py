"""Paired bootstrap comparison of two evaluated models on the same frames.

Both models must have been evaluated on the same labeled frames. Frames are
resampled with replacement, the metric of each model is computed from the
summed TP/FP/FN (or count errors) of the same resample, and the percentiles
of the differences give the confidence interval. Pairing removes the
variation in frame difficulty; resampling frames rather than objects keeps
the errors of one crowded frame together.
"""

import csv
import json
from pathlib import Path

import numpy as np

# Per-frame files, in the order they are looked for: a training run, a test.py report.
SAMPLE_FILES = ("validation_samples.csv", "samples.csv")
COLUMNS = ("tp", "fp", "fn", "true_count", "predicted_count")
METRICS = ("f1", "precision", "recall", "count_mae")


def samples_path(directory: str | Path) -> Path:
    """The per-frame file of a run, of a test.py report, or of the best trial of an Optuna study."""
    directory = Path(directory)
    best = directory / "best_params.json"
    if best.is_file():
        saved = json.loads(best.read_text(encoding="utf-8"))
        # The study may have moved since it was written; its trials folder moved with it.
        trial = directory / "trials" / f"trial_{saved['trial_number']:05d}"
        directory = trial if trial.is_dir() else Path(saved["run_dir"])
    for name in SAMPLE_FILES:
        if (directory / name).is_file():
            return directory / name
    raise ValueError(f"{directory} has no {' or '.join(SAMPLE_FILES)}")


def read_samples(path: str | Path) -> dict:
    """Frame id -> TP, FP, FN, labeled and predicted count."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not set(COLUMNS) <= set(rows[0]):
        raise ValueError(f"{path}: needs labeled frames with {', '.join(COLUMNS)}")
    frames = {row["id"]: tuple(int(row[key]) for key in COLUMNS) for row in rows}
    if len(frames) != len(rows):
        raise ValueError(f"{path}: repeated frame ids")
    return frames


def paired_arrays(baseline: dict, other: dict) -> tuple[np.ndarray, np.ndarray]:
    """[frames, columns] arrays of both models in the same frame order."""
    if baseline.keys() != other.keys():
        raise ValueError("The models were evaluated on different frames")
    ids = sorted(baseline)
    first = np.array([baseline[frame] for frame in ids], dtype=np.int64)
    second = np.array([other[frame] for frame in ids], dtype=np.int64)
    if not np.array_equal(first[:, 3], second[:, 3]):
        raise ValueError("The models were evaluated with different labels on the same frames")
    return first, second


def metric_of_sums(sums: np.ndarray, frames: int, metric: str) -> np.ndarray:
    """The metric from [..., columns] sums over `frames` frames; count MAE needs per-frame errors."""
    tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
    if metric == "f1":
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    if metric == "precision":
        return tp / np.maximum(tp + fp, 1)
    if metric == "recall":
        return tp / np.maximum(tp + fn, 1)
    return sums[..., 5] / frames


def with_count_error(frames: np.ndarray) -> np.ndarray:
    """The columns plus the absolute count error of each frame, which count MAE sums."""
    return np.column_stack((frames, np.abs(frames[:, 4] - frames[:, 3])))


def paired_bootstrap(baseline: dict, other: dict, metric: str = "f1", resamples: int = 10000,
                     confidence: float = 0.95, seed: int = 0) -> dict:
    """Metric of both models, their difference (other - baseline) and its bootstrap interval."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {', '.join(METRICS)}")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence in (0, 1)")
    first, second = (with_count_error(frames) for frames in paired_arrays(baseline, other))
    count = len(first)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    # In chunks of about two million frame draws, so a large test set stays small in memory.
    chunk = max(1, 2_000_000 // count)
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, count, size=(stop - start, count))
        differences[start:stop] = (metric_of_sums(second[indices].sum(axis=1), count, metric)
                                   - metric_of_sums(first[indices].sum(axis=1), count, metric))
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    base_value = float(metric_of_sums(first.sum(axis=0), count, metric))
    other_value = float(metric_of_sums(second.sum(axis=0), count, metric))
    return {
        "metric": metric,
        "frames": count,
        "baseline": base_value,
        "other": other_value,
        "difference": other_value - base_value,
        "interval": [float(low), float(high)],
        "confidence": confidence,
        "resamples": resamples,
        # Zero outside the interval: the difference is unlikely to be resampling noise.
        "distinguishable": bool(low > 0 or high < 0),
        "errors": [int(first[:, 1:3].sum()), int(second[:, 1:3].sum())],
    }
