"""Read existing YOLO data, cache statistics, and preserve source provenance."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from ...artifacts import write_json
from ...config import sensor_count
from ...data import find_images, image_to_signal, orphan_labels, read_yolo_boxes, validate_image
from .aggregation import SplitAnalysis, dataset_drift


SCHEMA_VERSION = 2
# Definitions stored with every report so the numbers can be interpreted later.
METHOD_NOTES = {
    "units": "stored image pixel values; physical distance cannot be recovered",
    "analysis_axes": "matrix [sensor, time]; boxes normalized x=time, y=sensor",
    "viewer_axes": "original image orientation from data.sensor_axis",
    "quantiles": "uniform priority reservoirs; signal/FG/BG <=65536 pixels, per-sensor <=4096, other distributions <=4096",
    "correlation": "pooled Pearson across columns",
    "noise": "MAD of temporal second differences / (0.67448975 * sqrt(6)); includes signal edges",
    "foreground": "union of boxes, pixel-center membership; approximate foreground includes background",
}


def clean_json(value):
    """JSON has no NaN: undefined correlations and unavailable estimates are null."""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_signature(data, seed):
    digest = hashlib.sha256(json.dumps({"data": data, "seed": seed, "schema": SCHEMA_VERSION},
                                       sort_keys=True).encode())
    # Include analyzer implementation so numerical fixes invalidate cached results.
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    from ... import data as data_module
    digest.update(Path(data_module.__file__).read_bytes())
    for split in ("train", "val"):
        root = Path(data[split])
        for folder in ("images", "labels"):
            directory = root / folder
            digest.update(str(directory.resolve()).encode())
            if directory.is_dir():
                for path in sorted(directory.iterdir()):
                    if path.is_file():
                        digest.update(path.name.encode())
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(chunk)
    return digest.hexdigest()


def list_split(root, split, issues):
    """Image paths of one split; unreadable folders and orphan labels become issues."""
    try:
        paths = find_images(root / "images")
    except (ValueError, OSError) as error:
        issues.append({"split": split, "id": None, "reason": str(error)})
        paths = []
    for label in orphan_labels(root, paths):
        issues.append({"split": split, "id": label.name, "reason": "Annotation without image"})
    return paths


def read_sample(root, path, sensor_axis, sensors):
    """Same adapter/axis/finite checks as training; raw stored pixel units."""
    validate_image(path, sensor_axis, sensors)
    boxes = read_yolo_boxes(root / "labels" / f"{path.stem}.txt", sensor_axis).numpy()
    matrix = image_to_signal(path, sensor_axis, 1.0, 0.0)[0].numpy().astype(float)
    return matrix, boxes


def matrix_digest(matrix):
    shape = np.asarray(matrix.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape + matrix.tobytes()).hexdigest()


def analyze_dataset(data, seed=17, progress=None):
    sensors = sensor_count(data)
    aggregators = {scope: SplitAnalysis(seed + index, sensors)
                   for index, scope in enumerate(("train", "val", "all"))}
    issues, hashes, inventory = [], {}, {}
    for split in ("train", "val"):
        root = Path(data[split])
        paths = list_split(root, split, issues)
        inventory[split] = len(paths)
        for index, path in enumerate(paths):
            if progress:
                progress(f"{split}: {index + 1}/{len(paths)}")
            try:
                matrix, boxes = read_sample(root, path, data["sensor_axis"], sensors)
            except (ValueError, OSError) as error:
                issues.append({"split": split, "id": path.name, "reason": str(error)})
                continue
            aggregators[split].add(matrix, boxes, path.name, split)
            aggregators["all"].add(matrix, boxes, path.name, split)
            hashes.setdefault(matrix_digest(matrix), {"train": [], "val": []})[split].append(path.name)

    if progress:
        progress("Summarizing distributions")
    results = {scope: aggregator.result() for scope, aggregator in aggregators.items()}
    duplicates = [group for group in hashes.values() if group["train"] and group["val"]]
    duplicate_counts = {split: sum(max(0, len(group[split]) - 1) for group in hashes.values())
                        for split in ("train", "val")}
    return clean_json({
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {"data": data, "seed": seed, "inventory": inventory, **METHOD_NOTES},
        "splits": results, "drift": dataset_drift(aggregators["train"], aggregators["val"]),
        "issues": issues,
        "duplicates": {"within_split_counts": duplicate_counts, "cross_split": duplicates},
    })


def load_or_run(data, output, seed=17, refresh=False, progress=None):
    output = Path(output).resolve()
    for split in ("train", "val"):
        root = Path(data[split]).resolve()
        if output == root or root in output.parents:
            raise ValueError("Analysis output must be outside source dataset directories")
    if progress:
        progress("Checking source fingerprint")
    signature = source_signature(data, seed)
    path = output / "analysis.json"
    if path.is_file() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source_signature") == signature:
            return cached
    report = analyze_dataset(data, seed, progress)
    if signature != source_signature(data, seed):
        raise ValueError("Dataset changed during analysis; rerun on a stable source")
    report["source_signature"] = signature
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, report, indent=None)
    return report
