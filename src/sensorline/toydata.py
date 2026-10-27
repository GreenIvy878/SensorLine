"""A small synthetic dataset in the format of the real one, so the project runs without private data.

Each frame is a grayscale PNG: the sensors (40 by default) along x, 40 time steps along y.
Objects are Gaussian blobs about 4 sensors wide and 7 time steps long, on a dark
background with per-sensor offsets and noise. Labels are YOLO boxes of class 0,
and a data.yaml at the root describes the dataset (see config.load_data_description).
"""

from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .config import DEFAULT_SENSORS

TIME = 40
# Objects per frame: mostly a few, sometimes a crowd, as in the real recordings.
COUNT_WEIGHTS = np.array([0, 18, 22, 20, 16, 11, 5, 3, 2, 1, 1, 1], dtype=float)
MAX_IOU = 0.2
# Object length in time steps and peak brightness (of 1.0) are drawn from these ranges;
# other ranges give a dataset with another distribution, for example to fine-tune on.
DEFAULT_LENGTHS = (4.0, 10.5)
DEFAULT_BRIGHTNESS = (0.55, 1.0)


def box_iou(box, boxes) -> np.ndarray:
    start = np.maximum(box[:2], boxes[:, :2])
    end = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(end - start, 0, None).prod(1)
    union = np.prod(box[2:] - box[:2]) + np.prod(boxes[:, 2:] - boxes[:, :2], 1) - intersection
    return intersection / union


def object_boxes(rng, count, sensors, lengths=DEFAULT_LENGTHS) -> np.ndarray:
    """Up to `count` boxes (x1, y1, x2, y2 in pixels) that overlap at most MAX_IOU."""
    boxes = np.empty((0, 4))
    for _ in range(count * 20):
        if len(boxes) == count:
            break
        width, length = rng.uniform(3.2, 5.0), rng.uniform(*lengths)
        x = rng.uniform(width / 2, sensors - width / 2)
        y = rng.uniform(length / 2, TIME - length / 2)
        box = np.array([x - width / 2, y - length / 2, x + width / 2, y + length / 2])
        if len(boxes) and box_iou(box, boxes).max() > MAX_IOU:
            continue
        boxes = np.vstack([boxes, box])
    return boxes


def render_frame(rng, boxes, sensors, brightness=DEFAULT_BRIGHTNESS) -> np.ndarray:
    """8-bit intensities: blobs inside the boxes over a noisy, striped background."""
    positions = np.arange(sensors) + 0.5
    time = np.arange(TIME)[:, None] + 0.5
    signal = rng.uniform(0, 0.03, sensors)[None, :] + rng.normal(0, 0.01, (TIME, sensors))
    for x1, y1, x2, y2 in boxes:
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        # The box spans about two standard deviations on each side of the centre.
        sigma_x, sigma_y = (x2 - x1) / 4, (y2 - y1) / 4
        blob = np.exp(-0.5 * (((positions - center_x) / sigma_x) ** 2 + ((time - center_y) / sigma_y) ** 2))
        signal = np.maximum(signal, rng.uniform(*brightness) * blob)
    return np.clip(np.round(signal * 255), 0, 255).astype(np.uint8)


def yolo_lines(boxes, sensors) -> str:
    rows = []
    for x1, y1, x2, y2 in boxes:
        rows.append(f"0 {(x1 + x2) / 2 / sensors:.6f} {(y1 + y2) / 2 / TIME:.6f} "
                    f"{(x2 - x1) / sensors:.6f} {(y2 - y1) / TIME:.6f}")
    return "\n".join(rows) + ("\n" if rows else "")


def write_split(root: Path, frames: int, rng, sensors, lengths, brightness) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    probabilities = COUNT_WEIGHTS / COUNT_WEIGHTS.sum()
    for index in range(frames):
        boxes = object_boxes(rng, int(rng.choice(len(probabilities), p=probabilities)), sensors, lengths)
        image = render_frame(rng, boxes, sensors, brightness)
        Image.fromarray(image, mode="L").save(root / "images" / f"toy_{index:04d}.png")
        (root / "labels" / f"toy_{index:04d}.txt").write_text(yolo_lines(boxes, sensors), encoding="utf-8")


def make_toy_dataset(root, train=300, val=80, seed=0, sensors=DEFAULT_SENSORS,
                     lengths=DEFAULT_LENGTHS, brightness=DEFAULT_BRIGHTNESS) -> Path:
    """Write train/ and val/ (images/ and labels/) and data.yaml under a new or empty `root`."""
    if sensors < 8:
        raise ValueError("A toy frame needs at least 8 sensors")
    if not 1 <= lengths[0] <= lengths[1] <= TIME:
        raise ValueError(f"Object lengths must satisfy 1 <= low <= high <= {TIME} time steps")
    if not 0 < brightness[0] <= brightness[1] <= 1:
        raise ValueError("Object brightness must satisfy 0 < low <= high <= 1")
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty")
    rng = np.random.default_rng(seed)
    write_split(root / "train", train, rng, sensors, lengths, brightness)
    write_split(root / "val", val, rng, sensors, lengths, brightness)
    description = {"data": {"type": "yolo_images", "train": "./train", "val": "./val",
                            "sensor_axis": "x", "sensors": sensors, "scale": 255.0, "offset": 0.0}}
    (root / "data.yaml").write_text(yaml.safe_dump(description, sort_keys=False), encoding="utf-8")
    return root
