"""Storage adapters and batching for signal [1, sensors, T].

Samples contain signal, normalized xyxy boxes, labels and id.
Internal box axes are x=time and y=sensor, regardless of the storage source.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_SENSORS, sensor_count


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
GRAYSCALE_MODES = {"L", "I", "I;16", "I;16L", "I;16B", "F"}


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images in {directory}")
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f"Ambiguous image stems in {directory}")
    return paths


def validate_image(path: Path, sensor_axis: str, sensors: int) -> None:
    with Image.open(path) as image:
        sensor_size = image.width if sensor_axis == "x" else image.height
        if sensor_size != sensors or image.mode not in GRAYSCALE_MODES:
            raise ValueError(f"{path}: expected grayscale with {sensors} sensors on {sensor_axis}")


def parse_yolo_box(line: str, sensor_axis: str) -> list[float]:
    values = np.asarray([float(value) for value in line.split()])
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("expected five finite numbers")
    class_id, center_x, center_y, width, height = values
    if class_id != 0 or min(width, height) <= 0:
        raise ValueError("expected class 0 (one object class) and positive width/height")

    box = np.asarray([
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    ])
    if box.min() < -1e-6 or box.max() > 1 + 1e-6:
        raise ValueError("box extends beyond image")
    box = np.clip(box, 0, 1)
    if not (box[2:] > box[:2]).all():
        raise ValueError("box has nonpositive width/height after clipping")
    if sensor_axis == "x":
        box = box[[1, 0, 3, 2]]
    box = box.astype(np.float32)
    if not (box[2:] > box[:2]).all():
        raise ValueError("box has nonpositive width/height in float32")
    return box.tolist()


def image_box(box: list, sensor_axis: str) -> list:
    """Internal xyxy box (x=time, y=sensor) in the axes of the stored image."""
    return [box[1], box[0], box[3], box[2]] if sensor_axis == "x" else box


def image_boxes(boxes, sensor_axis: str):
    """Like image_box, for an [N, 4] array or tensor."""
    return boxes[:, [1, 0, 3, 2]] if sensor_axis == "x" else boxes


def orphan_labels(root: Path, image_paths) -> list[Path]:
    """Label files of a split folder that belong to no image."""
    image_stems = {path.stem for path in image_paths}
    return sorted(path for path in (root / "labels").glob("*.txt") if path.stem not in image_stems)


def read_yolo_boxes(path: Path, sensor_axis: str) -> torch.Tensor:
    boxes = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_yolo_box(line, sensor_axis))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def pixels_to_signal(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    """Model-scale signal from float32 pixel values; training frames and CSV recordings share it."""
    return (values - offset) / scale


def image_to_signal(path: Path, sensor_axis: str, scale: float, offset: float):
    with Image.open(path) as image:
        values = np.array(image, dtype=np.float32)
    if sensor_axis == "x":
        values = values.T.copy()
    signal = torch.from_numpy(pixels_to_signal(values, scale, offset)).unsqueeze(0)
    if not torch.isfinite(signal).all():
        raise ValueError(f"Nonfinite signal in {path}")
    return signal


class YoloImageDataset(Dataset):
    """Convert grayscale images and YOLO annotations into common samples."""

    def __init__(
        self,
        root: str | Path,
        sensor_axis: str = "x",
        scale: float = 255.0,
        offset: float = 0.0,
        require_labels: bool = True,
        sensors: int = DEFAULT_SENSORS,
    ):
        self.root = Path(root)
        self.sensors = sensors
        self.sensor_axis = sensor_axis
        self.scale = float(scale)
        self.offset = float(offset)
        self.require_labels = require_labels

        self.paths = find_images(self.root / "images")
        self.boxes = self.load_annotations()

    def load_annotations(self) -> list[torch.Tensor]:
        boxes = []
        for path in self.paths:
            validate_image(path, self.sensor_axis, self.sensors)
            if not self.require_labels:
                boxes.append(torch.empty((0, 4), dtype=torch.float32))
                continue
            label_path = self.root / "labels" / f"{path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing YOLO annotation: {label_path}")
            boxes.append(read_yolo_boxes(label_path, self.sensor_axis))

        orphans = orphan_labels(self.root, self.paths) if self.require_labels else []
        if orphans:
            stems = sorted(path.stem for path in orphans)
            raise ValueError(f"Labels without images in {self.root}: {stems[:5]}")
        return boxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> dict:
        path = self.paths[index]
        signal = image_to_signal(path, self.sensor_axis, self.scale, self.offset)
        boxes = self.boxes[index].clone()
        return {
            "signal": signal,
            "boxes": boxes,
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "id": path.name,
        }


def build_dataset(data_config, split, root=None, require_labels=True):
    return YoloImageDataset(
        root=root or data_config[split],
        sensor_axis=data_config["sensor_axis"],
        scale=data_config["scale"],
        offset=data_config["offset"],
        require_labels=require_labels,
        sensors=sensor_count(data_config),
    )


def collate_samples(samples: list[dict]) -> tuple[torch.Tensor, list[dict]]:
    """Pad only time; boxes retain their sample's normalized coordinates."""
    sensors = samples[0]["signal"].shape[-2]
    max_time = max(sample["signal"].shape[-1] for sample in samples)
    batch = samples[0]["signal"].new_zeros((len(samples), 1, sensors, max_time))
    targets = []
    for index, sample in enumerate(samples):
        size = tuple(sample["signal"].shape[-2:])
        batch[index, :, :, :size[1]] = sample["signal"]
        targets.append({
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "size": size,
            "id": sample["id"],
        })
    return batch, targets


def dataset_signal_hashes(dataset):
    """Group identical signals and count objects using only the sample contract."""
    hashes = {}
    object_count = 0
    for sample in dataset:
        digest = hashlib.sha256()
        digest.update(np.asarray(sample["signal"].shape, dtype=np.int64).tobytes())
        digest.update(sample["signal"].contiguous().numpy().tobytes())
        hashes.setdefault(digest.hexdigest(), []).append(sample["id"])
        object_count += len(sample["boxes"])
    return hashes, object_count


def audit_sources(train, val) -> dict:
    train_hashes, train_objects = dataset_signal_hashes(train)
    val_hashes, val_objects = dataset_signal_hashes(val)
    duplicates = []
    for digest in sorted(train_hashes.keys() & val_hashes.keys()):
        duplicates.append({"train": train_hashes[digest], "val": val_hashes[digest]})
    return {
        "train_samples": len(train),
        "val_samples": len(val),
        "train_boxes": train_objects,
        "val_boxes": val_objects,
        "train_duplicate_images": sum(len(names) - 1 for names in train_hashes.values()),
        "val_duplicate_images": sum(len(names) - 1 for names in val_hashes.values()),
        "cross_split_duplicates": duplicates,
    }


def dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    for sample in dataset:
        identifier = sample["id"].encode()
        digest.update(len(identifier).to_bytes(8, "little"))
        digest.update(identifier)
        for key in ("signal", "boxes", "labels"):
            values = sample[key].contiguous().numpy()
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest()
