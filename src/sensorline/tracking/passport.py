"""The model passport: what a window must look like for a checkpoint, stored next to its weights.

Checkpoints of this repository do not record the frame length or the box statistics of
their training data, so the passport is rebuilt from that data (the same dataset code
that trained the model reads it) and saved as `<checkpoint stem>.passport.json`. The
passport is tied to the checkpoint file by its SHA-256. How raw conveyor recordings are
turned into such frames is not a property of one checkpoint; the tracking config says it
(see preprocess.py).
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ..artifacts import file_sha256, read_json, write_json
from ..config import sensor_count
from ..data import build_dataset, find_images

# 3: the recording transform of version 2 moved to the tracking config.
PASSPORT_VERSION = 3
PASSPORT_SUFFIX = ".passport.json"
# Bit depth of the stored training frames by PIL image mode.
MODE_BITS = {"L": 8, "I;16": 16, "I;16L": 16, "I;16B": 16}


def passport_path(checkpoint_path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(path.stem + PASSPORT_SUFFIX)


def git_commit(start: Path) -> str | None:
    """The commit checked out in the repository containing `start`, read from .git without running git."""
    for directory in (start, *start.parents):
        git = directory / ".git"
        if not (git / "HEAD").is_file():
            continue
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text(encoding="utf-8").strip()
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
        return None
    return None


def training_frames(data: dict) -> tuple[int, str, list[float]]:
    """Time length and image mode shared by all training frames, and box lengths in time steps."""
    shapes = set()
    for path in find_images(Path(data["train"]) / "images"):
        with Image.open(path) as image:
            shapes.add((image.height if data["sensor_axis"] == "x" else image.width, image.mode))
    if len(shapes) != 1:
        raise ValueError(f"Training frames differ in time length or mode: {sorted(shapes)}; the window length is ambiguous")
    (time_steps, mode), = shapes
    dataset = build_dataset(data, "train")
    # Internal boxes are normalized with x = time.
    lengths = [float(length) for boxes in dataset.boxes for length in (boxes[:, 2] - boxes[:, 0]).double() * time_steps]
    return time_steps, mode, lengths


def class_names(data: dict) -> list[str]:
    path = Path(data["train"]).parent / "classes.txt"
    if path.is_file():
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return ["object"]


def build_passport(checkpoint: dict, checkpoint_path) -> dict:
    saved = checkpoint["config"]
    data = saved["data"]
    if not Path(data["train"]).is_dir():
        raise ValueError(
            f"The checkpoint has no passport and its training data {data['train']} is not here to rebuild one; "
            f"copy {passport_path(checkpoint_path).name} next to the checkpoint"
        )
    time_steps, mode, lengths = training_frames(data)
    if mode not in MODE_BITS:
        raise ValueError(f"Training frames have image mode {mode}; the passport knows integer modes {sorted(MODE_BITS)}")
    bits = MODE_BITS[mode]
    return {
        "passport_version": PASSPORT_VERSION,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        # The model reads [batch, 1, sensors, time] float32 tensors.
        "input_shape": [sensor_count(data), time_steps],
        "axis_order": ["sensor", "time"],
        "dtype": "float32",
        "quantize": {"bits": bits, "range": [0, 2 ** bits - 1], "source": f"training frames are {mode} images"},
        "normalize": {"formula": "(pixel - offset) / scale", "scale": data["scale"], "offset": data["offset"],
                      "function": "sensorline.data.pixels_to_signal"},
        "class_names": class_names(data),
        "score_thr_default": saved["evaluation"]["confidence"],
        "postprocess": dict(saved["evaluation"]),
        "train_data_ref": {"train": data["train"], "fingerprint": checkpoint["provenance"]["train_fingerprint"]},
        "trace_lengths_time": sorted(round(length, 4) for length in lengths),
        "code_version": git_commit(Path(__file__).resolve().parent),
    }


def check_passport(passport: dict, checkpoint: dict, checkpoint_path) -> None:
    """The passport must describe this very checkpoint and the preprocessing it was trained with."""
    if passport.get("passport_version") != PASSPORT_VERSION:
        raise ValueError(f"Passport version {passport.get('passport_version')!r}; expected {PASSPORT_VERSION}")
    if passport["checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError(f"{passport_path(checkpoint_path)} belongs to another checkpoint file (SHA-256 differs)")
    data = checkpoint["config"]["data"]
    expected = {"scale": data["scale"], "offset": data["offset"]}
    if {key: passport["normalize"][key] for key in expected} != expected:
        raise ValueError("Passport normalization differs from the checkpoint's data.scale/offset")
    if passport["input_shape"][0] != sensor_count(data):
        raise ValueError("Passport sensor count differs from the checkpoint")
    if passport["axis_order"] != ["sensor", "time"] or passport["dtype"] != "float32":
        raise ValueError("Passport axis order or dtype is not the one this code feeds to the model")


def load_passport(checkpoint: dict, checkpoint_path, log=print) -> dict:
    """The saved passport of a checkpoint, built from its training data and saved on first use."""
    path = passport_path(checkpoint_path)
    passport = read_json(path)
    stale = passport is not None and passport.get("passport_version") != PASSPORT_VERSION
    if passport is None or stale:
        passport = build_passport(checkpoint, checkpoint_path)
        write_json(path, passport)
        log(f"Passport {'rebuilt' if stale else 'built'} from {passport['train_data_ref']['train']} and saved: {path}")
    check_passport(passport, checkpoint, checkpoint_path)
    return passport


def estimate_lmax(passport: dict, quantile: float) -> float | None:
    """The `quantile` of training box lengths in time steps; None without labeled boxes."""
    lengths = passport["trace_lengths_time"]
    return float(np.quantile(lengths, quantile)) if lengths else None
