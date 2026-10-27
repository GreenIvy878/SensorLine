"""Small training-only perturbations of [1, sensors, T] signals and xyxy boxes."""

from copy import deepcopy
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .validation import finite_number, require_integer, require_mapping


DEFAULT_AUGMENTATION = {
    "enabled": False,
    "probability": 0.8,
    "sensor_flip_probability": 0.5,
    "time_flip_probability": 0.0,
    "rotation_probability": 0.0,
    "rotation_degrees": 0.0,
    "shear_probability": 0.0,
    "shear_limit": 0.0,
    "gain_limit": 0.1,
    "noise_std": 0.004,
    "sensor_shift": 1,
    "time_shift": 2,
}
TUNABLE_AUGMENTATION = set(DEFAULT_AUGMENTATION) - {"enabled"}


def augmentation_options(config=None):
    """Complete optional settings; old configs keep augmentation disabled."""
    config = {} if config is None else config
    require_mapping(config, "training.augmentation")
    if "affine_box" in config:
        # Saved runs name the box model under rotation and shear; "corners" is the one kept.
        if config["affine_box"] != "corners":
            raise ValueError("training.augmentation.affine_box was removed; only corners remains")
        config = {key: value for key, value in config.items() if key != "affine_box"}
    unknown = set(config) - set(DEFAULT_AUGMENTATION)
    if unknown:
        raise ValueError(f"Unknown training.augmentation options: {sorted(unknown)}")
    options = {**DEFAULT_AUGMENTATION, **config}
    if not isinstance(options["enabled"], bool):
        raise ValueError("training.augmentation.enabled must be boolean")
    # Only what keeps a transform defined; which strengths suit a sensor line is the YAML's choice.
    for key, maximum in (("probability", 1), ("sensor_flip_probability", 1),
                         ("time_flip_probability", 1), ("rotation_probability", 1),
                         ("shear_probability", 1), ("rotation_degrees", 180),
                         ("shear_limit", 1), ("noise_std", math.inf)):
        value = finite_number(options[key], f"training.augmentation.{key}")
        if not 0 <= value <= maximum:
            raise ValueError(f"training.augmentation.{key} must be in [0, {maximum}]")
    # A gain of 1 - gain_limit must stay positive.
    if not 0 <= finite_number(options["gain_limit"], "training.augmentation.gain_limit") < 1:
        raise ValueError("training.augmentation.gain_limit must be in [0, 1)")
    # Shifts are capped by the frame size when they are drawn.
    for key in ("sensor_shift", "time_shift"):
        require_integer(options[key], f"training.augmentation.{key}")
    return deepcopy(options)


def uniform(low, high):
    return low if low == high else float(torch.empty(()).uniform_(low, high))


def valid_boxes(boxes):
    # Extremely narrow float32 annotations can collapse after reflection/shift.
    # Skip that geometric operation rather than silently removing an object.
    return bool((boxes[:, 2:] > boxes[:, :2]).all())


def shift_bounds(boxes, first, dimension, maximum):
    maximum = min(maximum, dimension - 1)
    low, high = -maximum, maximum
    if len(boxes):
        low = max(low, math.ceil(-float(boxes[:, first].min()) * dimension))
        high = min(high, math.floor((1 - float(boxes[:, first + 2].max())) * dimension))
    return low, high


def sample_shift(bounds):
    low, high = bounds
    return low if low == high else int(torch.randint(low, high + 1, ()))


def translate(signal, boxes, sensor_delta, time_delta, background):
    if sensor_delta == 0 and time_delta == 0:
        return signal, boxes
    sensors, time = signal.shape[-2:]
    offsets = boxes.new_tensor([time_delta / time, sensor_delta / sensors,
                               time_delta / time, sensor_delta / sensors],
                              dtype=torch.float64)
    moved_boxes = (boxes.double() + offsets).clamp(0, 1).to(boxes.dtype)
    if not valid_boxes(moved_boxes):
        return signal, boxes
    moved_signal = torch.full_like(signal, background)
    # Copy the overlapping rectangle; no interpolation or wrap to the other edge.
    source_sensor, source_time = max(0, -sensor_delta), max(0, -time_delta)
    dest_sensor, dest_time = max(0, sensor_delta), max(0, time_delta)
    height, width = sensors - abs(sensor_delta), time - abs(time_delta)
    moved_signal[:, dest_sensor:dest_sensor + height, dest_time:dest_time + width] = (
        signal[:, source_sensor:source_sensor + height, source_time:source_time + width]
    )
    return moved_signal, moved_boxes


def reflect(signal, boxes, axis):
    """Reflect time (axis=0) or sensors (axis=1), including box boundaries."""
    mirrored = boxes.clone()
    mirrored[:, axis] = 1 - boxes[:, axis + 2]
    mirrored[:, axis + 2] = 1 - boxes[:, axis]
    if not valid_boxes(mirrored):
        return signal, boxes
    return signal.flip(-1 - axis), mirrored


def corner_extents(boxes, matrix):
    """Axis-aligned float64 extents of the four transformed box corners."""
    corners = boxes.double()[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    moved = (corners - 0.5) @ matrix.T + 0.5
    return torch.cat((moved.amin(dim=1), moved.amax(dim=1)), dim=1)


def affine_geometry(signal, boxes, matrix, background):
    """Transform about the frame centre in normalized x=time, y=sensor space.

    The new axis-aligned box encloses the transformed label rectangle, so it
    grows with the angle. Reject the operation if any new box would leave the
    frame. Nearest-neighbour sampling avoids smoothing narrow signal valleys;
    padding uses calibrated raw zero.
    """
    matrix = torch.as_tensor(matrix, dtype=torch.float64, device=boxes.device)
    extents = corner_extents(boxes, matrix)
    if bool(((extents < 0) | (extents > 1)).any()):
        return signal, boxes
    moved_boxes = extents.to(boxes.dtype)
    if not valid_boxes(moved_boxes):
        return signal, boxes
    # grid_sample maps destination pixel centres back to source coordinates.
    theta = signal.new_zeros((1, 2, 3))
    theta[0, :, :2] = torch.linalg.inv(matrix).to(signal)
    grid = F.affine_grid(theta, (1, *signal.shape), align_corners=False)
    transformed = F.grid_sample(
        (signal - background).unsqueeze(0), grid, mode="nearest",
        padding_mode="zeros", align_corners=False,
    ).squeeze(0) + background
    return transformed, moved_boxes


def rotate(signal, boxes, degrees, background):
    if degrees == 0:
        return signal, boxes
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    sensors, time = signal.shape[-2:]
    # Rotate in pixel space even for rectangular [sensors, T] frames.
    matrix = [[cosine, -sine * sensors / time],
              [sine * time / sensors, cosine]]
    return affine_geometry(signal, boxes, matrix, background)


def shear(signal, boxes, amount, background):
    """Tilt along time: x' = x + amount * (y - 0.5), with y unchanged."""
    if amount == 0:
        return signal, boxes
    return affine_geometry(signal, boxes, [[1, amount], [0, 1]], background)


class SignalAugmentation:
    """Random geometry plus amplitude/noise, with every annotated object retained."""

    def __init__(self, config, background=0.0):
        self.options = augmentation_options(config)
        self.background = background

    # The order of random draws is fixed: sample choice, flips, rotation, shear,
    # shifts, gain, noise. Seeded runs depend on it.
    def __call__(self, sample):
        options = self.options
        if not options["enabled"] or options["probability"] == 0:
            return sample
        if options["probability"] < 1 and uniform(0, 1) >= options["probability"]:
            return sample
        signal, boxes = self.transform_geometry(sample["signal"].clone(), sample["boxes"].clone())
        signal = self.transform_amplitude(signal)
        return {**sample, "signal": signal, "boxes": boxes,
                "labels": sample["labels"].clone()}

    def transform_geometry(self, signal, boxes):
        options = self.options
        for axis, key in ((1, "sensor_flip_probability"), (0, "time_flip_probability")):
            if options[key] > 0 and uniform(0, 1) < options[key]:
                signal, boxes = reflect(signal, boxes, axis)
        for transform, probability, strength in (
            (rotate, "rotation_probability", "rotation_degrees"),
            (shear, "shear_probability", "shear_limit"),
        ):
            limit = options[strength]
            if limit > 0 and options[probability] > 0 and uniform(0, 1) < options[probability]:
                signal, boxes = transform(signal, boxes, uniform(-limit, limit), self.background)
        sensor_delta = sample_shift(shift_bounds(boxes, 1, signal.shape[-2],
                                                 options["sensor_shift"]))
        time_delta = sample_shift(shift_bounds(boxes, 0, signal.shape[-1],
                                               options["time_shift"]))
        return translate(signal, boxes, sensor_delta, time_delta, self.background)

    def transform_amplitude(self, signal):
        options = self.options
        if options["gain_limit"]:
            gain = uniform(1 - options["gain_limit"], 1 + options["gain_limit"])
            signal = self.background + (signal - self.background) * gain
        if options["noise_std"]:
            sigma = uniform(0, options["noise_std"])
            signal = signal + torch.randn_like(signal) * sigma
        if options["gain_limit"] or options["noise_std"]:
            # Raw grayscale measurements have a zero lower bound. Do not assume
            # an upper bound of 1: the adapter also accepts calibrated/16-bit data.
            signal = signal.clamp_min(self.background)
        return signal


class AugmentedDataset(Dataset):
    """Wrap only the training loader; fingerprints/audits use the original Dataset."""

    def __init__(self, dataset, config, background=0.0):
        self.dataset = dataset
        self.transform = SignalAugmentation(config, background)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset[index])
