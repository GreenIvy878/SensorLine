"""Read the supported checkpoint schema before using its contents."""

import torch

from .config import DATA_DESCRIPTION, sensor_count, storage_is_declared, validate_config
from .models import Detector
from .validation import require_mapping


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    require_mapping(checkpoint, "checkpoint")
    version = checkpoint.get("format_version")
    if version != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {version!r}; expected 1")
    required = {"model", "config", "epoch", "validation_metrics", "provenance"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Missing checkpoint fields: {sorted(missing)}")
    # The config may come from an older version of the code; the model is built from it.
    validate_config(checkpoint["config"])
    return checkpoint


def load_model(checkpoint: dict, device) -> Detector:
    """The detector of a loaded checkpoint, with its weights, on `device`."""
    model = Detector(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def require_same_model_and_signal(config: dict, saved: dict) -> None:
    """Checkpoint weights fit a config only with its architecture and preprocessing.

    Scale and offset say how pixels are stored, so they may differ for data that
    declare their own storage (a float32 TIFF in signal units next to 8-bit
    PNG). For any other data a different value is taken for a mistake in the YAML.
    """
    if config["model"] != saved["model"]:
        raise ValueError("YAML architecture differs from the checkpoint")
    for key in ("type", "sensor_axis"):
        if config["data"][key] != saved["data"][key]:
            raise ValueError(f"data.{key} differs from training preprocessing")
    if sensor_count(config["data"]) != sensor_count(saved["data"]):
        raise ValueError("data.sensors differs from the frames the checkpoint was trained on")
    declared = storage_is_declared(config["data"])
    for key in ("scale", "offset"):
        if config["data"][key] != saved["data"][key] and not declared:
            raise ValueError(
                f"data.{key} differs from training preprocessing; only data with "
                f"their own {DATA_DESCRIPTION} may be stored differently"
            )
