"""Connect the YAML backbone, optional neck and detection head."""

from torch import nn

from .blocks import block_geometry, build_block, validate_block_spec
from .heads import BBoxHead, DecoupledBBoxHead
from .necks import FPN, FeatureFusion
from ..validation import require_integer, require_mapping


DEFAULT_NECK = {"type": "none"}
DEFAULT_HEAD = {"type": "bbox"}
DEFAULT_NECK_CHANNELS = 32
DEFAULT_HEAD_CHANNELS = 24


def validate_backbone_specs(specs) -> list[tuple[int, int]]:
    """Validate every block and return the cumulative (sensor, time) strides."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("model.backbone must be a nonempty block list")
    strides = []
    in_channels = 1
    sensor_stride = time_stride = 1
    for index, spec in enumerate(specs):
        try:
            validate_block_spec(in_channels, spec)
        except ValueError as error:
            raise ValueError(f"backbone[{index}]: {error}") from error
        in_channels = spec["channels"]
        _, stride, _ = block_geometry(spec)
        sensor_stride *= stride[0]
        time_stride *= stride[1]
        strides.append((sensor_stride, time_stride))
    return strides


def require_feature_indices(indices, block_count, message) -> None:
    """Distinct increasing backbone indices; the caller checks how many."""
    if not isinstance(indices, list):
        raise ValueError(message)
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(message)
        if not 0 <= index < block_count:
            raise ValueError(message)
    if indices != sorted(set(indices)):
        raise ValueError(message)


def validate_fpn_features(indices, strides) -> None:
    message = "FPN features must be at least two distinct increasing backbone indices"
    require_feature_indices(indices, len(strides), message)
    if len(indices) < 2:
        raise ValueError(message)
    if len({strides[index] for index in indices}) != len(indices):
        raise ValueError("FPN requires selected features at distinct resolutions")


def validate_fusion_features(indices, strides) -> None:
    message = "Feature fusion requires exactly two distinct increasing backbone indices"
    require_feature_indices(indices, len(strides), message)
    if len(indices) != 2:
        raise ValueError(message)
    if strides[indices[0]] != strides[indices[1]]:
        raise ValueError("Feature fusion requires selected features at the same resolution")


def validate_neck_spec(config: dict, strides) -> None:
    require_mapping(config, "model.neck")
    kind = config.get("type", "none")
    if kind == "none":
        if set(config) != {"type"}:
            raise ValueError("neck none accepts only type")
    elif kind == "feature_fusion":
        if set(config) - {"type", "features", "channels", "refine"}:
            raise ValueError("Feature fusion accepts type, features, channels and refine")
        validate_fusion_features(config.get("features", []), strides)
        require_integer(
            config.get("channels", DEFAULT_NECK_CHANNELS), "Feature fusion channels", minimum=1
        )
        if not isinstance(config.get("refine", False), bool):
            raise ValueError("Feature fusion refine must be a boolean")
    elif kind == "fpn":
        if set(config) - {"type", "features", "channels"}:
            raise ValueError("FPN accepts type, features and channels")
        validate_fpn_features(config.get("features", []), strides)
        require_integer(config.get("channels", DEFAULT_NECK_CHANNELS), "FPN channels", minimum=1)
    else:
        raise ValueError(f"Unknown neck: {kind}")


def validate_head_spec(config: dict) -> None:
    require_mapping(config, "model.head")
    kind = config.get("type")
    if kind == "bbox":
        if set(config) != {"type"}:
            raise ValueError("head bbox accepts only type")
    elif kind == "bbox_decoupled":
        if set(config) - {"type", "channels"}:
            raise ValueError("head bbox_decoupled accepts type and channels")
        require_integer(
            config.get("channels", DEFAULT_HEAD_CHANNELS), "Decoupled head channels", minimum=1
        )
    else:
        raise ValueError(f"Unknown head: {kind}; known: bbox, bbox_decoupled")


def validate_model_config(config: dict) -> None:
    """Validate YAML structure and block parameters without initializing weights."""
    require_mapping(config, "model")
    if set(config) - {"backbone", "neck", "head"}:
        raise ValueError("model accepts backbone, neck and head only")
    strides = validate_backbone_specs(config.get("backbone"))
    validate_neck_spec(config.get("neck", DEFAULT_NECK), strides)
    validate_head_spec(config.get("head", DEFAULT_HEAD))


def build_backbone(specs: list[dict]) -> nn.ModuleList:
    blocks = nn.ModuleList()
    in_channels = 1
    for spec in specs:
        block = build_block(in_channels, spec)
        blocks.append(block)
        in_channels = block.out_channels
    return blocks


def build_neck(config: dict, backbone):
    """Return the neck, the backbone indices it reads and the head input channels."""
    if config["type"] == "none":
        return None, [len(backbone) - 1], [backbone[-1].out_channels]
    indices = config["features"]
    channels = config.get("channels", DEFAULT_NECK_CHANNELS)
    input_channels = [backbone[index].out_channels for index in indices]
    if config["type"] == "feature_fusion":
        neck = FeatureFusion(input_channels, channels, config.get("refine", False))
        return neck, indices, [channels]
    return FPN(input_channels, channels), indices, [channels] * len(indices)


def build_head(config: dict, input_channels):
    if config["type"] == "bbox_decoupled":
        return DecoupledBBoxHead(input_channels, config.get("channels", DEFAULT_HEAD_CHANNELS))
    return BBoxHead(input_channels)


class Detector(nn.Module):
    """Signal → selected backbone features → optional neck → boxes.

    The config is expected to have passed validate_model_config.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.backbone = build_backbone(config["backbone"])
        self.neck, self.feature_indices, head_channels = build_neck(
            config.get("neck", DEFAULT_NECK), self.backbone
        )
        self.head = build_head(config.get("head", DEFAULT_HEAD), head_channels)

    def features(self, signal):
        selected = []
        for index, block in enumerate(self.backbone):
            signal = block(signal)
            if index in self.feature_indices:
                selected.append(signal)
        if self.neck is not None:
            selected = self.neck(selected)
        return selected

    def forward(self, signal):
        return self.head(self.features(signal))
