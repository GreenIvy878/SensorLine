"""Backbone blocks and their YAML specs. Pairs are ordered as (sensor, time)."""

from torch import nn

from .layers import (
    ACTIVATIONS, NORMALIZATIONS, AxialSelfAttention, BackboneBlock, EfficientChannelAttention,
    activation_layer, conv_unit,
)
from ..validation import finite_number, require_integer, require_mapping


def skip_connection(in_channels, out_channels, stride):
    if in_channels == out_channels and stride == (1, 1):
        return nn.Identity()
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


class ConvBlock(BackboneBlock):
    kind = "conv"
    layers = 1

    def __init__(self, in_channels, out_channels, kernel, stride, dilation, **options):
        super().__init__(in_channels, out_channels, kernel, stride, dilation, **options)
        self.body = nn.Sequential(
            conv_unit(in_channels, out_channels, kernel, stride, dilation, **options)
        )

    def forward(self, signal):
        return self.body(signal)


class ResidualBlock(BackboneBlock):
    kind = "residual"
    layers = 2

    def __init__(self, in_channels, out_channels, kernel, stride, dilation, **options):
        super().__init__(in_channels, out_channels, kernel, stride, dilation, **options)
        self.body = nn.Sequential(
            conv_unit(in_channels, out_channels, kernel, stride, dilation, **options),
            conv_unit(out_channels, out_channels, kernel, (1, 1), dilation, **options),
        )
        self.skip = skip_connection(in_channels, out_channels, stride)
        self.activation = activation_layer(options.get("activation", "silu"))

    def forward(self, signal):
        return self.activation(self.body(signal) + self.skip(signal))


class DepthwiseSeparableBlock(BackboneBlock):
    kind = "depthwise"
    layers = 1

    def __init__(self, in_channels, out_channels, kernel, stride, dilation, **options):
        super().__init__(in_channels, out_channels, kernel, stride, dilation, **options)
        self.body = nn.Sequential(
            conv_unit(
                in_channels, in_channels, kernel, stride, dilation, groups=in_channels, **options
            ),
            conv_unit(in_channels, out_channels, (1, 1), (1, 1), (1, 1), **options),
        )

    def forward(self, signal):
        return self.body(signal)


class TCNBlock(BackboneBlock):
    kind = "tcn"
    layers = 2

    def __init__(self, in_channels, out_channels, kernel, stride, dilation, dropout, **options):
        super().__init__(in_channels, out_channels, kernel, stride, dilation, **options)
        self.body = nn.Sequential(
            conv_unit(in_channels, out_channels, kernel, stride, dilation, **options),
            nn.Dropout2d(dropout),
            conv_unit(out_channels, out_channels, kernel, (1, 1), dilation, **options),
            nn.Dropout2d(dropout),
        )
        self.skip = skip_connection(in_channels, out_channels, stride)
        self.activation = activation_layer(options.get("activation", "silu"))

    def forward(self, signal):
        return self.activation(self.body(signal) + self.skip(signal))


class ECABlock(BackboneBlock):
    kind = "eca"
    layers = 0

    def __init__(self, channels, kernel):
        super().__init__(channels, channels, (1, 1), (1, 1), (1, 1))
        self.global_context = True
        self.body = EfficientChannelAttention(kernel)

    def forward(self, signal):
        return self.body(signal)


class AxialAttentionBlock(BackboneBlock):
    kind = "axial_attention"
    layers = 0

    def __init__(self, channels, heads):
        super().__init__(channels, channels, (1, 1), (1, 1), (1, 1))
        self.global_context = True
        self.body = nn.Sequential(
            AxialSelfAttention(channels, heads, axis=3),
            AxialSelfAttention(channels, heads, axis=2),
        )

    def forward(self, signal):
        return self.body(signal)


CONV_TYPES = {"conv", "residual", "depthwise", "tcn"}
# Operations keep the channels and the resolution and see the whole feature map.
OPERATION_TYPES = {"eca", "axial_attention"}
BLOCK_TYPES = CONV_TYPES | OPERATION_TYPES
BLOCK_DEFAULTS = {
    "stride": [1, 1],
    "activation": "silu",
    "norm": "batch",
    "norm_groups": 1,
    "groups": 1,
    "dropout": 0.1,
    "eca_kernel": 3,
    "heads": 4,
}


def option(spec, key):
    return spec.get(key, BLOCK_DEFAULTS[key])


def pair(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two positive integers [sensor, time]")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"{name} must contain two positive integers [sensor, time]")
    return tuple(value)


def block_geometry(spec):
    """Kernel, stride and dilation of a block, with per-type defaults."""
    kind = spec["type"]
    if kind in OPERATION_TYPES:
        return (1, 1), (1, 1), (1, 1)
    default_kernel = [1, 5] if kind == "tcn" else [3, 5]
    default_dilation = [1, 2] if kind == "tcn" else [1, 1]
    return (
        pair(spec.get("kernel", default_kernel), "kernel"),
        pair(option(spec, "stride"), "stride"),
        pair(spec.get("dilation", default_dilation), "dilation"),
    )


def allowed_block_options(kind):
    allowed = {"type", "channels"}
    if kind in CONV_TYPES:
        allowed |= {"kernel", "stride", "dilation", "activation", "norm", "norm_groups"}
    if kind == "conv":
        allowed.add("groups")
    if kind == "tcn":
        allowed.add("dropout")
    if kind == "eca":
        allowed.add("eca_kernel")
    if kind == "axial_attention":
        allowed.add("heads")
    return allowed


def validate_layer_options(in_channels, spec):
    activation = option(spec, "activation")
    norm = option(spec, "norm")
    if not isinstance(activation, str) or activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
    if not isinstance(norm, str) or norm not in NORMALIZATIONS:
        raise ValueError(f"norm must be one of {sorted(NORMALIZATIONS)}")
    norm_groups = option(spec, "norm_groups")
    require_integer(norm_groups, "norm_groups", minimum=1)
    if "norm_groups" in spec and norm != "group":
        raise ValueError("norm_groups requires norm: group")
    if norm == "group":
        normalized_channels = [spec["channels"]]
        if spec["type"] == "depthwise":
            normalized_channels.append(in_channels)
        if any(channels % norm_groups for channels in normalized_channels):
            raise ValueError("norm_groups must divide every normalized channel count")
    groups = option(spec, "groups")
    require_integer(groups, "groups", minimum=1)
    if in_channels % groups or spec["channels"] % groups:
        raise ValueError("groups must divide input and output channels")


def validate_block_options(kind, in_channels, spec):
    """Range checks of the type-specific numeric options."""
    if kind == "tcn" and not 0 <= finite_number(option(spec, "dropout"), "dropout") < 1:
        raise ValueError("dropout must be in [0, 1)")
    if kind == "eca":
        require_integer(option(spec, "eca_kernel"), "eca_kernel", minimum=1)
        if option(spec, "eca_kernel") % 2 == 0:
            raise ValueError("eca_kernel must be odd")
    if kind == "axial_attention":
        require_integer(option(spec, "heads"), "heads", minimum=1)
        if in_channels % option(spec, "heads"):
            raise ValueError("heads must divide axial_attention channels")


def validate_block_spec(in_channels: int, spec: dict) -> None:
    require_mapping(spec, "backbone block")
    missing = {"type", "channels"} - set(spec)
    if missing:
        raise ValueError(f"Missing block options: {sorted(missing)}")
    kind = spec["type"]
    if not isinstance(kind, str) or kind not in BLOCK_TYPES:
        raise ValueError(f"Unknown backbone block: {kind}; known: {sorted(BLOCK_TYPES)}")
    unknown = set(spec) - allowed_block_options(kind)
    if unknown:
        raise ValueError(f"Unknown {kind} block options: {sorted(map(str, unknown))}")

    require_integer(spec["channels"], "Block channels", minimum=1)
    kernel, _, _ = block_geometry(spec)
    if any(value % 2 == 0 for value in kernel):
        raise ValueError("Use odd kernels for aligned same-padding features")
    if kind in OPERATION_TYPES and spec["channels"] != in_channels:
        raise ValueError(f"{kind} preserves channels; use a preceding conv to change them")
    validate_block_options(kind, in_channels, spec)
    validate_layer_options(in_channels, spec)


def build_block(in_channels: int, spec: dict) -> BackboneBlock:
    kind = spec["type"]
    kernel, stride, dilation = block_geometry(spec)
    arguments = (in_channels, spec["channels"], kernel, stride, dilation)
    layer_options = {key: spec[key] for key in ("activation", "norm", "norm_groups") if key in spec}
    if kind == "conv":
        return ConvBlock(*arguments, groups=option(spec, "groups"), **layer_options)
    if kind == "residual":
        return ResidualBlock(*arguments, **layer_options)
    if kind == "depthwise":
        return DepthwiseSeparableBlock(*arguments, **layer_options)
    if kind == "tcn":
        return TCNBlock(*arguments, dropout=float(option(spec, "dropout")), **layer_options)
    if kind == "eca":
        return ECABlock(in_channels, option(spec, "eca_kernel"))
    return AxialAttentionBlock(in_channels, option(spec, "heads"))
