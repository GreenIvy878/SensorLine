"""Small NCHW layers and the base class shared by all backbone blocks."""

from torch import nn


ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "hardswish": nn.Hardswish,
    "identity": nn.Identity,
}
NORMALIZATIONS = {"batch", "group", "layer", "none"}


def activation_layer(name):
    return ACTIVATIONS[name]()


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each position, independent of H/W and batch."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, signal):
        values = signal.permute(0, 2, 3, 1)
        return self.norm(values).permute(0, 3, 1, 2)


def normalization_layer(name, channels, norm_groups=1):
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "group":
        return nn.GroupNorm(norm_groups, channels)
    if name == "layer":
        return ChannelLayerNorm(channels)
    return nn.Identity()


class EfficientChannelAttention(nn.Module):
    """ECA: global average followed by a short convolution over channels."""

    def __init__(self, kernel):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel, padding=(kernel - 1) // 2, bias=False)

    def forward(self, signal):
        values = self.pool(signal).flatten(2).transpose(1, 2)
        weights = self.conv(values).sigmoid().transpose(1, 2).unsqueeze(-1)
        return signal * weights


class AxialSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention along one axis: each sensor row attends over
    its time positions (axis 3) or each time column over its sensors (axis 2)."""

    def __init__(self, channels, heads, axis):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        # Zero output projection: the residual branch starts as identity.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, signal):
        batch, channels, sensors, time = signal.shape
        # The attended axis becomes the sequence; the other axis joins the batch.
        if self.axis == 3:
            values = signal.permute(0, 2, 3, 1).reshape(batch * sensors, time, channels)
        else:
            values = signal.permute(0, 3, 2, 1).reshape(batch * time, sensors, channels)
        values = self.norm(values)
        update, _ = self.attention(values, values, values, need_weights=False)
        if self.axis == 3:
            update = update.reshape(batch, sensors, time, channels).permute(0, 3, 1, 2)
        else:
            update = update.reshape(batch, time, sensors, channels).permute(0, 3, 2, 1)
        return signal + update


def conv_unit(
    in_channels, out_channels, kernel, stride, dilation, groups=1,
    activation="silu", norm="batch", norm_groups=1,
):
    padding = tuple(d * (k - 1) // 2 for k, d in zip(kernel, dilation))
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm == "none",
        ),
        normalization_layer(norm, out_channels, norm_groups),
        activation_layer(activation),
    )


class BackboneBlock(nn.Module):
    """Shared metadata used by the builder and receptive-field report."""

    def __init__(self, in_channels, out_channels, kernel, stride, dilation, **options):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.global_context = options.get("norm") == "group"
