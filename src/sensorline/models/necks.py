"""Optional multi-resolution pyramid or same-resolution feature fusion."""

from torch import nn
from torch.nn import functional as F

from .layers import conv_unit


class FPN(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(input_width, channels, 1) for input_width in input_channels
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(channels, channels, 3, padding=1) for _ in input_channels
        )

    def merge_top_down(self, features):
        merged = [layer(feature) for layer, feature in zip(self.lateral, features)]
        for index in range(len(merged) - 2, -1, -1):
            upsampled = F.interpolate(
                merged[index + 1],
                size=merged[index].shape[-2:],
                mode="nearest",
            )
            merged[index] = merged[index] + upsampled
        return merged

    def forward(self, features):
        merged = self.merge_top_down(features)
        return [layer(feature) for layer, feature in zip(self.smooth, merged)]


class FeatureFusion(nn.Module):
    """Fuse early detail with deep context without resampling either feature."""

    def __init__(self, input_channels, channels, refine=False):
        super().__init__()
        self.early_projection = conv_unit(
            input_channels[0], channels, (1, 1), (1, 1), (1, 1)
        )
        self.deep_projection = (
            nn.Identity()
            if input_channels[1] == channels
            else conv_unit(input_channels[1], channels, (1, 1), (1, 1), (1, 1))
        )
        self.refine = (
            conv_unit(channels, channels, (3, 3), (1, 1), (1, 1))
            if refine else nn.Identity()
        )
        self.receptive_field_extent = (2, 2) if refine else (0, 0)

    def forward(self, features):
        early, deep = features
        fused = self.early_projection(early) + self.deep_projection(deep)
        return [self.refine(fused)]
