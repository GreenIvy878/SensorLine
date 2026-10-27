"""Direct box and objectness predictions on each feature level."""

import math

import torch
from torch import nn

from .layers import conv_unit


def feature_grid(raw):
    height, width = raw.shape[1:3]
    sensor_positions = torch.arange(height, device=raw.device, dtype=raw.dtype)
    time_positions = torch.arange(width, device=raw.device, dtype=raw.dtype)
    sensor_grid, time_grid = torch.meshgrid(
        sensor_positions, time_positions, indexing="ij"
    )
    grid = torch.stack((time_grid, sensor_grid), dim=-1)
    scale = raw.new_tensor([width, height])
    return grid, scale


# Initial biases: a box of sigmoid(-2) = 12% of the frame per side, and an object
# probability of 1% per cell, so early losses are not dominated by empty cells.
INITIAL_SIZE_LOGIT = -2.0
OBJECT_PRIOR = 0.01


def decode_level(raw) -> dict:
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite model output: raw head predictions")
    grid, scale = feature_grid(raw)
    centers = (grid + raw[..., :2].sigmoid() * 2 - 0.5) / scale
    sizes = raw[..., 2:4].sigmoid()
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
    height, width = raw.shape[1:3]
    return {
        "boxes": boxes.flatten(1, 2),
        "objectness": raw[..., 4].flatten(1),
        "anchors": ((grid + 0.5) / scale).reshape(-1, 2),
        "cell_sizes": (1 / scale).expand(height * width, 2),
    }


class BBoxHead(nn.Module):
    """One box and one objectness logit per cell for the single object class."""

    def __init__(self, input_channels):
        super().__init__()
        self.predictors = nn.ModuleList(
            nn.Conv2d(channels, 5, 1) for channels in input_channels
        )
        self.initialize_predictions()

    def initialize_predictions(self) -> None:
        for predictor in self.predictors:
            nn.init.normal_(predictor.weight, std=0.01)
            nn.init.zeros_(predictor.bias)
            with torch.no_grad():
                predictor.bias[2:4].fill_(INITIAL_SIZE_LOGIT)
                predictor.bias[4] = math.log(OBJECT_PRIOR / (1 - OBJECT_PRIOR))

    def forward(self, features) -> dict:
        levels = []
        for feature, predictor in zip(features, self.predictors):
            raw = predictor(feature).permute(0, 2, 3, 1)
            levels.append(decode_level(raw))
        return {
            "boxes": torch.cat([level["boxes"] for level in levels], dim=1),
            "objectness": torch.cat([level["objectness"] for level in levels], dim=1),
            "anchors": torch.cat([level["anchors"] for level in levels]),
            "cell_sizes": torch.cat([level["cell_sizes"] for level in levels]),
        }


class DecoupledPredictor(nn.Module):
    """Separate local transformations for box geometry and objectness."""

    receptive_field_extent = (2, 2)

    def __init__(self, input_channels, channels):
        super().__init__()
        self.boxes = nn.Sequential(
            conv_unit(input_channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.Conv2d(channels, 4, 1),
        )
        self.objectness = nn.Sequential(
            conv_unit(input_channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.Conv2d(channels, 1, 1),
        )
        self.initialize_predictions()

    def initialize_predictions(self) -> None:
        for projection in (self.boxes[-1], self.objectness[-1]):
            nn.init.normal_(projection.weight, std=0.01)
            nn.init.zeros_(projection.bias)
        with torch.no_grad():
            self.boxes[-1].bias[2:4].fill_(INITIAL_SIZE_LOGIT)
            self.objectness[-1].bias.fill_(math.log(OBJECT_PRIOR / (1 - OBJECT_PRIOR)))

    def forward(self, feature):
        return torch.cat((self.boxes(feature), self.objectness(feature)), dim=1)


class DecoupledBBoxHead(BBoxHead):
    """Keep BBoxHead decoding while learning box and score features separately."""

    def __init__(self, input_channels, channels=24):
        # Each DecoupledPredictor initializes itself, so BBoxHead.__init__ is skipped.
        nn.Module.__init__(self)
        self.predictors = nn.ModuleList(
            DecoupledPredictor(input_width, channels)
            for input_width in input_channels
        )
