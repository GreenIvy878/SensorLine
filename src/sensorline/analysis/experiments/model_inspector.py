"""Actual block shapes, parameter counts and receptive fields of a built model."""

import torch

from ...models import model_info
from ...models.heads import DecoupledPredictor


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def backbone_layers(model, stages, total):
    layers = []
    for stage, block in zip(stages, model.backbone):
        count = parameter_count(block)
        layers.append(
            {
                **stage,
                "name": f"backbone[{stage['index']}]",
                "kernel_sensor_time": list(block.kernel),
                "block_stride_sensor_time": list(block.stride),
                "dilation_sensor_time": list(block.dilation),
                "parameters": count,
                "parameter_fraction": count / total,
            }
        )
    return layers


def neck_layer(model, info, total):
    count = parameter_count(model.neck)
    return {
        "name": "neck",
        "type": type(model.neck).__name__,
        "shapes": info["head_feature_shapes"],
        "parameters": count,
        "parameter_fraction": count / total,
        "receptive_field_sensor_time": None,
    }


def branch_summary(branch):
    return {
        "parameters": parameter_count(branch),
        "layers": [
            {
                "type": type(layer).__name__,
                "in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_sensor_time": list(layer.kernel_size),
                "stride_sensor_time": list(layer.stride),
                "dilation_sensor_time": list(layer.dilation),
            }
            for layer in branch.modules()
            if isinstance(layer, torch.nn.Conv2d)
        ],
        "norm": "BatchNorm2d",
        "activation": "SiLU",
    }


def predictor_geometry(predictor):
    """Output channels of one head predictor and the columns describing its convolutions."""
    if isinstance(predictor, DecoupledPredictor):
        branches = {
            name: branch_summary(getattr(predictor, name)) for name in ("boxes", "objectness")
        }
        out_channels = sum(getattr(predictor, name)[-1].out_channels for name in branches)
        return out_channels, {"branches": branches}
    return predictor.out_channels, {
        "kernel_sensor_time": list(predictor.kernel_size),
        "block_stride_sensor_time": list(predictor.stride),
        "dilation_sensor_time": list(predictor.dilation),
    }


def head_layers(model, info, total):
    layers = []
    receptive_fields = info["head_receptive_fields_sensor_time"]
    for index, (predictor, feature_shape) in enumerate(
        zip(model.head.predictors, info["head_feature_shapes"])
    ):
        out_channels, geometry = predictor_geometry(predictor)
        count = parameter_count(predictor)
        layers.append(
            {
                "name": f"head[{index}]",
                "type": type(predictor).__name__,
                "shape": [feature_shape[0], out_channels, *feature_shape[-2:]],
                **geometry,
                "out_channels": out_channels,
                "parameters": count,
                "parameter_fraction": count / total,
                "receptive_field_sensor_time": receptive_fields[index],
            }
        )
    return layers


def dataset_coverage(receptive_fields, extents):
    """Each head level's theoretical RF next to the annotated object extents."""
    rows = []
    for axis, key in enumerate(("sensor_width", "temporal_length")):
        quantiles = extents.get(key, {})
        for level, rf in enumerate(receptive_fields):
            rows.append({
                "axis": key,
                "head_level": level,
                "receptive_field": rf[axis] if rf else None,
                "median": quantiles.get("50"),
                "p90": quantiles.get("90"),
                "p95": quantiles.get("95"),
            })
    return rows


def inspect_model(model, shape, extents=None):
    info = model_info(model, shape)
    total = parameter_count(model)
    head_rf = info["head_receptive_fields_sensor_time"]

    layers = backbone_layers(model, info["backbone"], total)
    if model.neck is not None:
        layers.append(neck_layer(model, info, total))
    layers.extend(head_layers(model, info, total))

    distribution = [
        {
            "name": name,
            "parameters": parameter_count(module),
            "fraction": parameter_count(module) / total,
        }
        for name, module in (
            ("backbone", model.backbone),
            ("neck", model.neck),
            ("head", model.head),
        )
        if module is not None
    ]
    return {
        "input_shape": list(shape),
        "layers": layers,
        "parameters": total,
        "trainable_parameters": info["trainable_parameters"],
        "parameter_distribution": distribution,
        "head_feature_shapes": info["head_feature_shapes"],
        "head_receptive_fields_sensor_time": head_rf,
        "dataset_coverage": dataset_coverage(head_rf, extents) if extents else [],
        "axes": "tensor [batch, channel, sensor, time]; feature maps X=time, Y=sensor",
        "rf_limitation": (
            "Maximum theoretical receptive field in evaluation mode, not the measured effective context. "
            "Residual and fusion paths count their largest dependency; the refinement convolution and the "
            "3x3 towers of a decoupled head extend it. ECA, axial attention and GroupNorm see the whole "
            "feature map, so the field of such a layer and of every later one is not reported (None); "
            "nor is the resampling inside an FPN, so the backbone field is the one before the neck."
        ),
    }
