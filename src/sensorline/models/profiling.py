"""Feature shapes, parameter count and backbone receptive fields."""

import torch

from .necks import FeatureFusion


def update_receptive_field(block, receptive_field: list, stride: list) -> None:
    for convolution_index in range(block.layers):
        for axis in range(2):
            kernel_extent = (block.kernel[axis] - 1) * block.dilation[axis]
            receptive_field[axis] += kernel_extent * stride[axis]
            if convolution_index == 0:
                stride[axis] *= block.stride[axis]


def profile_backbone(model, signal) -> list[dict]:
    stages = []
    receptive_field = [1, 1]
    stride = [1, 1]
    global_context = False
    for index, block in enumerate(model.backbone):
        update_receptive_field(block, receptive_field, stride)
        signal = block(signal)
        global_context = global_context or block.global_context
        stages.append(
            {
                "index": index,
                "type": block.kind,
                "in_channels": block.in_channels,
                "out_channels": block.out_channels,
                "shape": list(signal.shape),
                "receptive_field_sensor_time": None if global_context else receptive_field.copy(),
                "stride_sensor_time": stride.copy(),
            }
        )
    return stages


def head_receptive_fields(model, stages) -> list:
    """Maximum evaluation RF including known same-resolution neck/head paths."""
    selected = [stages[index] for index in model.feature_indices]
    if model.neck is None:
        fields = [stage["receptive_field_sensor_time"] for stage in selected]
        strides = [stage["stride_sensor_time"] for stage in selected]
    elif isinstance(model.neck, FeatureFusion):
        stride = selected[-1]["stride_sensor_time"]
        source_fields = [stage["receptive_field_sensor_time"] for stage in selected]
        if any(field is None for field in source_fields):
            fields = [None]
        else:
            fields = [[
                max(field[axis] for field in source_fields)
                + model.neck.receptive_field_extent[axis] * stride[axis]
                for axis in range(2)
            ]]
        strides = [stride]
    else:
        # FPN resampling changes dependencies; do not report a local approximation.
        return [None] * len(model.head.predictors)
    result = []
    for field, stride, predictor in zip(fields, strides, model.head.predictors):
        extent = getattr(predictor, "receptive_field_extent", (0, 0))
        result.append(None if field is None else [
            field[axis] + extent[axis] * stride[axis] for axis in range(2)
        ])
    return result


@torch.no_grad()
def model_info(model, shape) -> dict:
    """Report backbone RF and head RF for supported local neck/head paths."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    signal = torch.zeros(shape, device=device)
    try:
        stages = profile_backbone(model, signal)
        features = model.features(signal)
        output = model.head(features)
    finally:
        model.train(was_training)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "structure": str(model),
        "input_shape": list(shape),
        "backbone": stages,
        "head_feature_shapes": [list(feature.shape) for feature in features],
        "prediction_shape": list(output["boxes"].shape),
        "head_receptive_fields_sensor_time": head_receptive_fields(model, stages),
        "trainable_parameters": parameter_count,
    }
