"""Size and arithmetic cost of a built detector; computed from one zero input, no data."""

from collections import defaultdict
import math

import torch
from torch import nn

from ..experiments.model_inspector import inspect_model

# Layers with weights do the multiply-accumulates (MACs) that dominate the cost.
# Norms, activations and pooling are elementwise and are counted separately.
WEIGHTED = (nn.Conv1d, nn.Conv2d, nn.Linear)


def block_name(module_name: str) -> str:
    """The layer-table row a module belongs to: backbone[i], neck or head[i]."""
    parts = module_name.split(".")
    if parts[0] == "backbone":
        return f"backbone[{parts[1]}]"
    if parts[0] == "head":
        return f"head[{parts[2]}]"
    return parts[0]


def layer_macs(module, output) -> int:
    if isinstance(module, nn.Linear):
        return output.numel() * module.in_features
    return output.numel() * module.in_channels // module.groups * math.prod(module.kernel_size)


@torch.no_grad()
def measure_cost(model, shape) -> dict:
    """MACs, elementwise operations and layer outputs of one forward pass, by block."""
    macs, elementwise, outputs = defaultdict(int), defaultdict(int), defaultdict(int)
    weighted_layers = 0
    hooks = []
    for name, module in model.named_modules():
        if not name or any(True for _ in module.children()) or isinstance(module, nn.Identity):
            continue
        weighted_layers += isinstance(module, WEIGHTED)

        def hook(module, inputs, output, key=block_name(name)):
            if isinstance(module, WEIGHTED):
                macs[key] += layer_macs(module, output)
            else:
                elementwise[key] += output.numel()
            outputs[key] += output.numel()

        hooks.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        output = model(torch.zeros(shape, device=next(model.parameters()).device))
    finally:
        model.train(was_training)
        for handle in hooks:
            handle.remove()
    return {"macs": dict(macs), "elementwise": dict(elementwise), "outputs": dict(outputs),
            "weighted_layers": weighted_layers, "predictions": output["boxes"].shape[1]}


def model_profile(model, sensors: int, time_steps: int, extents=None) -> dict:
    """Everything about one detector that does not need data: size, cost, geometry."""
    shape = (1, 1, sensors, time_steps)
    inspection = inspect_model(model, shape, extents)
    cost = measure_cost(model, shape)
    total_macs = sum(cost["macs"].values())
    for layer in inspection["layers"]:
        layer["macs"] = cost["macs"].get(layer["name"], 0)
        layer["mac_fraction"] = layer["macs"] / total_macs if total_macs else None
    parts = {"backbone": "backbone[", "neck": "neck", "head": "head["}
    for row in inspection["parameter_distribution"]:
        prefix = parts[row["name"]]
        row["macs"] = sum(value for key, value in cost["macs"].items() if key.startswith(prefix))
        row["mac_fraction"] = row["macs"] / total_macs if total_macs else None

    head_shapes = inspection["head_feature_shapes"]
    fields = [field for field in inspection["head_receptive_fields_sensor_time"] if field]
    return {
        "input_shape": list(shape),
        "parameters": inspection["parameters"],
        "trainable_parameters": inspection["trainable_parameters"],
        "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "macs": total_macs,
        # FLOPs as usually quoted: one multiply and one add per MAC.
        "flops": 2 * total_macs,
        "macs_per_time_step": total_macs / time_steps,
        "elementwise_operations": sum(cost["elementwise"].values()),
        # Every layer output of one frame; training keeps most of them for backward.
        "activation_bytes": 4 * sum(cost["outputs"].values()),
        "weighted_layers": cost["weighted_layers"],
        # Box candidates per frame before the confidence filter and NMS.
        "predictions_per_frame": cost["predictions"],
        "output_grid_sensor_time": [shape[-2:] for shape in head_shapes],
        "output_stride_sensor_time": [[sensors / s[-2], time_steps / s[-1]] for s in head_shapes],
        "receptive_field_sensor_time": [max(f[axis] for f in fields) for axis in range(2)] if fields else None,
        "layers": inspection["layers"],
        "parameter_distribution": inspection["parameter_distribution"],
        "dataset_coverage": inspection["dataset_coverage"],
        "rf_limitation": inspection["rf_limitation"],
    }

