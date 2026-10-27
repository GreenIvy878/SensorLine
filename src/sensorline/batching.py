"""Run padded batches at each sample's original time length."""


def forward_groups(model, signal, targets):
    """Yield indices, output, targets and shape for equal-length sample groups.

    Equal-length batches use the original tensor and one forward. Mixed batches
    deliberately run separate groups so head coordinates and BN see no padding.
    """
    groups = {}
    for index, target in enumerate(targets):
        groups.setdefault(target["size"][1], []).append(index)

    for time_steps, indices in groups.items():
        if len(indices) == len(targets):
            group_signal = signal
            group_targets = targets
        else:
            group_signal = signal[indices]
            group_targets = [targets[index] for index in indices]
        if time_steps != signal.shape[-1]:
            group_signal = group_signal[..., :time_steps]
        output = model(group_signal)
        yield indices, output, group_targets, group_signal.shape[-2:]
