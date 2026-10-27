"""Exponential moving average of model weights for validation and checkpoints."""

from copy import deepcopy

import torch


# Early updates use a smaller decay so the average leaves the random initialization
# quickly; a decay of 0.99 is reached after about 890 steps, 0.999 after about 9000.
EMA_WARMUP_STEPS = 10


class ModelEma:
    """Averaged copy of a training model; it is never trained directly."""

    def __init__(self, model, decay: float):
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (EMA_WARMUP_STEPS + self.updates))
        source = model.state_dict()
        for name, average in self.model.state_dict().items():
            if average.dtype.is_floating_point:
                # Weights and BatchNorm running statistics are averaged alike.
                average.mul_(decay).add_(source[name], alpha=1 - decay)
            else:
                # Integer buffers such as num_batches_tracked are counters.
                average.copy_(source[name])


def build_ema(model, decay: float) -> ModelEma | None:
    return ModelEma(model, decay) if decay > 0 else None
