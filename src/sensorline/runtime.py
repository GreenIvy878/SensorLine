"""Device selection, reproducibility, DataLoader and optimizer setup."""

import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import training_option
from .data import collate_samples


# Cosine decay ends at this fraction of training.learning_rate instead of zero,
# so the final epochs still make small updates.
COSINE_FLOOR = 0.01


def configure_reproducibility(training: dict) -> None:
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(training["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this Python environment")
    return torch.device(name)


def make_loader(dataset, training: dict, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(training["seed"])
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        num_workers=training["num_workers"],
        collate_fn=collate_samples,
        generator=generator,
    )


def build_optimizer(model, training: dict) -> torch.optim.Optimizer:
    options = {
        "lr": training["learning_rate"],
        "weight_decay": training["weight_decay"],
    }
    if training["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), **options)
    if training["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), **options)
    if training["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=training["momentum"],
            **options,
        )
    raise ValueError(f"Unknown optimizer: {training['optimizer']}")


def learning_rate_factor(step: int, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """Multiplier of training.learning_rate for optimizer step number `step` (from 0)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    # The last optimizer step of the run reaches the floor exactly.
    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = min((step - warmup_steps) / decay_steps, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return COSINE_FLOOR + (1 - COSINE_FLOOR) * cosine


def build_scheduler(optimizer, training: dict, steps_per_epoch: int):
    """Per-step schedule, or None for the default constant rate without warmup."""
    schedule = training_option(training, "lr_schedule")
    warmup_steps = training_option(training, "warmup_epochs") * steps_per_epoch
    if schedule == "constant" and warmup_steps == 0:
        return None
    total_steps = training["epochs"] * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, warmup_steps, total_steps, schedule),
    )


def environment_info(device: torch.device, seed: int) -> dict:
    device_name = "CPU"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
        "seed": seed,
        "deterministic_algorithms": True,
        "device_name": device_name,
    }
