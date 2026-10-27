"""Measured speed and memory of a detector on real frames.

Wall-clock times on the chosen device, with CUDA synchronized around every measured
step. Reading files is not measured: the frames are loaded once beforehand.
"""

from copy import deepcopy
import math
import time

import numpy as np
import torch

from ...batching import forward_groups
from ...data import collate_samples
from ...evaluation import decode_predictions
from ...losses import DetectionLoss

WARMUP_STEPS = 10
LATENCY_FRAMES = 200
THROUGHPUT_SECONDS = 3.0
MAX_PASSES = 20
TRAINING_STEPS = 20


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distribution(seconds: list) -> dict:
    values = np.asarray(seconds) * 1000
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90)),
            "mean_ms": float(values.mean()), "count": len(values)}


def batches(samples: list, size: int) -> list:
    return [collate_samples(samples[start:start + size]) for start in range(0, len(samples), size)]


class PeakMemory:
    """Peak CUDA memory above what was allocated when the measurement began."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        if self.device.type == "cuda":
            synchronize(self.device)
            self.base = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *error):
        synchronize(self.device)
        self.peak = (torch.cuda.max_memory_allocated(self.device) - self.base
                     if self.device.type == "cuda" else None)


def pipeline_step(model, signal, targets, device, evaluation) -> tuple[float, float, list]:
    """One batch: transfer and forward, then decoding with NMS. Returns both times."""
    start = time.perf_counter()
    signal = signal.to(device)
    forward = postprocess = 0.0
    predictions = []
    for _, output, group_targets, shape in forward_groups(model, signal, targets):
        synchronize(device)
        middle = time.perf_counter()
        forward += middle - start
        # Decoding moves the candidates to the CPU, so it ends synchronized.
        predictions.extend(decode_predictions(output, group_targets, shape, evaluation))
        start = time.perf_counter()
        postprocess += start - middle
    return forward, postprocess, predictions


@torch.no_grad()
def inference_timing(model, samples, device, evaluation, batch_size, report) -> dict:
    model.eval()
    single = batches(samples[:LATENCY_FRAMES], 1)
    grouped = batches(samples, batch_size)
    report("Warm-up", 0, WARMUP_STEPS)
    for step in range(WARMUP_STEPS):
        signal, targets = grouped[step % len(grouped)]
        pipeline_step(model, signal, targets, device, evaluation)
    synchronize(device)

    forward, post = [], []
    for index, (signal, targets) in enumerate(single):
        report("Latency, one frame at a time", index, len(single))
        f, p, _ = pipeline_step(model, signal, targets, device, evaluation)
        forward.append(f)
        post.append(p)

    passes, elapsed, forward_total, frames = 0, 0.0, 0.0, 0
    candidates, detections = [], []
    with PeakMemory(device) as memory:
        while passes < MAX_PASSES and (passes == 0 or elapsed < THROUGHPUT_SECONDS):
            for index, (signal, targets) in enumerate(grouped):
                report(f"Throughput, batches of {batch_size} (pass {passes + 1})", index, len(grouped))
                f, p, predictions = pipeline_step(model, signal, targets, device, evaluation)
                forward_total += f
                elapsed += f + p
                frames += len(targets)
                if passes == 0:
                    candidates.extend(len(row["scores"]) for row in predictions)
                    detections.extend(int((row["scores"] >= evaluation["confidence"]).sum())
                                      for row in predictions)
            passes += 1
    return {
        "latency_forward": distribution(forward),
        "latency_pipeline": distribution([f + p for f, p in zip(forward, post)]),
        "latency_postprocess": distribution(post),
        "throughput_forward_fps": frames / forward_total,
        "throughput_pipeline_fps": frames / elapsed,
        "postprocess_share": 1 - forward_total / elapsed,
        "throughput_frames": frames,
        "throughput_passes": passes,
        "peak_inference_bytes": memory.peak,
        "candidates_per_frame": float(np.mean(candidates)),
        "detections_per_frame": float(np.mean(detections)),
    }


def training_timing(model, samples, device, loss_config, batch_size, report) -> dict:
    """Forward, loss and backward of one batch on a copy of the model; no optimizer step."""
    model = deepcopy(model).train()
    criterion = DetectionLoss(loss_config)
    signal, targets = collate_samples((samples * math.ceil(batch_size / len(samples)))[:batch_size])
    signal = signal.to(device)

    def step():
        model.zero_grad(set_to_none=True)
        total = None
        for _, output, group_targets, shape in forward_groups(model, signal, targets):
            loss = criterion(output, group_targets, shape)["total"] * len(group_targets) / len(targets)
            total = loss if total is None else total + loss
        total.backward()

    for index in range(3):
        report("Training step, warm-up", index, 3)
        step()
    seconds = []
    with PeakMemory(device) as memory:
        for index in range(TRAINING_STEPS):
            report(f"Training step, batch of {batch_size}", index, TRAINING_STEPS)
            synchronize(device)
            start = time.perf_counter()
            step()
            synchronize(device)
            seconds.append(time.perf_counter() - start)
    return {"step": distribution(seconds), "peak_training_bytes": memory.peak}
