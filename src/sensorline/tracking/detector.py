"""The model as a "frame -> boxes + scores" component.

Decoding, the candidate filter and NMS are those of sensorline.evaluation, so a window
that equals a training frame gets exactly the boxes of per-frame inference. As there,
NMS runs over all candidates above ap_min_confidence and the score threshold τ is
applied to what NMS keeps.

Boxes are half-open intervals in samples (t) and sensors (p), not normalized.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..checkpoint import load_model
from ..evaluation import batch_candidates, suppress_sample


@dataclass(frozen=True)
class Box:
    p1: float
    t1: float
    p2: float
    t2: float

    def shifted(self, dt: float) -> "Box":
        return Box(self.p1, self.t1 + dt, self.p2, self.t2 + dt)

    def as_list(self) -> list:
        return [self.p1, self.t1, self.p2, self.t2]


@dataclass
class Detection:
    window_k: int
    box_local: Box
    box_global: Box
    score: float
    cut_left: bool
    cut_right: bool
    raw_index: int            # position among the window's pre-NMS candidates

    @property
    def cut(self) -> bool:
        return self.cut_left or self.cut_right


@dataclass
class WindowResult:
    k: int
    segment: int
    t0: int
    width: int
    first: bool
    last: bool
    detections_pre_nms: list   # every candidate, as Detection, with nms_kept / accepted flags in `flags`
    flags: list                # (nms_kept, accepted) per pre-NMS candidate
    detections: list           # accepted Detection, after NMS and τ
    infer_ms: float
    window_ms: float = 0.0
    frame: object = None
    valid: int = 0
    raw: object = None   # the window's stream samples before the model input was made, for a screen
    extra: dict = field(default_factory=dict)


class WindowDetector:
    """Run a checkpoint over windows; batching only groups forward passes."""

    def __init__(self, checkpoint: dict, passport: dict, model_settings: dict, edge_tol: float, device):
        self.device = device
        self.model = load_model(checkpoint, device).eval()
        self.sensors, self.width = passport["input_shape"]
        self.postprocess = dict(passport["postprocess"])
        self.score_thr = model_settings["score_thr"]
        self.nms_iou = model_settings["nms_iou"]
        self.edge_tol = edge_tol

    def to_detection(self, window, values, score, index) -> Detection:
        # Internal normalized boxes are (time, sensor, time, sensor).
        t1, p1, t2, p2 = (float(value) for value in values)
        local = Box(p1 * self.sensors, t1 * self.width, p2 * self.sensors, t2 * self.width)
        cut_left = not window.first and local.t1 <= self.edge_tol
        cut_right = not window.last and local.t2 >= self.width - self.edge_tol
        return Detection(window.k, local, local.shifted(window.t0), float(score), cut_left, cut_right, index)

    def __call__(self, windows: list) -> list[WindowResult]:
        """Results in the order of `windows`; a skipped window gets no detections without running the model."""
        inferred = iter(self.infer([window for window in windows if window.skip is None]))
        results = []
        for window in windows:
            if window.skip is not None:
                results.append(WindowResult(window.k, window.segment, window.t0, self.width, window.first,
                                            window.last, [], [], [], 0.0, frame=window.frame, valid=window.valid,
                                            raw=window.raw, extra={"skip": window.skip, "peak": window.peak}))
            else:
                results.append(next(inferred))
        return results

    @torch.no_grad()
    def infer(self, windows: list) -> list[WindowResult]:
        if not windows:
            return []
        frames = torch.from_numpy(np.stack([window.frame for window in windows])).unsqueeze(1).to(self.device)
        started = time.perf_counter()
        output = self.model(frames)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        infer_ms = (time.perf_counter() - started) * 1000 / len(windows)
        targets = [{"size": (self.sensors, self.width)}] * len(windows)
        candidates = batch_candidates(output, targets, (self.sensors, self.width), self.postprocess)
        results = []
        for window, sample in zip(windows, candidates):
            # Candidates are kept in their decoded order; raw_index refers to it.
            sample = {**sample, "index": torch.arange(len(sample["scores"]))}
            kept = suppress_sample(sample, self.nms_iou, self.postprocess["max_detections"])
            kept_indices = set(kept["index"].tolist())
            pre, flags, accepted = [], [], []
            for index, (values, score) in enumerate(zip(sample["boxes"].tolist(), sample["scores"].tolist())):
                pre.append(self.to_detection(window, values, score, index))
                flags.append((index in kept_indices, False))
            for index, score in zip(kept["index"].tolist(), kept["scores"].tolist()):
                detection = pre[index]
                if score < self.score_thr:
                    continue
                # pad_background: a box starting in the padding is not in the recording.
                if detection.box_local.t1 >= window.valid:
                    continue
                flags[index] = (True, True)
                accepted.append(detection)
            results.append(WindowResult(
                window.k, window.segment, window.t0, self.width, window.first, window.last,
                pre, flags, accepted, infer_ms, frame=window.frame, valid=window.valid, raw=window.raw,
                extra={"skip": None, "peak": window.peak},
            ))
        return results
