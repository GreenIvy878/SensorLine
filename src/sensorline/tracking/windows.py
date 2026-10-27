"""Cut the preprocessed stream into windows of the training frame length.

Window k of a segment starts at segment start + k·S. A window is emitted as soon as
one sample after it has arrived (then it is not the last) or the segment ends (then it
is), so offline and streaming runs see the same windows with the same flags. The
buffer holds stream samples; `render` (the window stage of the Preprocessor) turns the
samples of a window into the model input.
"""

import math
from dataclasses import dataclass

import numpy as np

from .passport import estimate_lmax


@dataclass
class Window:
    k: int                 # index over the whole recording
    segment: int
    t0: int                # stream sample of the first column
    frame: np.ndarray      # [sensors, W] float32 in model scale
    first: bool            # first window of its segment: nothing before it
    last: bool             # last window of its segment: nothing after it
    valid: int             # columns holding data; the rest is padding
    skip: str | None = None   # "empty" or "stationary": not given to the model (see preprocess)
    peak: float | None = None
    raw: np.ndarray | None = None   # [sensors, W] stream samples before the window stage


def auto_stride(width: int, lmax: float | None, margin: float) -> tuple[int, str]:
    """S = min(W // 2, floor(W - L_max - margin)), and how it was chosen."""
    half = width // 2
    if lmax is None:
        return half, "no training boxes to estimate L_max: S = W // 2"
    limit = math.floor(width - lmax - margin)
    if limit < 1:
        raise ValueError(f"Window {width} is too short for objects of {lmax:.2f} samples plus margin {margin}")
    stride = min(half, limit)
    return stride, f"L_max {lmax:.2f} samples, margin {margin}: S = min({half}, {limit}) = {stride}"


def resolve_stride(settings: dict, passport: dict) -> dict:
    """Window geometry: W from the passport, S from the config or estimated, and the overlap check."""
    width = passport["input_shape"][1]
    options = settings["windows"]
    lmax = estimate_lmax(passport, options["lmax_quantile"])
    if options["stride"] == "auto":
        stride, reason = auto_stride(width, lmax, options["margin"])
    else:
        stride, reason = options["stride"], "set in the config"
    if stride > width:
        raise ValueError(f"Stride {stride} exceeds the window length {width}: samples would be skipped")
    overlap = width - stride
    warning = None
    if lmax is not None and overlap < lmax + options["margin"]:
        warning = (f"Overlap O = {overlap} is below L_max + margin = {lmax + options['margin']:.2f}: "
                   "long objects may never fit whole into one window")
    return {"width": width, "stride": stride, "overlap": overlap, "lmax": lmax, "reason": reason, "warning": warning}


class Windower:
    """Feed preprocessed blocks of one segment at a time; get the windows that became complete."""

    def __init__(self, sensors: int, width: int, stride: int, tail_mode: str, render=None):
        self.sensors, self.width, self.stride, self.tail_mode = sensors, width, stride, tail_mode
        self.render = render or (lambda values: (values.astype(np.float32), None, None))
        self.k = 0
        self.segment = None

    def start_segment(self, segment: int, start: int) -> None:
        self.segment = segment
        self.segment_start = start
        self.buffer = np.empty((self.sensors, 0), dtype=np.float64)
        self.buffer_start = start          # stream sample of buffer column 0
        self.next_t0 = start
        self.emitted = 0
        self.received = 0

    def window(self, t0: int, last: bool, frame=None, valid=None) -> Window:
        if frame is None:
            offset = t0 - self.buffer_start
            frame = self.buffer[:, offset:offset + self.width]
        raw = frame.copy()
        frame, skip, peak = self.render(frame)
        window = Window(self.k, self.segment, t0, frame, self.emitted == 0, last,
                        self.width if valid is None else valid, skip, peak, raw)
        self.k += 1
        self.emitted += 1
        return window

    def trim(self) -> None:
        # Keep the next window and the last W samples, which an align_end tail may need.
        end = self.buffer_start + self.buffer.shape[1]
        keep_from = min(self.next_t0, end - self.width) - self.buffer_start
        if keep_from > 0:
            self.buffer = self.buffer[:, keep_from:]
            self.buffer_start += keep_from

    def push(self, start: int, values: np.ndarray) -> list[Window]:
        if start != self.buffer_start + self.buffer.shape[1]:
            raise ValueError(f"Block at sample {start} does not continue the segment")
        self.buffer = np.concatenate((self.buffer, values), axis=1)
        self.received += values.shape[1]
        windows = []
        end = self.buffer_start + self.buffer.shape[1]
        # A window is not the last once a sample after it exists.
        while self.next_t0 + self.width < end:
            windows.append(self.window(self.next_t0, last=False))
            self.next_t0 += self.stride
        self.trim()
        return windows

    def background(self) -> np.ndarray:
        """Padding level: per-sensor median of the samples still in the buffer."""
        return np.median(self.buffer, axis=1)[:, None]

    def padded(self, t0: int, end: int) -> Window:
        offset = t0 - self.buffer_start
        data = self.buffer[:, offset:end - self.buffer_start]
        frame = np.repeat(self.background(), self.width, axis=1)
        frame[:, :data.shape[1]] = data
        return self.window(t0, last=True, frame=frame, valid=data.shape[1])

    def finish(self) -> list[Window]:
        """Windows that only the end of the segment completes, then the tail."""
        end = self.buffer_start + self.buffer.shape[1]
        if self.received == 0:
            return []
        if self.received < self.width:
            # A segment shorter than one window: one padded window.
            return [self.padded(self.segment_start, end)]
        windows = []
        if self.next_t0 + self.width == end:
            windows.append(self.window(self.next_t0, last=True))
        elif self.next_t0 < end:
            if self.tail_mode == "align_end":
                windows.append(self.window(end - self.width, last=True))
            else:
                windows.append(self.padded(self.next_t0, end))
        return windows
