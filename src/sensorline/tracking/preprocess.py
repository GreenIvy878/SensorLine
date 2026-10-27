"""Bring CSV recordings to the frames the model was trained on.

Two kinds of input:

pixels   the values are stored like the training images: 8-bit pixels. A window goes
         through sensorline.data.pixels_to_signal, as the training frames do.
raw_adc  the recordings of the conveyor: 12-bit ADC codes at the full poll rate. The
         training frames were made from them by the recording transform of the
         passport (reconstructed by tools/locate_frames.py and verified there on the
         training frames):
           stream:  mean over 1 / model_rate_hz of polls (10 polls of the 200 Hz
                    training recordings), on a grid aligned to the file start. A
                    recording polled at another rate is averaged over the same time,
                    with fractional weights when that is not a whole number of polls;
           window:  minus the median of all values of the window (the belt level),
                    times 255 / the window's maximum, rounded and clipped to [0, 255];
         then the same pixels_to_signal. Two kinds of window are not given to the
         model, since training had nothing like them:
           empty       its peak is below empty_peak × the weakest training frame;
                       scaled to 255 its noise would look like objects;
           stationary  a sensor stays that far above the median through the whole
                       window: something lies still under the line (the belt
                       stopped). A moving object covers a sensor for a few samples;
                       when the belt moves on, it passes and is counted then.

Stream coordinates are in model samples: one sample = 1 / model_rate_hz seconds.
"""

import math

import numpy as np

from ..data import pixels_to_signal

TRANSFORM_VERSION = 1


def render_window(values: np.ndarray, transform: dict) -> tuple[np.ndarray, float]:
    """[sensors, W] decimated ADC values -> float32 pixels as in the training frames, and the window peak."""
    excess = values - np.median(values)
    peak = float(excess.max())
    if peak <= 0:
        return np.zeros(values.shape, dtype=np.float32), peak
    low, high = transform["clip"]
    pixels = np.clip(np.round(excess * (high / peak)), low, high)
    return pixels.astype(np.float32), peak


def decimate(values: np.ndarray, factor: int) -> np.ndarray:
    """[sensors, n·factor] -> [sensors, n] block means."""
    sensors, length = values.shape
    return values.reshape(sensors, length // factor, factor).mean(axis=2)


# A rate ratio this close to a whole number is that number: exact block means.
WHOLE_RATIO = 1e-9


def polls_per_sample(poll_rate_hz: float, transform: dict) -> int | float:
    """How many polls of a recording make one model sample: a whole number where possible."""
    model_rate = transform["poll_rate_hz"] / transform["decimate"]
    factor = poll_rate_hz / model_rate
    if factor < 1 - WHOLE_RATIO:
        raise ValueError(f"Poll rate {poll_rate_hz} Hz is below the model's sample rate {model_rate} Hz")
    return round(factor) if abs(factor - round(factor)) < WHOLE_RATIO else factor


class FractionalBins:
    """Means of the polls over consecutive intervals of `factor` polls, factor not whole.

    Poll i covers [i, i + 1); sample j is the mean over [j·factor, (j + 1)·factor),
    each poll weighted by the part of it inside. Samples whose interval is not fully
    inside the segment are dropped.
    """

    def __init__(self, factor: float):
        self.factor = factor

    def start_segment(self, raw_start: int) -> int:
        self.next = math.ceil(raw_start / self.factor - WHOLE_RATIO)
        self.buffer, self.buffer_start = None, raw_start
        return self.next

    def push(self, raw_start: int, values: np.ndarray) -> tuple[int, np.ndarray]:
        if self.buffer is None:
            self.buffer, self.buffer_start = values, raw_start
        else:
            if raw_start != self.buffer_start + self.buffer.shape[1]:
                raise ValueError(f"Polls at {raw_start} do not continue the segment")
            self.buffer = np.concatenate((self.buffer, values), axis=1)
        end = self.buffer_start + self.buffer.shape[1]
        last = math.floor(end / self.factor + WHOLE_RATIO)      # samples j < last fit
        first = self.next
        if last <= first:
            return first, np.empty((self.buffer.shape[0], 0))
        cumulative = np.concatenate((np.zeros((self.buffer.shape[0], 1)), np.cumsum(self.buffer, axis=1)), axis=1)
        edges = np.arange(first, last + 1) * self.factor - self.buffer_start
        whole = np.minimum(np.floor(edges).astype(np.int64), self.buffer.shape[1] - 1)
        part = edges - whole
        integral = cumulative[:, whole] + part * self.buffer[:, whole]
        samples = np.diff(integral, axis=1) / self.factor
        self.next = last
        keep = math.floor(last * self.factor + WHOLE_RATIO) - self.buffer_start
        self.buffer = self.buffer[:, keep:]
        self.buffer_start += keep
        return first, samples


def input_range(passport: dict, kind: str) -> tuple:
    """Values a CSV of this kind may hold: ADC codes or stored pixel values."""
    if kind == "raw_adc" and passport.get("recording_transform"):
        return tuple(passport["recording_transform"]["adc_range"])
    return tuple(passport["quantize"]["range"])


class Preprocessor:
    """Stream stage (`start_segment`, `push`) and window stage (`window`) of one recording."""

    def __init__(self, passport: dict, data_settings: dict, windows_settings: dict, poll_rate_hz: float | None = None):
        self.kind = data_settings["input_kind"]
        self.scale = passport["normalize"]["scale"]
        self.offset = passport["normalize"]["offset"]
        self.transform = None
        self.factor = 1
        self.bins = None
        self.empty_peak = None
        if self.kind == "raw_adc":
            self.transform = passport.get("recording_transform")
            if self.transform is None:
                raise ValueError(
                    "data.input_kind raw_adc needs the recording transform in the model passport: run "
                    "tools/locate_frames.py on the recordings and the training dataset, then delete the "
                    "passport file next to the checkpoint so that it is rebuilt"
                )
            if self.transform["version"] != TRANSFORM_VERSION:
                raise ValueError(f"Recording transform version {self.transform['version']}; expected {TRANSFORM_VERSION}")
            if poll_rate_hz is None:
                raise ValueError("data.input_kind raw_adc needs the poll rate: set data.sample_rate_hz")
            self.factor = polls_per_sample(poll_rate_hz, self.transform)
            if isinstance(self.factor, float):
                self.bins = FractionalBins(self.factor)
            self.empty_peak = windows_settings["empty_peak"] * self.transform["min_peak"]
            self.model_rate_hz = self.transform["poll_rate_hz"] / self.transform["decimate"]
        else:
            self.model_rate_hz = poll_rate_hz

    def describe(self) -> str:
        if self.kind != "raw_adc":
            return "Input: stored pixels"
        factor = f"{self.factor:.4g}" if self.bins else str(self.factor)
        return (f"Input: raw ADC; mean of {factor} polls per sample; windows with a peak below "
                f"{self.empty_peak:.1f} ADC units are empty")

    def start_segment(self, raw_start: int) -> int:
        """Model sample at which a segment starting at poll `raw_start` begins: the first whole block."""
        if self.bins is not None:
            return self.bins.start_segment(raw_start)
        # Polls not yet turned into samples start at carry_start (a multiple of factor).
        self.carry_start = math.ceil(raw_start / self.factor) * self.factor
        self.carry = None
        return self.carry_start // self.factor

    def push(self, raw_start: int, values: np.ndarray) -> tuple[int, np.ndarray]:
        """Polls [raw_start, raw_start + n) of the segment -> (first model sample, [sensors, m] samples).

        Polls before the first whole block of the segment are dropped, and so is an
        incomplete block at its end: a sample is always the mean of `factor` polls.
        """
        if self.factor == 1:
            return raw_start, values
        if self.bins is not None:
            return self.bins.push(raw_start, values)
        if self.carry is None:
            values = values[:, max(self.carry_start - raw_start, 0):]
            if values.shape[1] == 0:
                return self.carry_start // self.factor, values
        else:
            if raw_start != self.carry_start + self.carry.shape[1]:
                raise ValueError(f"Polls at {raw_start} do not continue the segment")
            values = np.concatenate((self.carry, values), axis=1)
        whole = values.shape[1] // self.factor * self.factor
        first = self.carry_start // self.factor
        self.carry = values[:, whole:]
        self.carry_start += whole
        return first, decimate(values[:, :whole], self.factor)

    def window(self, values: np.ndarray) -> tuple[np.ndarray, str | None, float | None]:
        """A window of stream samples -> (float32 model input, why it is skipped or None, peak)."""
        if self.kind != "raw_adc":
            return pixels_to_signal(values.astype(np.float32), self.scale, self.offset), None, None
        pixels, peak = render_window(values, self.transform)
        skip = None
        if peak < self.empty_peak:
            skip = "empty"
        elif (values.min(axis=1) - np.median(values)).max() >= self.empty_peak:
            skip = "stationary"
        return pixels_to_signal(pixels, self.scale, self.offset), skip, peak
