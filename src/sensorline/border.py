"""Boxes that the window edge cut: the rule behind the border_ignore option.

A frame is a window over a longer recording, and an object at its first or last time
sample is seen only in part. When the box of such a stub is a few samples long, whether
it is annotated or detected is nearly a coin flip, and the object appears whole in a
neighbouring window anyway. `border_ignore` of the evaluation and of the loss lets those
boxes stay out of the metrics and out of the training signal.

Only the time edges count. The sensor line has no neighbouring window: an object at its
end is cut for good, so a miss there is a real miss.
"""

import numpy as np

# How close to the edge a box must come to count as cut, in samples. The tracker
# calls a box this close to a window edge cut as well (tracker.edge_tol).
TOUCH_TOLERANCE = 1.0


def cut_by_time_border(boxes, time_steps: int, max_length: float) -> np.ndarray:
    """Mask of the boxes that touch a time edge of the frame and are shorter than max_length samples.

    `boxes` are [N, 4] in the signal layout (time0, sensor0, time1, sensor1), as fractions of
    the frame of `time_steps` samples. A `max_length` of 0 selects nothing.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if max_length <= 0:
        return np.zeros(len(boxes), dtype=bool)
    start, end = boxes[:, 0] * time_steps, boxes[:, 2] * time_steps
    touches = (start <= TOUCH_TOLERANCE) | (end >= time_steps - TOUCH_TOLERANCE)
    return touches & (end - start < max_length)
