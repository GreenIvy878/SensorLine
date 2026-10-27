"""Small numerical diagnostics. Matrices here are always [sensor, time]."""

import numpy as np


PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


class Distribution:
    """Exact moments/extrema plus a bounded uniform priority sample for quantiles."""

    def __init__(self, rng, capacity=4096):
        self.rng = rng
        self.capacity = capacity
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.values = np.empty(0)
        self.keys = np.empty(0)

    def add(self, values):
        values = np.asarray(values, dtype=float).ravel()
        if not values.size:
            return
        mean = float(values.mean())
        delta = mean - self.mean
        total = self.count + values.size
        self.m2 += float(np.square(values - mean).sum()) + delta**2 * self.count * values.size / total
        self.mean += delta * values.size / total
        self.count = total
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        keys = np.concatenate((self.keys, self.rng.random(values.size)))
        retained = np.concatenate((self.values, values))
        if retained.size > self.capacity:
            indices = np.argpartition(keys, -self.capacity)[-self.capacity:]
            keys, retained = keys[indices], retained[indices]
        self.keys, self.values = keys, retained

    def summary(self):
        if not self.count:
            return {"count": 0, "mean": None, "median": None, "std": None,
                    "min": None, "max": None, "dynamic_range": None, "percentiles": {},
                    "quantile_sample_size": 0, "histogram": {"edges": [], "counts": []}}
        quantiles = np.percentile(self.values, PERCENTILES)
        counts, edges = np.histogram(self.values, bins=40, range=(self.minimum, self.maximum)
                                     if self.maximum > self.minimum else None)
        return {"count": int(self.count), "mean": self.mean, "median": float(np.median(self.values)),
                "std": float(np.sqrt(self.m2 / self.count)), "min": self.minimum,
                "max": self.maximum, "dynamic_range": self.maximum - self.minimum,
                "percentiles": dict(zip(map(str, PERCENTILES), quantiles.tolist())),
                "quantile_sample_size": len(self.values),
                "histogram": {"edges": edges.tolist(), "counts": counts.tolist()}}


def bbox_geometry(boxes, shape):
    """Internal normalized xyxy -> original cell units, without rounding boxes."""
    sensors, time = shape
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    cells = boxes * [time, sensors, time, sensors]
    lengths = cells[:, 2] - cells[:, 0]
    widths = cells[:, 3] - cells[:, 1]
    return {"sensor_width": widths, "temporal_length": lengths, "area": widths * lengths,
            "aspect_ratio": widths / lengths,
            "sensor_center": (cells[:, 1] + cells[:, 3]) / 2,
            "time_center": (cells[:, 0] + cells[:, 2]) / 2,
            "boundary_distance": np.minimum.reduce(
                [cells[:, 0], cells[:, 1], time - cells[:, 2], sensors - cells[:, 3]])}


def foreground_mask(shape, boxes):
    """Pixel-center membership in the union of boxes; boxes are approximate FG."""
    sensors, time = shape
    sensor = (np.arange(sensors) + .5)[:, None] / sensors
    temporal = (np.arange(time) + .5)[None, :] / time
    mask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        mask |= (temporal >= x1) & (temporal < x2) & (sensor >= y1) & (sensor < y2)
    return mask


def gradients(matrix):
    return np.diff(matrix, axis=0), np.diff(matrix, axis=1)


class SensorCorrelation:
    """Streaming Pearson covariance across all time columns, preserving offsets."""

    def __init__(self, sensors):
        self.count = 0
        self.mean = np.zeros(sensors)
        self.cross = np.zeros((sensors, sensors))

    def add(self, matrix):
        count = matrix.shape[1]
        mean = matrix.mean(axis=1)
        centered = matrix - mean[:, None]
        delta = mean - self.mean
        total = self.count + count
        self.cross += centered @ centered.T + np.outer(delta, delta) * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def result(self):
        std = np.sqrt(np.maximum(0, self.cross.diagonal()))
        denominator = np.outer(std, std)
        result = np.full_like(self.cross, np.nan)
        np.divide(self.cross, denominator, out=result, where=denominator > 0)
        return np.clip(result, -1, 1)


def correlation_length(curve):
    """First measured distance <= 1/e; None means undefined or right-censored."""
    curve = np.asarray(curve, dtype=float)
    valid = np.flatnonzero(np.isfinite(curve[1:]) & (curve[1:] <= np.exp(-1))) + 1
    return int(valid[0]) if valid.size else None


def sensor_noise(matrix):
    """Noise scale of each sensor from second differences along time; NaN if too short."""
    second = np.diff(matrix, n=2, axis=1)
    if not second.shape[1]:
        return np.full(matrix.shape[0], np.nan)
    # Gaussian-equivalent scale, also responds to real sharp object boundaries.
    noise = np.median(np.abs(second - np.median(second, axis=1, keepdims=True)), axis=1)
    return noise / (0.67448975 * np.sqrt(6))


def robust_scores(values):
    values = np.asarray(values, dtype=float)
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    scale = 1.4826 * mad
    if scale == 0:
        q25, q75 = np.percentile(values, [25, 75])
        scale = (q75 - q25) / 1.349
    if scale == 0:
        # Equal peers with a different member are diagnostically significant.
        return np.where(values == center, 0.0, np.where(values > center, np.inf, -np.inf))
    return (values - center) / scale


def sensor_diagnostics(sensors, threshold=3.5):
    findings = []
    for feature in ("mean", "std", "noise"):
        values = [row[feature] for row in sensors]
        if any(value is None for value in values):
            continue
        for index, score in enumerate(robust_scores(values)):
            if abs(score) > threshold:
                findings.append({"sensor": index, "feature": feature, "value": values[index],
                                 "peer_median": float(np.median(values)),
                                 "robust_score": float(score) if np.isfinite(score) else None,
                                 "zero_peer_spread": not bool(np.isfinite(score))})
    for row in sensors:
        if row["dynamic_range"] == 0:
            findings.append({"sensor": row["sensor"], "feature": "constant",
                             "value": row["mean"], "peer_median": None, "robust_score": None})
    return findings


def compare_distributions(train, val):
    """Empirical KS, Wasserstein-1 and standardized mean difference, no p-values."""
    train, val = np.sort(train), np.sort(val)
    if not len(train) or not len(val):
        return {"train": None, "val": None, "ks": None, "wasserstein": None,
                "normalized_mean_difference": None, "large_difference": False}
    support = np.unique(np.concatenate((train, val)))
    a = np.searchsorted(train, support, side="right") / len(train)
    b = np.searchsorted(val, support, side="right") / len(val)
    ks = float(np.abs(a - b).max())
    wasserstein = float(np.sum(np.abs(a[:-1] - b[:-1]) * np.diff(support)))
    pooled_std = float(np.sqrt((train.var() + val.var()) / 2))
    difference = float(val.mean() - train.mean())
    normalized = difference / pooled_std if pooled_std else (0.0 if difference == 0 else None)
    large = ks >= .25 or (normalized is not None and abs(normalized) >= .5)
    return {"train": float(train.mean()), "val": float(val.mean()), "ks": ks,
            "wasserstein": wasserstein, "normalized_mean_difference": normalized,
            "large_difference": bool(large)}


def foreground_separation(foreground, background):
    result = compare_distributions(background, foreground)
    result["background_mean"] = result.pop("train")
    result["foreground_mean"] = result.pop("val")
    result["standardized_effect_size"] = result.pop("normalized_mean_difference")
    result.pop("large_difference")
    if not len(foreground) or not len(background):
        return {**result, "median_difference": None, "histogram_overlap": None, "magnitude_auc": None}
    edges = np.histogram_bin_edges(np.concatenate((foreground, background)), bins=40)
    a = np.histogram(foreground, edges)[0] / len(foreground)
    b = np.histogram(background, edges)[0] / len(background)
    sorted_bg = np.sort(background)
    lower = np.searchsorted(sorted_bg, foreground, side="left")
    upper = np.searchsorted(sorted_bg, foreground, side="right")
    auc = float(np.mean((lower + upper) / (2 * len(background))))
    return {**result, "mean_difference": float(np.mean(foreground) - np.mean(background)),
            "median_difference": float(np.median(foreground) - np.median(background)),
            "histogram_overlap": float(np.minimum(a, b).sum()), "magnitude_auc": auc,
            "histograms": {"edges": edges.tolist(), "foreground": a.tolist(), "background": b.tolist()}}


def object_overlap(boxes, shape):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    ious, nearest = [], []
    if len(boxes) > 1:
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 * [shape[1], shape[0]]
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1).tolist()
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        for i in range(len(boxes) - 1):
            others = boxes[i + 1:]
            intersection = np.maximum(0, np.minimum(boxes[i, 2:], others[:, 2:])
                                      - np.maximum(boxes[i, :2], others[:, :2])).prod(axis=1)
            ious.extend((intersection / (areas[i] + areas[i + 1:] - intersection)).tolist())
    return np.asarray(ious), np.asarray(nearest)
