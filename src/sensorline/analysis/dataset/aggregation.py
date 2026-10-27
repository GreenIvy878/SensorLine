"""Streaming split aggregation and the train/validation comparison."""

import numpy as np

from .numerics import (
    Distribution, SensorCorrelation, bbox_geometry, compare_distributions, correlation_length,
    foreground_mask, foreground_separation, gradients, object_overlap, sensor_diagnostics,
    sensor_noise,
)


SCATTER_LIMIT = 2000
# Per-sample scalars that also feed the population distributions.
SAMPLE_FEATURES = ("object_count", "noise")
# Object measurements with their own distribution; area and aspect ratio follow from these.
OBJECT_FEATURES = (
    "sensor_width", "temporal_length", "sensor_center", "time_center", "boundary_distance",
)


def reservoir_insert(reservoir, item, seen, capacity, rng):
    """Keep a uniform sample of a stream; `seen` counts items including this one."""
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    index = int(rng.integers(seen))
    if index < capacity:
        reservoir[index] = item


def mean_or_none(values):
    return float(values.mean()) if values.size else None


class SplitAnalysis:
    """One population's numerical state; images are discarded after each update."""

    def __init__(self, seed, sensors):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.signal = Distribution(rng, 65536)
        self.sensors = [Distribution(rng) for _ in range(sensors)]
        self.sensor_noise = [Distribution(rng, 2048) for _ in range(sensors)]
        self.features = {}
        self.foreground = Distribution(rng, 65536)
        self.background = Distribution(rng, 65536)
        self.correlation = SensorCorrelation(sensors)
        self.samples = []
        self.shapes = {}
        self.scatter = []
        self.scatter_seen = 0

    def feature(self, name, values):
        if name not in self.features:
            self.features[name] = Distribution(self.rng)
        self.features[name].add(values)

    # The update order below fixes the random quantile samples; keep it stable.
    def add(self, matrix, boxes, identifier, split):
        mask = foreground_mask(matrix.shape, boxes)
        noise = sensor_noise(matrix)
        self.add_signal(matrix, mask, noise)
        spatial, temporal = gradients(matrix)
        ious = self.add_objects(matrix.shape, boxes, identifier, split)

        row = {"id": identifier, "split": split, "shape": list(matrix.shape),
               "object_count": len(boxes), "mean": float(matrix.mean()), "std": float(matrix.std()),
               "foreground_level": mean_or_none(matrix[mask]),
               "background_level": mean_or_none(matrix[~mask]),
               "spatial_gradient": float(np.abs(spatial).mean()),
               "temporal_gradient": mean_or_none(np.abs(temporal)),
               "noise": float(np.nanmean(noise)) if np.isfinite(noise).any() else None,
               "max_iou": float(ious.max()) if ious.size else 0.0,
               "pair_count": int(ious.size), "overlapping_pairs": int((ious > 0).sum())}
        self.samples.append(row)
        shape = str(list(matrix.shape))
        self.shapes[shape] = self.shapes.get(shape, 0) + 1
        for name in SAMPLE_FEATURES:
            if row[name] is not None:
                self.feature(name, [row[name]])

    def add_signal(self, matrix, mask, sensor_noise):
        self.signal.add(matrix)
        self.foreground.add(matrix[mask])
        self.background.add(matrix[~mask])
        for index, trace in enumerate(matrix):
            self.sensors[index].add(trace)
            if np.isfinite(sensor_noise[index]):
                self.sensor_noise[index].add([sensor_noise[index]])
        self.correlation.add(matrix)

    def add_objects(self, shape, boxes, identifier, split):
        geometry = bbox_geometry(boxes, shape)
        for name in OBJECT_FEATURES:
            self.feature(name, geometry[name])
        for width, length in zip(geometry["sensor_width"], geometry["temporal_length"]):
            point = {"sensor_width": float(width), "temporal_length": float(length),
                     "id": identifier, "split": split}
            self.scatter_seen += 1
            reservoir_insert(self.scatter, point, self.scatter_seen, SCATTER_LIMIT, self.rng)
        ious, nearest = object_overlap(boxes, shape)
        self.feature("pair_iou", ious)
        self.feature("nearest_center_distance", nearest)
        return ious

    def sensor_rows(self):
        return [
            {"sensor": index, **distribution.summary(),
             "noise": self.sensor_noise[index].summary()["mean"]}
            for index, distribution in enumerate(self.sensors)
        ]

    def spatial_summary(self):
        correlation = self.correlation.result()
        by_distance, pair_counts = [], []
        for lag in range(len(self.sensors)):
            diagonal = np.diag(correlation, lag)
            finite = diagonal[np.isfinite(diagonal)]
            by_distance.append(float(finite.mean()) if finite.size else None)
            pair_counts.append(int(finite.size))
        length = correlation_length([np.nan if value is None else value for value in by_distance])
        return {"correlation_matrix": correlation.tolist(), "correlation_vs_distance": by_distance,
                "valid_pairs": pair_counts, "correlation_length": length,
                "censored_or_undefined": length is None}

    def result(self):
        sensors = self.sensor_rows()
        rows = self.samples
        count = len(rows)
        empty = sum(row["object_count"] == 0 for row in rows)
        multiple = sum(row["object_count"] > 1 for row in rows)
        pairs = sum(row["pair_count"] for row in rows)
        overlaps = sum(row["overlapping_pairs"] for row in rows)
        return {
            "overview": {"samples": count, "objects": sum(row["object_count"] for row in rows),
                         "empty_samples": empty, "empty_fraction": empty / count if count else None,
                         "multiple_object_fraction": multiple / count if count else None,
                         "shapes_sensor_time": self.shapes, "signal": self.signal.summary()},
            "sensors": sensors, "sensor_diagnostics": sensor_diagnostics(sensors) if count else [],
            "spatial": self.spatial_summary(),
            "distributions": {name: distribution.summary() for name, distribution in self.features.items()},
            "objects": {"scatter": self.scatter, "scatter_population": self.scatter_seen,
                        "pair_count": pairs, "overlapping_pairs": overlaps,
                        "overlap_pair_fraction": overlaps / pairs if pairs else None,
                        "samples_with_overlap": sum(row["max_iou"] > 0 for row in rows)},
            "foreground_background": {"foreground": self.foreground.summary(), "background": self.background.summary(),
                                      "separation": foreground_separation(self.foreground.values, self.background.values)},
            "samples": rows,
        }


def dataset_drift(train, val):
    """Distances between the train and validation distributions of each feature."""
    rows = []
    for feature in ("mean", "std", "background_level", "foreground_level", "object_count",
                    "spatial_gradient", "temporal_gradient", "noise"):
        a = [row[feature] for row in train.samples if row[feature] is not None]
        b = [row[feature] for row in val.samples if row[feature] is not None]
        rows.append({"feature": feature, "grain": "sample", **compare_distributions(a, b)})
    for feature in ("sensor_width", "temporal_length"):
        a = train.features.get(feature)
        b = val.features.get(feature)
        rows.append({"feature": feature, "grain": "object (bounded sample)",
                     **compare_distributions(a.values if a else [], b.values if b else [])})
    for index, (a, b) in enumerate(zip(train.sensors, val.sensors)):
        rows.append({"feature": f"sensor_{index}", "grain": "pixel (bounded sample)",
                     **compare_distributions(a.values, b.values)})
    return {
        "thresholds": {
            "ks": 0.25,
            "absolute_normalized_mean_difference": 0.5,
            "rule": "flag if either threshold is reached; descriptive screening, no p-values",
        },
        "rows": rows,
    }
