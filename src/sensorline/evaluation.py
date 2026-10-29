"""Decode boxes, match detections, compute metrics and export predictions."""

import numpy as np
import torch
from torchvision.ops import nms

from .batching import forward_groups
from .border import cut_by_time_border
from .data import image_boxes
from .losses import pairwise_iou


def prepare_candidates(boxes, scores, anchors, time_ratio, config):
    """Prepare clipped original-frame candidates before NMS, preserving score ties."""
    keep = (scores >= config["ap_min_confidence"]) & (anchors[:, 0] < time_ratio)
    boxes = boxes[keep].clone()
    scores = scores[keep]

    # Undo time padding and clip to the original sample's normalized frame.
    boxes[:, [0, 2]] /= time_ratio
    boxes.clamp_(0, 1)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    boxes, scores = boxes[valid], scores[valid]
    if len(scores) > config["pre_nms_topk"]:
        top_indices = scores.topk(config["pre_nms_topk"]).indices
        boxes, scores = boxes[top_indices], scores[top_indices]

    return {"boxes": boxes, "scores": scores}


def suppress_sample(candidates, nms_iou, max_detections):
    """The highest-scoring candidates left after removing boxes that overlap a better one more than nms_iou."""
    selected = nms(candidates["boxes"], candidates["scores"], nms_iou)[:max_detections]
    return {name: values[selected] for name, values in candidates.items()}


@torch.no_grad()
def batch_candidates(output, targets, image_shape, config):
    """Pre-NMS candidates of every sample in one forward output, on CPU."""
    boxes = output["boxes"].detach().cpu()
    scores = output["objectness"].sigmoid().detach().cpu()
    anchors = output["anchors"].detach().cpu()
    candidates = []
    for index, target in enumerate(targets):
        time_ratio = target["size"][1] / image_shape[1]
        candidates.append(
            prepare_candidates(boxes[index], scores[index], anchors, time_ratio, config)
        )
    return candidates


def decode_predictions(output, targets, image_shape, config):
    return [
        suppress_sample(candidates, config["nms_iou"], config["max_detections"])
        for candidates in batch_candidates(output, targets, image_shape, config)
    ]


def prepare_matching(prediction, target_boxes):
    """Compute score order and IoUs once for reuse across AP thresholds."""
    order = prediction["scores"].argsort(descending=True)
    overlaps = pairwise_iou(prediction["boxes"][order], target_boxes).numpy()
    return prediction["scores"][order].numpy(), overlaps


def matched_target_indices(overlaps, threshold):
    """Greedy one-to-one matching of IoU rows already ordered by confidence."""
    used_targets = np.zeros(overlaps.shape[1], dtype=bool)
    matched_targets = np.full(len(overlaps), -1, dtype=int)
    for prediction_index, row in enumerate(overlaps):
        candidates = np.flatnonzero(~used_targets & (row >= threshold))
        if len(candidates):
            target_index = candidates[np.argmax(row[candidates])]
            used_targets[target_index] = True
            matched_targets[prediction_index] = target_index
    return matched_targets


def match_overlaps(overlaps, threshold):
    return matched_target_indices(overlaps, threshold) >= 0


IGNORED = -2


def prepare_ignoring_matching(prediction, target, border):
    """Score order, IoUs and the cut boxes on both sides, for a `border_ignore` evaluation.

    A box is cut when it touches a time edge of the frame and is shorter than `border`
    samples (see border.py). Predictions are in score order, as in prepare_matching.
    """
    order = prediction["scores"].argsort(descending=True)
    boxes = prediction["boxes"][order]
    time_steps = target["size"][1]
    overlaps = pairwise_iou(boxes, target["boxes"]).numpy()
    return (
        prediction["scores"][order].numpy(), overlaps,
        cut_by_time_border(target["boxes"], time_steps, border),
        cut_by_time_border(boxes, time_steps, border),
    )


def matched_ignoring_cut(overlaps, threshold, cut_targets, cut_predictions):
    """Greedy one-to-one matching in which cut boxes count neither for nor against the model.

    Per prediction in score order: the index of the target it matches, -1 for a false
    positive and IGNORED for a detection that is set aside. A detection prefers an uncut
    target; failing that, it is set aside when it overlaps a cut target or is cut itself.
    A cut target that nothing detects is not a miss either (the caller leaves it out).
    """
    used_targets = np.zeros(overlaps.shape[1], dtype=bool)
    matched_targets = np.full(len(overlaps), -1, dtype=int)
    for prediction_index, row in enumerate(overlaps):
        close = row >= threshold
        candidates = np.flatnonzero(~used_targets & ~cut_targets & close)
        if len(candidates):
            target_index = candidates[np.argmax(row[candidates])]
            used_targets[target_index] = True
            matched_targets[prediction_index] = target_index
            continue
        cut_candidates = np.flatnonzero(~used_targets & cut_targets & close)
        if len(cut_candidates):
            used_targets[cut_candidates[np.argmax(row[cut_candidates])]] = True
        if len(cut_candidates) or cut_predictions[prediction_index]:
            matched_targets[prediction_index] = IGNORED
    return matched_targets


def average_precision(scores, true_positive, total_targets):
    if total_targets == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    cumulative_hits = np.cumsum(true_positive[order])
    recall = cumulative_hits / total_targets
    precision = cumulative_hits / np.arange(1, len(scores) + 1)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

    interpolated_precision = []
    for recall_level in np.linspace(0, 1, 101):
        candidates = np.flatnonzero(recall >= recall_level)
        value = 0.0
        if len(candidates):
            value = float(precision_envelope[candidates[0]])
        interpolated_precision.append(value)
    return float(np.mean(interpolated_precision))


def average_precisions_ignoring_cut(predictions, targets, border) -> list[float]:
    """average_precisions with the cut boxes left out of the targets and of the detections."""
    prepared = [
        prepare_ignoring_matching(prediction, target, border)
        for prediction, target in zip(predictions, targets)
    ]
    total_targets = sum(int((~cut_targets).sum()) for _, _, cut_targets, _ in prepared)
    values = []
    for threshold in np.linspace(0.5, 0.95, 10):
        kept_scores, hits = [], []
        for scores, overlaps, cut_targets, cut_predictions in prepared:
            matches = matched_ignoring_cut(overlaps, float(threshold), cut_targets, cut_predictions)
            kept = matches != IGNORED
            kept_scores.append(scores[kept])
            hits.append(matches[kept] >= 0)
        values.append(average_precision(np.concatenate(kept_scores), np.concatenate(hits), total_targets))
    return values


def average_precisions(predictions, targets, border=0) -> list[float]:
    if border:
        return average_precisions_ignoring_cut(predictions, targets, border)
    total_targets = sum(len(target["boxes"]) for target in targets)
    prepared = [
        prepare_matching(prediction, target["boxes"])
        for prediction, target in zip(predictions, targets)
    ]
    scores = np.concatenate([sample_scores for sample_scores, _ in prepared])
    values = []
    for threshold in np.linspace(0.5, 0.95, 10):
        hits = [match_overlaps(overlaps, float(threshold)) for _, overlaps in prepared]
        values.append(average_precision(scores, np.concatenate(hits), total_targets))
    return values


def sample_row(target, predicted_count: int, true_positives: int, ignored=None) -> dict:
    """The counts of one frame; `ignored` is (cut targets, cut detections) of a border_ignore evaluation."""
    true_count = len(target["boxes"]) - (ignored[0] if ignored else 0)
    row = {
        "id": target["id"],
        "true_count": true_count,
        "predicted_count": predicted_count,
        "count_error": predicted_count - true_count,
        "tp": true_positives,
        "fp": predicted_count - true_positives,
        "fn": true_count - true_positives,
    }
    if ignored:
        row["ignored_objects"], row["ignored_detections"] = ignored
    return row


def sample_detection_metrics(prediction, target, confidence, match_iou, border=0) -> dict:
    keep = prediction["scores"] >= confidence
    accepted = {
        "boxes": prediction["boxes"][keep],
        "scores": prediction["scores"][keep],
    }
    if border:
        _, overlaps, cut_targets, cut_predictions = prepare_ignoring_matching(accepted, target, border)
        matches = matched_ignoring_cut(overlaps, match_iou, cut_targets, cut_predictions)
        ignored = int((matches == IGNORED).sum())
        return sample_row(
            target, len(accepted["boxes"]) - ignored, int((matches >= 0).sum()),
            ignored=(int(cut_targets.sum()), ignored),
        )
    _, overlaps = prepare_matching(accepted, target["boxes"])
    matches = match_overlaps(overlaps, match_iou)
    return sample_row(target, len(accepted["boxes"]), int(matches.sum()))


def summarize_metrics(rows, ap_values=None) -> dict:
    """Aggregate sample rows; AP columns are omitted when ap_values is None."""
    true_positives = sum(row["tp"] for row in rows)
    false_positives = sum(row["fp"] for row in rows)
    false_negatives = sum(row["fn"] for row in rows)
    errors = np.asarray([row["count_error"] for row in rows])
    summary = {
        "samples": len(rows),
        "objects": sum(row["true_count"] for row in rows),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": true_positives / max(true_positives + false_positives, 1),
        "recall": true_positives / max(true_positives + false_negatives, 1),
        "f1": 2
        * true_positives
        / max(2 * true_positives + false_positives + false_negatives, 1),
    }
    if ap_values is not None:
        summary["map50"] = ap_values[0]
        summary["map75"] = ap_values[5]
        summary["map50_95"] = float(np.mean(ap_values))
    summary["count_mae"] = float(np.abs(errors).mean())
    summary["count_rmse"] = float(np.sqrt(np.mean(np.square(errors))))
    summary["count_bias"] = float(np.mean(errors))
    summary["exact_count_accuracy"] = float(np.mean(errors == 0))
    if rows and "ignored_objects" in rows[0]:
        summary["ignored_objects"] = sum(row["ignored_objects"] for row in rows)
        summary["ignored_detections"] = sum(row["ignored_detections"] for row in rows)
    return summary


def detection_metrics(predictions, targets, confidence, match_iou, border=0):
    ap_values = average_precisions(predictions, targets, border)
    rows = []
    for prediction, target in zip(predictions, targets):
        rows.append(sample_detection_metrics(prediction, target, confidence, match_iou, border))
    return summarize_metrics(rows, ap_values), rows


def confidence_metrics(predictions, targets, confidences, match_iou, border=0):
    """Reuse greedy matches of score prefixes; retain evaluator ordering on ties."""
    rows_by_confidence = {confidence: [] for confidence in confidences}
    for prediction, target in zip(predictions, targets):
        if border:
            scores, overlaps, cut_targets, cut_predictions = prepare_ignoring_matching(prediction, target, border)
            matches = matched_ignoring_cut(overlaps, match_iou, cut_targets, cut_predictions)
            hits = np.concatenate(([0], np.cumsum(matches >= 0)))
            ignored_counts = np.concatenate(([0], np.cumsum(matches == IGNORED)))
        else:
            scores, overlaps = prepare_matching(prediction, target["boxes"])
            hits = np.concatenate(([0], np.cumsum(match_overlaps(overlaps, match_iou))))
        has_ties = len(np.unique(scores)) != len(scores)
        for confidence in confidences:
            if has_ties:
                # torch.argsort can reorder equal scores when the array is shortened.
                # Match the accepted array exactly as the ordinary evaluator does.
                row = sample_detection_metrics(prediction, target, confidence, match_iou, border)
            else:
                accepted = int((prediction["scores"] >= confidence).sum())
                if border:
                    ignored = int(ignored_counts[accepted])
                    row = sample_row(target, accepted - ignored, int(hits[accepted]),
                                     ignored=(int(cut_targets.sum()), ignored))
                else:
                    row = sample_row(target, accepted, int(hits[accepted]))
            rows_by_confidence[confidence].append(row)
    return {confidence: summarize_metrics(rows) for confidence, rows in rows_by_confidence.items()}


def evaluation_settings(evaluation: dict, data: dict) -> dict:
    """The evaluation section of a run plus the sensor axis that the exported image boxes need."""
    return {**evaluation, "sensor_axis": data["sensor_axis"]}


def serialize_predictions(predictions, targets, config) -> list[dict]:
    serialized = []
    for prediction, target in zip(predictions, targets):
        keep = prediction["scores"] >= config["confidence"]
        boxes = prediction["boxes"][keep]
        serialized.append(
            {
                "id": target["id"],
                "count": len(boxes),
                "boxes_signal_xyxy": boxes.tolist(),
                "boxes_image_xyxy": image_boxes(boxes, config["sensor_axis"]).tolist(),
                "scores": prediction["scores"][keep].tolist(),
                "labels": [0] * len(boxes),
                "target_boxes_signal_xyxy": target["boxes"].tolist(),
                "size_sensor_time": list(target["size"]),
                "all_boxes_signal_xyxy": prediction["boxes"].tolist(),
                "all_scores": prediction["scores"].tolist(),
            }
        )
    return serialized


@torch.no_grad()
def evaluate(model, loader, device, config, criterion=None, labeled=True):
    model.eval()
    predictions, targets = [], []
    loss_sums = {"total": 0.0, "bbox": 0.0, "objectness": 0.0}
    for signal, batch_targets in loader:
        signal = signal.to(device)
        batch_predictions = [None] * len(batch_targets)
        for indices, output, group_targets, image_shape in forward_groups(
            model, signal, batch_targets
        ):
            if criterion is not None:
                losses = criterion(output, group_targets, image_shape)
                for name in loss_sums:
                    loss_sums[name] += float(losses[name]) * len(group_targets)
            group_predictions = decode_predictions(
                output, group_targets, image_shape, config
            )
            for index, prediction in zip(indices, group_predictions):
                batch_predictions[index] = prediction
        predictions.extend(batch_predictions)
        targets.extend(batch_targets)

    serialized = serialize_predictions(predictions, targets, config)
    if labeled:
        metrics, rows = detection_metrics(
            predictions, targets, config["confidence"], config["match_iou"],
            config.get("border_ignore", 0),
        )
    else:
        # Without labels a frame has only its count.
        metrics = {}
        rows = [{"id": row["id"], "predicted_count": row["count"]} for row in serialized]
    if criterion is not None:
        for name, total in loss_sums.items():
            metrics[f"loss_{name}"] = total / len(targets)
    return metrics, rows, serialized
