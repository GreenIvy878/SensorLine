"""TP/FP/FN classification of saved predictions, for the frame viewer of the Inference section."""

import torch

from ..evaluation import (
    IGNORED, matched_ignoring_cut, matched_target_indices, prepare_ignoring_matching, prepare_matching,
)


def box_tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def classify_detections(prediction, targets, match_iou, time_steps=None, border=0):
    """Accepted detections in confidence order, and the indices of the matched targets.

    With `border` (the border_ignore option, see border.py) a detection that the metrics
    set aside is classified "ignored", and `time_steps` is the length of the frame.
    Also returns the mask of the targets that are cut.
    """
    if border:
        ordered_scores, overlaps, cut_targets, cut_predictions = prepare_ignoring_matching(
            prediction, {"boxes": targets, "size": (None, time_steps)}, border
        )
        matches = matched_ignoring_cut(overlaps, match_iou, cut_targets, cut_predictions)
    else:
        ordered_scores, overlaps = prepare_matching(prediction, targets)
        matches = matched_target_indices(overlaps, match_iou)
        cut_targets = None
    order = prediction["scores"].argsort(descending=True)
    ordered_boxes = prediction["boxes"][order].tolist()
    detections = [
        {"box": box, "score": float(score),
         "classification": "ignored" if matched == IGNORED else "tp" if matched >= 0 else "fp"}
        for box, score, matched in zip(ordered_boxes, ordered_scores, matches)
    ]
    return detections, {int(matched) for matched in matches if matched >= 0}, cut_targets


def below_threshold_detections(row, evaluation):
    """Post-NMS candidates between ap_min_confidence and confidence."""
    return [
        {"box": box, "score": score, "classification": "below_threshold"}
        for box, score in zip(row["all_boxes_signal_xyxy"], row["all_scores"])
        if evaluation["ap_min_confidence"] <= score < evaluation["confidence"]
    ]


def classify_sample(row, evaluation):
    """Counts, annotated objects and accepted detections of one saved prediction row."""
    boxes = box_tensor(row["boxes_signal_xyxy"])
    scores = torch.tensor(row["scores"], dtype=torch.float32)
    targets = box_tensor(row["target_boxes_signal_xyxy"])
    keep = scores >= evaluation["confidence"]
    border = evaluation.get("border_ignore", 0)
    detections, matched, cut = classify_detections(
        {"boxes": boxes[keep], "scores": scores[keep]}, targets, evaluation["match_iou"],
        row["size_sensor_time"][1], border,
    )
    if cut is None:
        cut = [False] * len(targets)
    counted = [detection for detection in detections if detection["classification"] != "ignored"]
    return {
        "id": row["id"],
        "tp": len(matched),
        "fp": len(counted) - len(matched),
        "fn": len(targets) - int(sum(cut)) - len(matched),
        "objects": [
            {"box": box, "detected": index in matched, "ignored": bool(cut[index]) and index not in matched}
            for index, box in enumerate(targets.tolist())
        ],
        "detections": detections,
    }
