"""TP/FP/FN classification of saved predictions, for the frame viewer of the Inference section."""

import torch

from ..evaluation import matched_target_indices, prepare_matching


def box_tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


def classify_detections(prediction, targets, match_iou):
    """Accepted detections in confidence order, and the indices of the matched targets."""
    ordered_scores, overlaps = prepare_matching(prediction, targets)
    matches = matched_target_indices(overlaps, match_iou)
    order = prediction["scores"].argsort(descending=True)
    ordered_boxes = prediction["boxes"][order].tolist()
    detections = [
        {"box": box, "score": float(score), "classification": "tp" if matched >= 0 else "fp"}
        for box, score, matched in zip(ordered_boxes, ordered_scores, matches)
    ]
    return detections, {int(matched) for matched in matches if matched >= 0}


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
    detections, matched = classify_detections(
        {"boxes": boxes[keep], "scores": scores[keep]}, targets, evaluation["match_iou"]
    )
    return {
        "id": row["id"],
        "tp": len(matched),
        "fp": len(detections) - len(matched),
        "fn": len(targets) - len(matched),
        "objects": [
            {"box": box, "detected": index in matched}
            for index, box in enumerate(targets.tolist())
        ],
        "detections": detections,
    }
