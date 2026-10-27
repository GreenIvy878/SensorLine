"""Bounding-box geometry, target assignment and detector loss."""

import math
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import BOX_LOSSES, OBJECTNESS_LOSSES, with_defaults


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    intersection_start = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    intersection_end = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (intersection_end - intersection_start).clamp_min(0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp_min(1e-9)


def bbox_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: str = "ciou",
) -> torch.Tensor:
    """Elementwise IoU-family loss for aligned prediction/target pairs."""
    if kind not in BOX_LOSSES:
        raise ValueError(f"Unknown bbox loss: {kind}")

    predicted_sizes = (prediction[:, 2:] - prediction[:, :2]).clamp_min(1e-9)
    target_sizes = (target[:, 2:] - target[:, :2]).clamp_min(1e-9)
    intersection_start = torch.maximum(prediction[:, :2], target[:, :2])
    intersection_end = torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (intersection_end - intersection_start).clamp_min(0).prod(-1)
    union = predicted_sizes.prod(-1) + target_sizes.prod(-1) - intersection
    iou = intersection / union.clamp_min(1e-9)
    if kind == "iou":
        return 1 - iou

    enclosing_start = torch.minimum(prediction[:, :2], target[:, :2])
    enclosing_end = torch.maximum(prediction[:, 2:], target[:, 2:])
    enclosing_sizes = (enclosing_end - enclosing_start).clamp_min(1e-9)
    if kind == "giou":
        enclosing_area = enclosing_sizes.prod(-1)
        return 1 - iou + (enclosing_area - union.clamp_min(1e-9)) / enclosing_area

    predicted_centers = (prediction[:, :2] + prediction[:, 2:]) / 2
    target_centers = (target[:, :2] + target[:, 2:]) / 2
    center_distance = (predicted_centers - target_centers).square().sum(-1)
    distance_penalty = center_distance / enclosing_sizes.square().sum(-1)
    if kind == "diou":
        return 1 - iou + distance_penalty

    return ciou_loss(iou, distance_penalty, predicted_sizes, target_sizes)


def ciou_loss(iou, distance_penalty, predicted_sizes, target_sizes):
    target_angle = torch.atan(target_sizes[:, 0] / target_sizes[:, 1])
    predicted_angle = torch.atan(predicted_sizes[:, 0] / predicted_sizes[:, 1])
    aspect_penalty = 4 / math.pi**2 * (target_angle - predicted_angle).square()
    with torch.no_grad():
        aspect_weight = aspect_penalty / (1 - iou + aspect_penalty).clamp_min(1e-9)
    return 1 - iou + distance_penalty + aspect_weight * aspect_penalty


# A ground-truth box prefers the feature level where it spans about this many cells
# per side; the level preference weighs this much against the distance to the centre.
CELLS_PER_OBJECT_SIDE = 4
SCALE_COST_WEIGHT = 0.2


def assignment_candidates(anchors, cell_sizes, boxes, valid):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    center_offsets = (anchors[None] - centers[:, None]) / cell_sizes[None]
    possible = (np.abs(center_offsets) < 0.999999).all(-1) & valid[None]

    relative_area = sizes.prod(-1)[:, None] / cell_sizes.prod(-1)[None]
    scale_cost = np.abs(np.log(np.sqrt(relative_area) / CELLS_PER_OBJECT_SIDE))
    cost = (center_offsets**2).sum(-1) + SCALE_COST_WEIGHT * scale_cost
    order = np.lexsort((sizes.prod(-1), possible.sum(-1)))
    return possible, cost, order


def reassign_target(target_index, possible, cost, target_cells, cell_targets) -> bool:
    """Find an augmenting path only when greedy assignment reaches a dead end."""
    queue = deque([target_index])
    seen_targets = {target_index}
    parents = np.full(len(cell_targets), -1, dtype=np.int64)
    while queue:
        current = queue.popleft()
        candidates = np.flatnonzero(possible[current])
        candidates = candidates[np.argsort(cost[current, candidates], kind="stable")]
        for cell in candidates:
            if parents[cell] != -1:
                continue
            parents[cell] = current
            owner = cell_targets[cell]
            if owner == -1:
                # Walk backwards and move every target to the next cell in the path.
                while cell != -1:
                    target = parents[cell]
                    previous_cell = target_cells[target]
                    target_cells[target] = cell
                    cell_targets[cell] = target
                    cell = previous_cell
                return True
            if owner not in seen_targets:
                seen_targets.add(owner)
                queue.append(owner)
    return False


def assign_targets(anchors, cell_sizes, boxes, valid):
    """Assign a unique nearby cell to each GT, constrained targets first."""
    if not len(boxes):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    possible, cost, order = assignment_candidates(anchors, cell_sizes, boxes, valid)
    cell_targets = np.full(len(anchors), -1, dtype=np.int64)
    target_cells = np.full(len(boxes), -1, dtype=np.int64)
    for target_index in order:
        candidates = np.flatnonzero(possible[target_index] & (cell_targets == -1))
        if not len(candidates):
            if reassign_target(target_index, possible, cost, target_cells, cell_targets):
                continue
            raise ValueError(
                "Detection grid cannot represent all boxes with unique nearby cells. "
                "Reduce backbone stride or use an FPN with a finer feature map."
            )
        cell_index = candidates[np.argmin(cost[target_index, candidates])]
        cell_targets[cell_index] = target_index
        target_cells[target_index] = cell_index
    return target_cells[order], order


def objectness_loss(logits, targets, kind="bce", alpha=0.25, gamma=2.0):
    """Elementwise BCE or RetinaNet sigmoid focal loss on logits."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if kind not in OBJECTNESS_LOSSES:
        raise ValueError(f"Unknown objectness loss: {kind}")
    if kind == "bce":
        return loss
    # Binary targets: log(1 - p_t) = -softplus(signed_logit).
    # The log form avoids 0**gamma derivatives for saturated logits and gamma < 1.
    signed_logits = logits * (2 * targets - 1)
    loss = loss * torch.exp(-gamma * F.softplus(signed_logits))
    if alpha != -1:
        class_weight = alpha * targets + (1 - alpha) * (1 - targets)
        loss = loss * class_weight
    return loss


def prepare_batch_targets(output, targets, image_width) -> dict:
    """Prepare matching, masks and counts on CPU before batched GPU reductions."""
    anchors = output["anchors"].detach().cpu().numpy()
    cells = output["cell_sizes"].detach().cpu().numpy()
    batch_size, cell_count = output["objectness"].shape
    positive = np.zeros((batch_size, cell_count), dtype=bool)
    negative = np.zeros_like(positive)
    matched_cells, matched_boxes = [], []

    for index, target in enumerate(targets):
        time_ratio = target["size"][1] / image_width
        valid = anchors[:, 0] < time_ratio
        boxes = target["boxes"].cpu().numpy().copy()
        boxes[:, [0, 2]] *= time_ratio
        cell_indices, target_indices = assign_targets(anchors, cells, boxes, valid)
        positive[index, cell_indices] = True
        negative[index] = valid & ~positive[index]
        matched_cells.append(cell_indices)
        matched_boxes.append(boxes[target_indices])

    positive_counts = positive.sum(axis=1)
    negative_counts = negative.sum(axis=1)
    max_matches = max(int(positive_counts.max()), 1)
    padded_cells = np.zeros((batch_size, max_matches), dtype=np.int64)
    truth_dtype = np.result_type(*matched_boxes)
    padded_boxes = np.zeros((batch_size, max_matches, 4), dtype=truth_dtype)
    matched = np.zeros((batch_size, max_matches), dtype=bool)
    for index, (indices, boxes) in enumerate(zip(matched_cells, matched_boxes)):
        count = len(indices)
        padded_cells[index, :count] = indices
        padded_boxes[index, :count] = boxes
        matched[index, :count] = True

    scores = output["objectness"]
    options = {"device": scores.device, "dtype": scores.dtype}
    return {
        "positive": torch.as_tensor(positive, **options),
        "negative": torch.as_tensor(negative, **options),
        "positive_count": torch.as_tensor(np.maximum(positive_counts, 1), **options),
        "negative_count": torch.as_tensor(np.maximum(negative_counts, 1), **options),
        "cells": torch.as_tensor(padded_cells, device=scores.device),
        "boxes": torch.as_tensor(padded_boxes, device=scores.device),
        "matched": torch.as_tensor(matched, device=scores.device),
    }


class DetectionLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        options = with_defaults("loss", config)
        self.kind = options["bbox"]
        self.objectness_kind = options["objectness"]
        self.focal_alpha = float(options["focal_alpha"])
        self.focal_gamma = float(options["focal_gamma"])
        self.box_weight = float(options["bbox_weight"])
        self.object_weight = float(options["objectness_weight"])
        self.negative_weight = float(options["negative_weight"])
        self.hard_negative_ratio = options["hard_negative_ratio"]

    def mean_negative_loss(self, objectness_losses: torch.Tensor, assigned: dict) -> torch.Tensor:
        """Mean negative objectness loss per sample, optionally focusing on likely false positives."""
        if self.hard_negative_ratio == 0:
            return (objectness_losses * assigned["negative"]).sum(1) / assigned["negative_count"]

        negative = assigned["negative"].bool()
        negative_count = negative.sum(1)
        positive_count = assigned["positive_count"].to(torch.long)
        # Any ratio above the grid size already selects every available negative.
        ratio = min(self.hard_negative_ratio, objectness_losses.shape[1])
        budget = torch.minimum(negative_count, ratio * positive_count)

        # Positives and padding must rank below every actual negative, even when
        # easy negative BCE is zero. No per-sample device synchronization is needed.
        ranked_losses = objectness_losses.masked_fill(~negative, -torch.inf).sort(
            dim=1, descending=True
        ).values
        ranks = torch.arange(objectness_losses.shape[1], device=objectness_losses.device).unsqueeze(0)
        selected = ranks < budget.unsqueeze(1)
        selected_losses = torch.where(selected, ranked_losses, 0)
        return selected_losses.sum(1) / budget.clamp_min(1)

    def forward(self, output, targets, image_shape) -> dict:
        assigned = prepare_batch_targets(output, targets, image_shape[1])
        objectness_losses = objectness_loss(
            output["objectness"], assigned["positive"], self.objectness_kind,
            self.focal_alpha, self.focal_gamma,
        )
        positive_loss = (objectness_losses * assigned["positive"]).sum(1) / assigned["positive_count"]
        negative_loss = self.mean_negative_loss(objectness_losses, assigned)
        objectness = positive_loss + self.negative_weight * negative_loss

        indices = assigned["cells"].unsqueeze(-1).expand(-1, -1, 4)
        matched_boxes = output["boxes"].gather(1, indices)
        # Padding must not evaluate arbitrary unmatched boxes before the reduction.
        matched_boxes = torch.where(assigned["matched"].unsqueeze(-1), matched_boxes, 0)
        box_losses = bbox_loss(
            matched_boxes.reshape(-1, 4), assigned["boxes"].reshape(-1, 4), self.kind
        ).reshape_as(assigned["matched"])
        regression = (box_losses * assigned["matched"]).sum(1) / assigned["positive_count"]
        total = self.box_weight * regression + self.object_weight * objectness
        return {
            "total": total.mean(),
            "bbox": regression.mean(),
            "objectness": objectness.mean(),
        }
