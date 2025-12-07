"""
metrics.py

Evaluation metrics for TF-CLIMB experiments.

Provided functions:
    - compute_overall_macro: overall accuracy + macro (per-class) accuracy
    - compute_per_class_accuracy: per-class accuracy array
    - split_head_tail: split classes into head / tail based on counts
    - compute_head_tail_macro: macro accuracy over head and tail class groups
"""

from typing import Tuple

import numpy as np
import torch


def compute_per_class_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """
    Compute per-class accuracy.

    Args:
        pred:    [N] tensor of predicted class indices
        labels:  [N] tensor of ground-truth class indices
        num_classes: total number of classes C

    Returns:
        per_class_acc: numpy array [C], with NaN for classes not present in labels
    """
    pred_np = pred.cpu().numpy()
    labels_np = labels.cpu().numpy()

    per_class_acc = np.full(shape=(num_classes,), fill_value=np.nan, dtype=np.float32)

    for c in range(num_classes):
        mask = (labels_np == c)
        num = mask.sum()
        if num == 0:
            continue
        correct = (pred_np[mask] == labels_np[mask]).sum()
        per_class_acc[c] = correct / float(num)

    return per_class_acc


def compute_overall_macro(
    pred: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float]:
    """
    Compute overall accuracy and macro (per-class) accuracy.

    Args:
        pred:    [N] tensor of predicted class indices
        labels:  [N] tensor of ground-truth class indices
        num_classes: total number of classes C

    Returns:
        overall: overall top-1 accuracy (float)
        macro:   macro-averaged accuracy over classes (float)
    """
    pred_np = pred.cpu().numpy()
    labels_np = labels.cpu().numpy()

    overall = float((pred_np == labels_np).mean())

    per_class_acc = compute_per_class_accuracy(pred, labels, num_classes)
    # ignore NaN (classes not present in labels)
    valid = ~np.isnan(per_class_acc)
    if valid.sum() > 0:
        macro = float(per_class_acc[valid].mean())
    else:
        macro = float("nan")

    return overall, macro


def split_head_tail(
    counts: torch.Tensor,
    head_ratio: float = 1.0 / 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split classes into head and tail groups based on support counts.

    Args:
        counts: [C] tensor of support counts per class
        head_ratio: fraction of classes to treat as "head" (default: top 1/3)

    Returns:
        head_idx: LongTensor of class indices in the head group
        tail_idx: LongTensor of class indices in the tail group
    """
    if not (0.0 < head_ratio < 1.0):
        raise ValueError(f"head_ratio must be in (0, 1), got {head_ratio}")

    num_classes = counts.shape[0]
    sorted_idx = torch.argsort(counts, descending=True)
    num_head = max(1, int(num_classes * head_ratio))
    num_tail = num_head  # symmetric: same number of tail classes

    head_idx = sorted_idx[:num_head]
    tail_idx = sorted_idx[-num_tail:]

    return head_idx, tail_idx


def compute_group_macro_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    group_idx: torch.Tensor,
) -> float:
    """
    Compute macro accuracy restricted to a subset of classes.

    Args:
        pred:      [N] predictions
        labels:    [N] ground-truth labels
        group_idx: [G] tensor of class indices defining the group (e.g., head or tail)

    Returns:
        macro_acc: macro-averaged accuracy over classes in group (float)
    """
    pred_np = pred.cpu().numpy()
    labels_np = labels.cpu().numpy()
    group = set(int(c) for c in group_idx.cpu().numpy())

    per_class_acc = []

    for c in group:
        mask = (labels_np == c)
        num = mask.sum()
        if num == 0:
            continue
        correct = (pred_np[mask] == labels_np[mask]).sum()
        per_class_acc.append(correct / float(num))

    if len(per_class_acc) == 0:
        return float("nan")

    return float(np.mean(per_class_acc))


def compute_head_tail_macro(
    pred: torch.Tensor,
    labels: torch.Tensor,
    counts: torch.Tensor,
    head_ratio: float = 1.0 / 3.0,
) -> Tuple[float, float]:
    """
    Compute macro accuracy separately for head and tail classes.

    Args:
        pred:       [N] predictions
        labels:     [N] ground-truth labels
        counts:     [C] support counts per class
        head_ratio: fraction of classes to treat as "head"

    Returns:
        head_macro: macro accuracy over head classes
        tail_macro: macro accuracy over tail classes
    """
    head_idx, tail_idx = split_head_tail(counts, head_ratio=head_ratio)
    head_macro = compute_group_macro_accuracy(pred, labels, head_idx)
    tail_macro = compute_group_macro_accuracy(pred, labels, tail_idx)
    return head_macro, tail_macro
