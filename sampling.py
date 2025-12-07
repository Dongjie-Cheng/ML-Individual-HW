"""
sampling.py

Few-shot and class-imbalance sampling utilities for TF-CLIMB experiments.

Core functions:
    - build_class_index: map class -> list of indices
    - sample_counts: sample per-class support counts with given average K and ratio rho
    - sample_support_indices: sample support indices from each class
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


def build_class_index(labels: torch.Tensor) -> Dict[int, List[int]]:
    """
    Build an index mapping each class to the list of dataset indices.

    Args:
        labels: [N] tensor of integer labels

    Returns:
        class2idx: dict mapping class_id -> list of indices
    """
    class2idx: Dict[int, List[int]] = {}
    labels_np = labels.cpu().numpy()
    for idx, y in enumerate(labels_np):
        y = int(y)
        if y not in class2idx:
            class2idx[y] = []
        class2idx[y].append(idx)
    return class2idx


def sample_counts(num_classes: int, K: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample per-class support counts {n_c} with given average shots K and imbalance ratio rho.

    We use a simple geometric progression approach:
        - if rho == 1, all classes have exactly K shots;
        - otherwise, we create a smooth geometric sequence between 1 and rho,
          normalize it to have mean K, and round to integers (at least 1).

    Args:
        num_classes: number of classes C
        K: target average shots per class
        rho: imbalance ratio = n_max / n_min (approximate)
        rng: numpy random generator for reproducibility

    Returns:
        n_c: numpy array of shape [C], integer counts per class
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}.")

    if rho <= 1.0:
        # Balanced case: all classes get exactly K shots
        return np.full(shape=(num_classes,), fill_value=K, dtype=int)

    # Create a geometric progression between [1, rho]
    # Example: for C=5, rho=10, we might get [1.0, ~1.78, ~3.16, ~5.62, 10.0]
    geom = np.geomspace(1.0, rho, num_classes).astype(float)

    # Normalize to have mean ~K
    geom = geom / geom.mean() * K

    # Add small random noise to avoid too many identical counts
    noise = rng.uniform(low=0.0, high=0.3, size=num_classes)
    geom = geom * (1.0 + noise)

    # Round to integers and clamp to at least 1
    n_c = np.round(geom).astype(int)
    n_c[n_c < 1] = 1

    # Occasionally adjust to keep average close to K (not strictly necessary)
    # We do a simple rescaling if the deviation is large.
    current_mean = n_c.mean()
    if abs(current_mean - K) > 0.5:
        scale = K / current_mean
        n_c = np.round(n_c * scale).astype(int)
        n_c[n_c < 1] = 1

    return n_c


def sample_support_indices(
    class2idx: Dict[int, List[int]],
    n_c: np.ndarray,
    seed: int,
) -> List[int]:
    """
    Sample support indices from each class according to n_c.

    Args:
        class2idx: mapping from class_id -> list of available indices in the training set
        n_c: numpy array [C] with counts per class (C = len(class2idx))
        seed: random seed for reproducibility

    Returns:
        support_indices: sorted list of selected indices (global indices in the dataset)
    """
    rng = np.random.default_rng(seed)
    support_indices: List[int] = []

    # We assume classes are labeled from 0..C-1, consistent with PyTorch datasets
    # If your dataset uses different labels, you may need to map accordingly.
    class_ids = sorted(class2idx.keys())
    if len(class_ids) != len(n_c):
        raise ValueError(
            f"Mismatch: len(class2idx)={len(class_ids)} but len(n_c)={len(n_c)}."
        )

    for idx_c, c in enumerate(class_ids):
        available = class2idx[c]
        k = int(n_c[idx_c])
        if k > len(available):
            # If requested shots exceed available samples, just take all
            k = len(available)

        chosen = rng.choice(available, size=k, replace=False)
        support_indices.extend(int(i) for i in chosen)

    support_indices = sorted(support_indices)
    return support_indices


def counts_to_tensor(n_c: np.ndarray) -> torch.Tensor:
    """
    Convenience function: convert numpy array of counts to torch.LongTensor.

    Args:
        n_c: numpy array [C] of counts

    Returns:
        counts: LongTensor [C]
    """
    return torch.from_numpy(n_c.astype(int))
