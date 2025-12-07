"""
make_imbalanced_supports.py

Script to pre-generate imbalanced few-shot support splits for multiple datasets,
shot budgets, imbalance ratios, and seeds.

Each split is saved as a .pt file under:

    splits/{dataset}/geom_K{K}_rho{rho}_seed{seed}.pt

with contents:
    {
        "dataset": dataset_name,
        "K": K,
        "rho": rho,
        "seed": seed,
        "n_c": numpy array [C] of per-class counts,
        "support_idx": list of selected indices in the training set,
    }
"""

import os
from typing import List, Dict, Any

import numpy as np
import torch

from config import (
    DATASETS,
    SHOT_LIST,
    IMBALANCE_RATIOS,
    SEEDS,
    SPLIT_ROOT,
)
from data_utils import get_or_compute_features
from sampling import build_class_index, sample_counts, sample_support_indices


def make_splits_for_dataset(dataset_name: str):
    print(f"\n================ Building splits for dataset: {dataset_name} ================\n")

    # 我们只需要 train_labels 来做采样；features 不用管
    feat_dict = get_or_compute_features(dataset_name)
    train_labels: torch.Tensor = feat_dict["train_labels"]
    class_names: List[str] = feat_dict["class_names"]
    num_classes = len(class_names)

    class2idx = build_class_index(train_labels)

    out_dir = os.path.join(SPLIT_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for K in SHOT_LIST:
        for rho in IMBALANCE_RATIOS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)

                # 1) 采样各类支持样本数 n_c
                n_c = sample_counts(num_classes, K, rho, rng)  # numpy array [C]

                # 2) 根据 n_c 采样 support indices
                support_idx = sample_support_indices(class2idx, n_c, seed)  # list[int]

                split_name = f"geom_K{K}_rho{rho}_seed{seed}.pt"
                out_path = os.path.join(out_dir, split_name)

                save_obj: Dict[str, Any] = {
                    "dataset": dataset_name,
                    "K": K,
                    "rho": rho,
                    "seed": seed,
                    "n_c": n_c,  # numpy array
                    "support_idx": support_idx,
                }

                torch.save(save_obj, out_path)
                print(f"[make_imbalanced_supports] Saved split: {out_path}")


def main():
    os.makedirs(SPLIT_ROOT, exist_ok=True)

    for dataset_name in DATASETS:
        make_splits_for_dataset(dataset_name)


if __name__ == "__main__":
    main()
