"""
run_experiments.py

Main script to run TF-CLIMB experiments on multiple datasets, shot budgets,
and imbalance ratios. Results are saved as CSV files in RESULT_ROOT.
"""

import os
from typing import Dict, Any, List

import numpy as np
import torch
import pandas as pd

from config import (
    DATASETS,
    SHOT_LIST,
    IMBALANCE_RATIOS,
    SEEDS,
    RESULT_ROOT,
    TFCLIMB_CFG,
    SPLIT_ROOT,
    USE_PREGENERATED_SPLITS
)
from data_utils import get_or_compute_features
from sampling import build_class_index, sample_counts, sample_support_indices
from tfclimb import (
    build_text_features,
    build_tfclimb_stats,
    predict_zero_shot,
    predict_prototypes_only,
    predict_fused,
    predict_tfclimb,
)
from metrics import (
    compute_overall_macro,
    compute_head_tail_macro,
)

def compute_imbalance_ratio_from_counts(counts: torch.Tensor) -> float:
    """
    Compute empirical imbalance ratio rho = n_max / n_min from support counts.
    Only considers classes with count > 0.
    """
    nonzero = counts[counts > 0]
    if nonzero.numel() == 0:
        return 1.0
    n_max = nonzero.max().item()
    n_min = nonzero.min().item()
    return float(n_max / max(n_min, 1))

def run_for_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Run experiments on a single dataset over all K (shots), rho (imbalance ratios),
    and seeds. Returns a pandas DataFrame with all results.
    """
    print(f"\n================ Dataset: {dataset_name} ================\n")

    # 1) Load or compute CLIP features
    feat_dict = get_or_compute_features(dataset_name)
    train_features: torch.Tensor = feat_dict["train_features"]  # [N_train, d]
    train_labels: torch.Tensor = feat_dict["train_labels"]      # [N_train]
    test_features: torch.Tensor = feat_dict["test_features"]    # [N_test, d]
    test_labels: torch.Tensor = feat_dict["test_labels"]        # [N_test]
    class_names: List[str] = feat_dict["class_names"]

    num_classes = len(class_names)
    print(f"[run_experiments] num_train={len(train_labels)}, num_test={len(test_labels)}, C={num_classes}")

    # 2) Precompute text features (prompts -> CLIP embeddings) and logit_scale
    text_features, logit_scale = build_text_features(class_names)  # [C, d] + scalar
    print(f"[run_experiments] CLIP logit_scale (tau_text) = {logit_scale:.4f}")

    # 3) Build class index on training data
    class2idx = build_class_index(train_labels)

    # 4) Storage for all results rows
    rows: List[Dict[str, Any]] = []

    for K in SHOT_LIST:
        for rho in IMBALANCE_RATIOS:
            print(f"\n[run_experiments] Dataset={dataset_name}, K={K}, rho={rho}")

            for seed in SEEDS:
                print(f"  - seed={seed}")
                rng = np.random.default_rng(seed)

                # 4.1) Get per-class counts and support indices:
                #      either from pre-generated splits, or sample on the fly.
                split_path = os.path.join(
                    SPLIT_ROOT, dataset_name, f"geom_K{K}_rho{rho}_seed{seed}.pt"
                )

                if USE_PREGENERATED_SPLITS and os.path.exists(split_path):
                    split_obj = torch.load(split_path)
                    n_c = split_obj["n_c"]              # numpy array [C]
                    support_idx = split_obj["support_idx"]  # list[int]
                    print(f"    Using pre-generated split: {split_path}")
                else:
                    # fallback to on-the-fly sampling (old behavior)
                    n_c = sample_counts(num_classes, K, rho, rng)  # numpy array [C]
                    support_idx = sample_support_indices(class2idx, n_c, seed)
                    print("    Using on-the-fly sampling for support set.")

                # 4.2) Build TF-CLIMB statistics (prototypes, priors)
                prototypes, pi, counts = build_tfclimb_stats(
                    train_features=train_features,
                    train_labels=train_labels,
                    support_indices=support_idx,
                    num_classes=num_classes,
                )
                rho_emp = compute_imbalance_ratio_from_counts(counts)
                base_lam = TFCLIMB_CFG.lam
                if rho_emp <= 2.0:
                    lambda_eff = 0.0
                elif rho_emp >= 10.0:
                    lambda_eff = base_lam
                else:
                    # 线性插值: ρ_emp 从 2 -> 10 时，lambda 从 0 -> base_lam
                    t = (rho_emp - 2.0) / (10.0 - 2.0)
                    lambda_eff = t * base_lam

                print(f"    rho_emp={rho_emp:.2f}, lambda_eff={lambda_eff:.3f}")
                # 4.3) Run different methods

                # zero-shot CLIP (with official logit_scale)
                zs_pred = predict_zero_shot(
                    test_features=test_features,
                    text_features=text_features,
                    tau_text=logit_scale,
                )

                # prototypes only
                proto_pred = predict_prototypes_only(
                    test_features=test_features,
                    prototypes=prototypes,
                    tau_img=TFCLIMB_CFG.tau_img,
                )

                # fused (no adjustment)
                fused_pred = predict_fused(
                    test_features=test_features,
                    text_features=text_features,
                    prototypes=prototypes,
                    tau_text=logit_scale,
                    tau_img=TFCLIMB_CFG.tau_img,
                    alpha=TFCLIMB_CFG.alpha,
                )

                # TF-CLIMB (ours)
                tfclimb_pred = predict_tfclimb(
                    test_features=test_features,
                    text_features=text_features,
                    prototypes=prototypes,
                    pi=pi,
                    tau_text=logit_scale,
                    tau_img=TFCLIMB_CFG.tau_img,
                    alpha=TFCLIMB_CFG.alpha,
                    lam=TFCLIMB_CFG.lam,
                    eps=TFCLIMB_CFG.eps,
                )

                # 4.4) Compute metrics for each method
                methods = {
                    "zero_shot": zs_pred,
                    "prototypes": proto_pred,
                    "fused_no_adj": fused_pred,
                    "tfclimb": tfclimb_pred,
                }

                for method_name, pred in methods.items():
                    overall, macro = compute_overall_macro(
                        pred=pred,
                        labels=test_labels,
                        num_classes=num_classes,
                    )
                    head_macro, tail_macro = compute_head_tail_macro(
                        pred=pred,
                        labels=test_labels,
                        counts=counts,
                        head_ratio=1.0 / 3.0,
                    )

                    row = {
                        "dataset": dataset_name,
                        "K": K,
                        "rho": rho,
                        "seed": seed,
                        "method": method_name,
                        "overall_acc": overall,
                        "macro_acc": macro,
                        "head_macro_acc": head_macro,
                        "tail_macro_acc": tail_macro,
                        "min_count": int(counts.min().item()),
                        "max_count": int(counts.max().item()),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    for dataset_name in DATASETS:
        df = run_for_dataset(dataset_name)

        out_path = os.path.join(RESULT_ROOT, f"{dataset_name}_results.csv")
        df.to_csv(out_path, index=False)
        print(f"[run_experiments] Saved results for '{dataset_name}' to: {out_path}")


if __name__ == "__main__":
    main()
