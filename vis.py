"""
visualize_results.py

Read saved experiment results (CSV files) and visualize meaningful metrics,
such as macro accuracy and head/tail macro differences.

Expected input CSVs (from run_experiments.py):

    results/{dataset}_results.csv

Columns:
    dataset, K, rho, seed, method,
    overall_acc, macro_acc, head_macro_acc, tail_macro_acc,
    min_count, max_count

This script will:
    1) Load a CSV for a given dataset
    2) For a given dataset and K, aggregate over seeds (mean/std)
    3) Produce several plots:
        - macro vs. rho (for each method)
        - head vs. tail macro vs. rho
        - tail - head macro gap vs. rho

Usage example:

    python visualize_results.py --dataset cifar100 --K 4

Plots will be saved under: results/figs/{dataset}/
"""

import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATASETS, RESULT_ROOT


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_results_for_dataset(dataset: str) -> pd.DataFrame:
    """Load the CSV results for a single dataset."""
    path = os.path.join(RESULT_ROOT, f"{dataset}_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found for dataset={dataset}: {path}")
    df = pd.read_csv(path)
    print(f"[visualize_results] Loaded results from: {path} (rows={len(df)})")
    return df


def aggregate_results_for_K(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """
    Aggregate results for a given shot K across seeds.

    Returns a DataFrame grouped by (method, rho) with mean/std of metrics.
    """
    df_sub = df[df["K"] == K].copy()
    if df_sub.empty:
        print(f"[visualize_results] No rows for K={K}")
        return pd.DataFrame()

    grouped = df_sub.groupby(["method", "rho"])

    agg = grouped.agg(
        overall_mean=("overall_acc", "mean"),
        overall_std=("overall_acc", "std"),
        macro_mean=("macro_acc", "mean"),
        macro_std=("macro_acc", "std"),
        head_macro_mean=("head_macro_acc", "mean"),
        head_macro_std=("head_macro_acc", "std"),
        tail_macro_mean=("tail_macro_acc", "mean"),
        tail_macro_std=("tail_macro_acc", "std"),
    ).reset_index()

    return agg


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_macro_vs_rho(
    agg: pd.DataFrame,
    dataset: str,
    K: int,
    out_dir: str,
):
    """
    Plot macro accuracy vs. rho for each method.

    Saves a PNG to out_dir.
    """
    if agg.empty:
        print("[visualize_results] No aggregated data for macro vs rho.")
        return

    method_order = ["zero_shot", "prototypes", "fused_no_adj", "tfclimb"]
    method_display = {
        "zero_shot": "Zero-shot CLIP",
        "prototypes": "Prototypes",
        "fused_no_adj": "Fused (no adj.)",
        "tfclimb": "TF-CLIMB (ours)",
    }

    rho_values = sorted(agg["rho"].unique())

    plt.figure(figsize=(6, 4))

    for method in method_order:
        df_m = agg[agg["method"] == method]
        if df_m.empty:
            continue
        xs, ys = [], []
        for rho in rho_values:
            df_cell = df_m[df_m["rho"] == rho]
            if df_cell.empty:
                continue
            xs.append(rho)
            ys.append(df_cell["macro_mean"].iloc[0] * 100.0)  # convert to %
        if xs:
            plt.plot(xs, ys, marker="o", label=method_display.get(method, method))

    plt.xlabel(r"$\rho$ (imbalance ratio)")
    plt.ylabel("Macro accuracy (%)")
    # 标题精简；数据集和 K 也可以只在 caption 中说明
    plt.title(f"{dataset}, {K}-shot", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 图例放在图内右下角，去掉边框，减少留白
    plt.legend(loc="lower right", frameon=False)

    fname = os.path.join(out_dir, f"{dataset}_K{K}_macro_vs_rho.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[visualize_results] Saved plot: {fname}")


def plot_head_tail_vs_rho(
    agg: pd.DataFrame,
    dataset: str,
    K: int,
    out_dir: str,
):
    """
    Plot head and tail macro accuracy vs. rho for each method
    in two subplots, each with its own legend inside the axes.
    """
    if agg.empty:
        print("[visualize_results] No aggregated data for head/tail vs rho.")
        return

    method_order = ["zero_shot", "prototypes", "fused_no_adj", "tfclimb"]
    method_display = {
        "zero_shot": "Zero-shot CLIP",
        "prototypes": "Prototypes",
        "fused_no_adj": "Fused (no adj.)",
        "tfclimb": "TF-CLIMB (ours)",
    }

    rho_values = sorted(agg["rho"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax_head, ax_tail = axes

    for method in method_order:
        df_m = agg[agg["method"] == method]
        if df_m.empty:
            continue
        xs, ys_head, ys_tail = [], [], []
        for rho in rho_values:
            df_cell = df_m[df_m["rho"] == rho]
            if df_cell.empty:
                continue
            xs.append(rho)
            ys_head.append(df_cell["head_macro_mean"].iloc[0] * 100.0)
            ys_tail.append(df_cell["tail_macro_mean"].iloc[0] * 100.0)

        if xs:
            label = method_display.get(method, method)
            ax_head.plot(xs, ys_head, marker="o", label=label)
            ax_tail.plot(xs, ys_tail, marker="o", label=label)

    ax_head.set_title("Head macro (%)", fontsize=11)
    ax_tail.set_title("Tail macro (%)", fontsize=11)

    for ax in axes:
        ax.set_xlabel(r"$\rho$")
        ax.grid(True, linestyle="--", alpha=0.5)
    ax_head.set_ylabel("Accuracy (%)")

    # 每个子图各自 legend；如果觉得信息太多，可以只保留右图 legend
    ax_head.legend(loc="lower right", frameon=False, fontsize=8)
    ax_tail.legend(loc="lower right", frameon=False, fontsize=8)

    # 不使用 suptitle，避免额外顶部留白
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{dataset}_K{K}_head_tail_vs_rho.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[visualize_results] Saved plot: {fname}")


def plot_tail_head_gap_vs_rho(
    agg: pd.DataFrame,
    dataset: str,
    K: int,
    out_dir: str,
):
    """
    Plot (tail_macro - head_macro) vs. rho for each method.

    Positive values mean tail classes perform better than head classes,
    negative values mean tail classes are worse.
    """
    if agg.empty:
        print("[visualize_results] No aggregated data for tail-head gap vs rho.")
        return

    method_order = ["zero_shot", "prototypes", "fused_no_adj", "tfclimb"]
    method_display = {
        "zero_shot": "Zero-shot CLIP",
        "prototypes": "Prototypes",
        "fused_no_adj": "Fused (no adj.)",
        "tfclimb": "TF-CLIMB (ours)",
    }

    rho_values = sorted(agg["rho"].unique())

    plt.figure(figsize=(6, 4))

    for method in method_order:
        df_m = agg[agg["method"] == method]
        if df_m.empty:
            continue
        xs, ys_gap = [], []
        for rho in rho_values:
            df_cell = df_m[df_m["rho"] == rho]
            if df_cell.empty:
                continue
            head = df_cell["head_macro_mean"].iloc[0]
            tail = df_cell["tail_macro_mean"].iloc[0]
            xs.append(rho)
            ys_gap.append((tail - head) * 100.0)  # percentage points
        if xs:
            plt.plot(xs, ys_gap, marker="o", label=method_display.get(method, method))

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(r"$\rho$ (imbalance ratio)")
    plt.ylabel("Tail - Head macro (pp)")
    plt.title(f"{dataset}, {K}-shot", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend(loc="lower right", frameon=False)

    fname = os.path.join(out_dir, f"{dataset}_K{K}_tail_head_gap_vs_rho.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[visualize_results] Saved plot: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize TF-CLIMB results: macro and head/tail metrics vs. imbalance ratio."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Dataset name, one of {DATASETS}",
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Shot number K to visualize (e.g., 1, 2, 4, 8).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    K = args.K

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset='{dataset}', expected one of {DATASETS}")

    df = load_results_for_dataset(dataset)
    agg = aggregate_results_for_K(df, K)
    if agg.empty:
        print(f"[visualize_results] No data for dataset={dataset}, K={K}. Nothing to plot.")
        return

    # output directory for plots
    out_dir = os.path.join(RESULT_ROOT, "figs", dataset)
    _ensure_dir(out_dir)

    # Plot 1: macro vs rho
    plot_macro_vs_rho(agg, dataset, K, out_dir)

    # Plot 2: head / tail macro vs rho
    plot_head_tail_vs_rho(agg, dataset, K, out_dir)

    # Plot 3: tail - head gap vs rho
    plot_tail_head_gap_vs_rho(agg, dataset, K, out_dir)


if __name__ == "__main__":
    main()
