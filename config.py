"""
config.py

Global configuration and default hyperparameters for TF-CLIMB experiments.
You can modify paths, datasets, and hyperparameters here without touching
the rest of the code.
"""

from dataclasses import dataclass
from typing import List


# ------------------------------
# Basic paths
# ------------------------------

# Root directory for raw datasets (torchvision will download here)
DATA_ROOT: str = "data"

# Directory to cache extracted CLIP features (so we don't recompute every time)
FEATURE_ROOT: str = "features"

# Directory to store experiment logs / CSV result files
RESULT_ROOT: str = "results"

# Root directory for pre-generated few-shot / imbalance splits
SPLIT_ROOT: str = "splits"

# Whether run_experiments.py should use pre-generated splits
# If False, it will randomly sample support sets on the fly (old behavior).
USE_PREGENERATED_SPLITS: bool = True

# ------------------------------
# CLIP model config
# ------------------------------

# Which CLIP model to use (must be supported by openai/CLIP)
CLIP_MODEL_NAME: str = "ViT-B/16"


# ------------------------------
# Experiment settings
# ------------------------------

# Datasets to run; supported names are "cifar100", "eurosat", "pets"
DATASETS: List[str] = ["cifar100", "eurosat", "pets"]

# Average shots per class
SHOT_LIST: List[int] = [1, 2, 4, 8]

# Imbalance ratios rho = n_max / n_min
IMBALANCE_RATIOS: List[int] = [1, 5, 10]

# Random seeds for sampling support sets
SEEDS: List[int] = [0, 1, 2]

# Batch size when extracting CLIP features
FEATURE_BATCH_SIZE: int = 64

# Number of workers for DataLoader
NUM_WORKERS: int = 4


# ------------------------------
# TF-CLIMB hyperparameters
# ------------------------------

@dataclass
class TFClimbConfig:
    """
    Hyperparameters for the TF-CLIMB method.
    These are intentionally simple and shared across datasets.
    """
    # Temperature for prototype (image) logits
    tau_img: float = 10.0

    # Temperature for text (zero-shot) logits
    # In practice, CLIP has an internal logit scale; we treat this as 1.0
    tau_text: float = 1.0

    # Fusion weight between text and prototype logits:
    # fused = (1 - alpha) * text + alpha * proto
    alpha: float = 0.5

    # Logit adjustment strength lambda
    lam: float = 0.5

    # Small constant to avoid log(0) in class-frequency computation
    eps: float = 1e-6
    
    prior_beta: float = 0.7


# Default configuration instance used by run_experiments.py
TFCLIMB_CFG = TFClimbConfig()


# ------------------------------
# Other experiment options
# ------------------------------

# Whether to recompute CLIP features even if cached files exist
FORCE_RECOMPUTE_FEATURES: bool = False

# Device selection hint; actual code will also check torch.cuda availability
# Allowed values: "auto", "cpu", "cuda"
DEVICE_PREF: str = "auto"
