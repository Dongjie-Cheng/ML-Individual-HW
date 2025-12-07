"""
data_utils.py

Dataset loading and CLIP feature extraction utilities for TF-CLIMB experiments.
"""

import os
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from urllib.error import URLError
import ssl
import clip

from config import (
    DATA_ROOT,
    FEATURE_ROOT,
    FEATURE_BATCH_SIZE,
    NUM_WORKERS,
    CLIP_MODEL_NAME,
    FORCE_RECOMPUTE_FEATURES,
    DEVICE_PREF,
)


# ---------------------------------------------------------------------------
# Device / CLIP helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Select device based on config and availability."""
    if DEVICE_PREF == "cpu":
        return torch.device("cpu")
    if DEVICE_PREF == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE_PREF == "auto"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_model():
    """
    Load CLIP model and its preprocess function.

    Returns:
        model: CLIP model (on correct device, in eval mode)
        preprocess: torchvision-like transform for PIL images
        device: torch.device
    """
    device = get_device()
    print(f"[data_utils] Loading CLIP model '{CLIP_MODEL_NAME}' on device: {device}")
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()
    return model, preprocess, device


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_cifar100(preprocess) -> Tuple[torch.utils.data.Dataset,
                                       torch.utils.data.Dataset,
                                       List[str]]:
    """Load CIFAR-100 train/test with CLIP preprocess."""
    train_set = datasets.CIFAR100(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=preprocess,
    )
    test_set = datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=preprocess,
    )
    class_names = list(train_set.classes)  # list of 100 class names
    return train_set, test_set, class_names


def load_eurosat(preprocess) -> Tuple[torch.utils.data.Dataset,
                                      torch.utils.data.Dataset,
                                      List[str]]:
    """
    Load EuroSAT dataset with CLIP preprocess.

    If online download fails due to SSL/connection issues, this function
    will fall back to using a locally available copy (download=False).
    You must manually download and place EuroSAT under DATA_ROOT/EuroSAT.
    """
    try:
        from torchvision.datasets import EuroSAT
    except ImportError as e:
        raise ImportError(
            "torchvision.datasets.EuroSAT not available. "
            "Please upgrade torchvision or implement a custom EuroSAT loader."
        ) from e

    # Try online download first
    try:
        full_dataset = EuroSAT(
            root=DATA_ROOT,
            download=True,
            transform=preprocess,
        )
    except (URLError, ssl.SSLError) as e:
        print("[data_utils] Failed to download EuroSAT due to network/SSL issue:")
        print(f"  {repr(e)}")
        print("[data_utils] Falling back to local EuroSAT copy (download=False).")
        print("[data_utils] Please make sure EuroSAT is already downloaded under "
              f"'{os.path.join(DATA_ROOT, 'EuroSAT')}'.")
        # Try to load without downloading
        full_dataset = EuroSAT(
            root=DATA_ROOT,
            download=False,
            transform=preprocess,
        )

    # 80/20 random split with fixed seed for reproducibility
    num_total = len(full_dataset)
    num_train = int(0.8 * num_total)
    num_val = num_total - num_train
    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(
        full_dataset,
        lengths=[num_train, num_val],
        generator=generator,
    )

    # class names
    if hasattr(full_dataset, "classes"):
        class_names = list(full_dataset.classes)
    else:
        labels = [full_dataset[i][1] for i in range(len(full_dataset))]
        unique = sorted(set(int(y) for y in labels))
        class_names = [f"class_{c}" for c in unique]

    return train_set, test_set, class_names


def load_pets(preprocess) -> Tuple[torch.utils.data.Dataset,
                                   torch.utils.data.Dataset,
                                   List[str]]:
    """
    Load Oxford-IIIT Pets with CLIP preprocess.

    We use the 'trainval' split as training, 'test' as test.
    """
    from torchvision.datasets import OxfordIIITPet

    train_set = OxfordIIITPet(
        root=DATA_ROOT,
        split="trainval",
        download=True,
        transform=preprocess,
    )
    test_set = OxfordIIITPet(
        root=DATA_ROOT,
        split="test",
        download=True,
        transform=preprocess,
    )

    # Class names
    if hasattr(train_set, "classes"):
        class_names = list(train_set.classes)
    else:
        # Fallback: derive from internal mapping if needed
        # This is unlikely to be used on modern torchvision
        labels = [train_set[i][1] for i in range(len(train_set))]
        unique = sorted(set(int(y) for y in labels))
        class_names = [f"pet_class_{c}" for c in unique]

    return train_set, test_set, class_names


def load_dataset_by_name(
    dataset_name: str,
    preprocess,
) -> Tuple[torch.utils.data.Dataset,
           torch.utils.data.Dataset,
           List[str]]:
    """
    Unified entry point to load a dataset by name.

    Args:
        dataset_name: one of {"cifar100", "eurosat", "pets"}
        preprocess: CLIP preprocess transform

    Returns:
        train_set, test_set, class_names
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar100":
        return load_cifar100(preprocess)
    elif dataset_name == "eurosat":
        return load_eurosat(preprocess)
    elif dataset_name == "pets":
        return load_pets(preprocess)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_image_features(
    dataset: torch.utils.data.Dataset,
    model,
    device: torch.device,
    batch_size: int = FEATURE_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract CLIP image features and labels for an entire dataset.

    Args:
        dataset: torchvision-style dataset returning (image, label)
        model: CLIP model
        device: torch.device
        batch_size: DataLoader batch size
        num_workers: DataLoader num_workers

    Returns:
        feats: [N, d] tensor of normalized CLIP image features
        labels: [N] tensor of integer labels
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats.float() 
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


def _feature_cache_paths(dataset_name: str) -> Tuple[str, str]:
    """Return paths for cached train/test feature files."""
    os.makedirs(FEATURE_ROOT, exist_ok=True)
    train_path = os.path.join(FEATURE_ROOT, f"{dataset_name}_train.pt")
    test_path = os.path.join(FEATURE_ROOT, f"{dataset_name}_test.pt")
    return train_path, test_path


def get_or_compute_features(dataset_name: str) -> Dict[str, object]:
    """
    Load or compute CLIP features for a given dataset.

    Args:
        dataset_name: one of {"cifar100", "eurosat", "pets"}

    Returns:
        A dict with keys:
            - "train_features": [N_train, d] tensor
            - "train_labels":   [N_train] tensor
            - "test_features":  [N_test, d] tensor
            - "test_labels":    [N_test] tensor
            - "class_names":    list of str, length C
    """
    dataset_name = dataset_name.lower()
    train_path, test_path = _feature_cache_paths(dataset_name)

    # Try to load cached features
    if (not FORCE_RECOMPUTE_FEATURES
            and os.path.exists(train_path)
            and os.path.exists(test_path)):
        print(f"[data_utils] Loading cached features for '{dataset_name}'...")
        train_data = torch.load(train_path)
        test_data = torch.load(test_path)

        # train_data/test_data expected to be dicts with keys below
        result = {
            "train_features": train_data["features"],
            "train_labels": train_data["labels"],
            "test_features": test_data["features"],
            "test_labels": test_data["labels"],
            "class_names": train_data["class_names"],
        }
        return result

    # Otherwise, compute features from scratch
    print(f"[data_utils] Computing CLIP features for '{dataset_name}'...")
    model, preprocess, device = load_clip_model()
    train_set, test_set, class_names = load_dataset_by_name(dataset_name, preprocess)

    train_feats, train_labels = extract_image_features(
        train_set, model, device,
        batch_size=FEATURE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    test_feats, test_labels = extract_image_features(
        test_set, model, device,
        batch_size=FEATURE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    train_feats = train_feats.float()
    test_feats = test_feats.float()
    # Save to cache
    train_data = {
        "features": train_feats,
        "labels": train_labels,
        "class_names": class_names,
    }
    test_data = {
        "features": test_feats,
        "labels": test_labels,
        "class_names": class_names,
    }
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    print(f"[data_utils] Saved features to '{train_path}' and '{test_path}'.")

    train_feats = train_data["features"].float()
    test_feats = test_data["features"].float()
    result = {
        "train_features": train_feats,
        "train_labels": train_labels,
        "test_features": test_feats,
        "test_labels": test_labels,
        "class_names": class_names,
    }
    return result
