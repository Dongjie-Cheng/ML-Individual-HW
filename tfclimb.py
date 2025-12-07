"""
tfclimb.py

Implementation of TF-CLIMB and related baselines on top of precomputed CLIP features.

Main functionalities:
    - build_text_features: encode class names into CLIP text embeddings (and return logit_scale)
    - build_tfclimb_stats: compute class prototypes and empirical priors from support set
    - predict_zero_shot: CLIP zero-shot classifier
    - predict_prototypes_only: prototype-based classifier
    - predict_fused: fusion of text and prototype logits (no adjustment)
    - predict_tfclimb: full TF-CLIMB (fusion + logit adjustment)
"""

from typing import List, Tuple

import torch
import clip

from config import TFCLIMB_CFG, CLIP_MODEL_NAME
from data_utils import get_device


# ---------------------------------------------------------------------------
# Cached CLIP text model (to avoid repeated loading)
# ---------------------------------------------------------------------------

_CLIP_TEXT_MODEL = None
_CLIP_TEXT_DEVICE = None


def _get_clip_text_model():
    """
    Load CLIP model (if not already loaded) and return (model, device).

    We only use the text encoder and logit_scale here, but for simplicity
    we keep the full model.
    """
    global _CLIP_TEXT_MODEL, _CLIP_TEXT_DEVICE
    if _CLIP_TEXT_MODEL is None:
        device = get_device()
        print(f"[tfclimb] Loading CLIP text model '{CLIP_MODEL_NAME}' on {device}...")
        model, _ = clip.load(CLIP_MODEL_NAME, device=device)
        model.eval()
        _CLIP_TEXT_MODEL = model
        _CLIP_TEXT_DEVICE = device
    return _CLIP_TEXT_MODEL, _CLIP_TEXT_DEVICE


# ---------------------------------------------------------------------------
# Text features (prompts -> CLIP text embeddings)
# ---------------------------------------------------------------------------

def build_prompts(class_names: List[str]) -> List[str]:
    """
    Build simple text prompts given class names.

    You can customize this function if you want more elaborate prompts.

    Args:
        class_names: list of class name strings, length C

    Returns:
        prompts: list of prompt strings, length C
    """
    prompts = [f"a photo of a {name}" for name in class_names]
    return prompts


def build_text_features(class_names: List[str]) -> Tuple[torch.Tensor, float]:
    """
    Encode class names into CLIP text embeddings and return logit_scale.

    This uses a cached CLIP model to avoid re-loading for each dataset.

    Args:
        class_names: list of class names, length C

    Returns:
        text_features: [C, d] tensor (on CPU), normalized row-wise
        logit_scale:  float, exp(model.logit_scale) from CLIP
    """
    model, device = _get_clip_text_model()

    print(f"[tfclimb] Building text features for {len(class_names)} classes.")

    prompts = build_prompts(class_names)
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)  # [C, L]
        text_feats = model.encode_text(tokens)      # [C, d]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.float()
        # CLIP stores logit_scale as a learnable parameter; official zero-shot
        # logits are: logit_scale * (img @ text.T)
        logit_scale = model.logit_scale.exp().item()

    # Move to CPU for later matrix multiplications with CPU features
    return text_feats.cpu(), float(logit_scale)


# ---------------------------------------------------------------------------
# TF-CLIMB statistics: prototypes and empirical priors
# ---------------------------------------------------------------------------

def build_tfclimb_stats(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    support_indices: List[int],
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute prototypes, empirical priors, and raw counts for TF-CLIMB.

    Args:
        train_features: [N_train, d] tensor of CLIP image features (normalized)
        train_labels:   [N_train] tensor of integer labels
        support_indices: list of indices into train_features/labels used as support set
        num_classes:    total number of classes C

    Returns:
        prototypes: [C, d] tensor of normalized prototypes (zero vector if class has no support)
        pi:         [C] tensor of empirical class priors (counts / total)
        counts:     [C] tensor of integer counts per class
    """
    feats = train_features.cpu()
    labels = train_labels.cpu()

    s_idx = torch.tensor(support_indices, dtype=torch.long)
    s_feats = feats[s_idx]      # [Ns, d]
    s_labels = labels[s_idx]    # [Ns]

    d = s_feats.shape[1]
    prototypes = torch.zeros(num_classes, d, dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.long)

    # Accumulate sum of features per class
    for feat, y in zip(s_feats, s_labels):
        y = int(y)
        prototypes[y] += feat
        counts[y] += 1

    # Normalize to get prototypes
    for c in range(num_classes):
        if counts[c] > 0:
            prototypes[c] /= counts[c].float()
            norm = prototypes[c].norm() + 1e-8
            prototypes[c] /= norm
        else:
            # No support for this class: leave as zero vector
            prototypes[c] = torch.zeros(d)

    total = counts.sum().item()
    C = num_classes
    if total > 0:
        empirical = counts.float() / float(total)                 # 经验先验
        uniform = torch.full_like(empirical, 1.0 / C)             # 均匀先验
        beta = TFCLIMB_CFG.prior_beta
        pi = beta * empirical + (1.0 - beta) * uniform            # ★ 平滑后的先验
    else:
        # Fallback: uniform if somehow there is no support (should not happen)
        pi = torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)

    return prototypes, pi, counts

# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------

def predict_zero_shot(
    test_features: torch.Tensor,
    text_features: torch.Tensor,
    tau_text: float,
) -> torch.Tensor:
    """
    CLIP zero-shot predictions using only text logits.

    Args:
        test_features: [N_test, d] tensor (normalized) on CPU
        text_features: [C, d] tensor (normalized) on CPU
        tau_text: scalar temperature (use CLIP logit_scale for official behavior)

    Returns:
        pred: [N_test] tensor of predicted class indices
    """
    test_features = test_features.float()
    text_features = text_features.float()
    logits = tau_text * (test_features @ text_features.T)  # [N, C]
    pred = logits.argmax(dim=1)
    return pred


def predict_prototypes_only(
    test_features: torch.Tensor,
    prototypes: torch.Tensor,
    tau_img: float = TFCLIMB_CFG.tau_img,
) -> torch.Tensor:
    """
    Prototype-only classifier (no text, no logit adjustment).

    Args:
        test_features: [N_test, d] tensor (normalized)
        prototypes:    [C, d] tensor (normalized, some rows may be zero)
        tau_img:       scalar temperature for image logits

    Returns:
        pred: [N_test] tensor of predicted class indices
    """
    logits = tau_img * (test_features @ prototypes.T)  # [N, C]
    pred = logits.argmax(dim=1)
    return pred


def predict_fused(
    test_features: torch.Tensor,
    text_features: torch.Tensor,
    prototypes: torch.Tensor,
    tau_text: float,
    tau_img: float = TFCLIMB_CFG.tau_img,
    alpha: float = TFCLIMB_CFG.alpha,
) -> torch.Tensor:
    """
    Fused classifier: (1 - alpha) * text_logits + alpha * proto_logits, no adjustment.

    Args:
        test_features: [N_test, d]
        text_features: [C, d]
        prototypes:    [C, d]
        tau_text:      temperature for text logits (usually CLIP logit_scale)
        tau_img:       temperature for proto logits
        alpha:         fusion weight in [0, 1]

    Returns:
        pred: [N_test] tensor of predicted class indices
    """
    test_features = test_features.float()
    text_features = text_features.float()
    text_logits = tau_text * (test_features @ text_features.T)    # [N, C]
    proto_logits = tau_img * (test_features @ prototypes.T)       # [N, C]
    fused_logits = (1.0 - alpha) * text_logits + alpha * proto_logits
    pred = fused_logits.argmax(dim=1)
    return pred


def predict_tfclimb(
    test_features: torch.Tensor,
    text_features: torch.Tensor,
    prototypes: torch.Tensor,
    pi: torch.Tensor,
    tau_text: float,
    tau_img: float = TFCLIMB_CFG.tau_img,
    alpha: float = TFCLIMB_CFG.alpha,
    lam: float = TFCLIMB_CFG.lam,
    eps: float = TFCLIMB_CFG.eps,
) -> torch.Tensor:
    """
    Full TF-CLIMB prediction: fusion + class-frequency-based logit adjustment.

    Args:
        test_features: [N_test, d]
        text_features: [C, d]
        prototypes:    [C, d]
        pi:            [C] empirical class priors from support set
        tau_text:      temperature for text logits (usually CLIP logit_scale)
        tau_img:       temperature for proto logits
        alpha:         fusion weight
        lam:           logit adjustment strength lambda
        eps:           small constant for numerical stability

    Returns:
        pred: [N_test] tensor of predicted class indices
    """
    test_features = test_features.float()
    text_features = text_features.float()
    # base logits
    text_logits = tau_text * (test_features @ text_features.T)    # [N, C]
    proto_logits = tau_img * (test_features @ prototypes.T)       # [N, C]
    fused_logits = (1.0 - alpha) * text_logits + alpha * proto_logits  # [N, C]

    # logit adjustment: b_c = lam * log(1 / (pi_c + eps))
    bias = lam * torch.log(1.0 / (pi + eps))   # [C]

    final_logits = fused_logits + bias.unsqueeze(0)  # [N, C]
    pred = final_logits.argmax(dim=1)
    return pred
