"""
Centralized model creation and target layer resolution.

Ensures consistent architectures across training, evaluation, and inference.
"""

from typing import Any
import torch
from torch import nn
from torchvision import models


def build_classifier_head(
    in_features: int,
    num_classes: int = 4,
    dropout_rate: float = 0.5,
    hidden_dim: int = 256,
) -> nn.Sequential:
    """
    Standard classifier head used across all model architectures:
    Dropout -> Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear
    """
    return nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_dim, num_classes),
    )


def normalize_model_name(model_name: str) -> str:
    """
    Standardize model name strings to canonical identifiers.
    """
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized in ("resnet", "resnet18"):
        return "resnet18"
    if normalized in ("efficientnet", "efficientnet_b0"):
        return "efficientnet"
    if normalized in ("densenet", "densenet121"):
        return "densenet"
    return normalized


def create_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = False,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    Create a neural network model with the standardized classifier head.

    Args:
        model_name: 'resnet18', 'efficientnet', or 'densenet'
        num_classes: Number of classification targets (default: 4)
        pretrained: If True, load ImageNet pretrained backbone weights
        dropout_rate: Dropout probability in the classifier head (default: 0.5)

    Returns:
        Configured PyTorch nn.Module
    """
    canonical_name = normalize_model_name(model_name)

    if canonical_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = build_classifier_head(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        return model

    if canonical_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = build_classifier_head(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        return model

    if canonical_name == "densenet":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = build_classifier_head(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
        return model

    raise ValueError(
        f"Unsupported model architecture: {model_name}. Supported: resnet18, efficientnet, densenet"
    )


def get_target_layer(model: nn.Module, model_name: str) -> list[Any]:
    """
    Return the target layer list for GradCAM visualization for a given model.
    """
    canonical_name = normalize_model_name(model_name)
    if canonical_name == "resnet18":
        return [model.layer4[-1]]  # type: ignore[union-attr]
    if canonical_name == "efficientnet":
        return [model.features[-1]]  # type: ignore[union-attr]
    if canonical_name == "densenet":
        return [model.features[-1]]  # type: ignore[union-attr]

    raise ValueError(f"Cannot resolve target layer for unknown model: {model_name}")
