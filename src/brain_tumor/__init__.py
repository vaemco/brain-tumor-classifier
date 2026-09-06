"""
Shared helpers for the brain tumor classifier project.

This package centralizes reusable utilities (paths, transforms, devices, models)
so scripts, the web app, and notebooks can import a single source of truth.
"""

from .device import get_device
from .model_factory import (
    build_classifier_head,
    create_model,
    get_target_layer,
    normalize_model_name,
)
from .paths import get_data_dirs, get_models_dir, get_runs_dir, project_root
from .transforms import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    AddGaussianNoise,
    CenterBiasedCrop,
    build_train_transforms,
    build_val_transforms,
    mixup_data,
    mixup_criterion,
)

__all__ = [
    "project_root",
    "get_data_dirs",
    "get_models_dir",
    "get_runs_dir",
    "get_device",
    "create_model",
    "build_classifier_head",
    "get_target_layer",
    "normalize_model_name",
    "DEFAULT_IMAGE_MEAN",
    "DEFAULT_IMAGE_STD",
    "AddGaussianNoise",
    "CenterBiasedCrop",
    "build_train_transforms",
    "build_val_transforms",
    "mixup_data",
    "mixup_criterion",
]
