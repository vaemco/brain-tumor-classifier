"""
Shared helpers for the brain tumor classifier project.

This package centralizes reusable utilities (paths, transforms) so scripts,
the web app, and notebooks can import a single source of truth.
"""

from .paths import get_data_dirs, get_models_dir, get_runs_dir, project_root
from .transforms import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    AddGaussianNoise,
    build_train_transforms,
    build_val_transforms,
)

__all__ = [
    "project_root",
    "get_data_dirs",
    "get_models_dir",
    "get_runs_dir",
    "DEFAULT_IMAGE_MEAN",
    "DEFAULT_IMAGE_STD",
    "AddGaussianNoise",
    "build_train_transforms",
    "build_val_transforms",
]
