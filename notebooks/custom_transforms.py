
"""
Notebook-friendly import for custom transforms.
Re-exports the shared transforms used across the project.

Anti-Clever-Hans transforms to prevent learning from edge/skull artifacts.
"""

import sys
from pathlib import Path

# Allow running notebooks without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brain_tumor.transforms import (
    AddGaussianNoise,
    CenterBiasedCrop,
    build_train_transforms,
    build_val_transforms,
    mixup_criterion,
    mixup_data,
)

__all__ = [
    "AddGaussianNoise",
    "CenterBiasedCrop",
    "build_train_transforms",
    "build_val_transforms",
    "mixup_data",
    "mixup_criterion",
]
