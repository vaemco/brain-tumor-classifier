"""
Dataset transforms for brain tumor classification.

Uses the unified transforms defined in src/brain_tumor/transforms.py to ensure
strict consistency between training and inference.
"""

from pathlib import Path
import sys

# Add src to path for shared transforms if not installed
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchvision import transforms

from brain_tumor.transforms import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    build_val_transforms,
)

MEAN = DEFAULT_IMAGE_MEAN
STD = DEFAULT_IMAGE_STD

# Validation and inference transform matching training pipeline exactly
val_tf = build_val_transforms(image_size=224)

# Legacy transform preserved for backwards compatibility reference
val_tf_legacy = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)
