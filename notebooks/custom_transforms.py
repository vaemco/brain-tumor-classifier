
"""
Notebook-friendly import for custom transforms.
Re-exports the shared AddGaussianNoise used across the project.
"""

import sys
from pathlib import Path

# Allow running notebooks without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brain_tumor.transforms import AddGaussianNoise

__all__ = ["AddGaussianNoise"]
