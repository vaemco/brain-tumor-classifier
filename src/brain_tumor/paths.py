"""
Path helpers used across scripts, notebooks, and the web app.
"""

from pathlib import Path
from typing import Optional, Tuple


def project_root() -> Path:
    """
    Returns the repository root assuming this file lives in src/brain_tumor/.
    """
    return Path(__file__).resolve().parents[2]


def get_data_dirs(root: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Returns (primary_training_dir, external_training_dir).
    """
    root = root or project_root()
    data_root = root / "data" / "Brain_Tumor_Dataset"
    primary = data_root / "Training"
    external = data_root / "external_dataset" / "training"
    return primary, external


def get_models_dir(root: Optional[Path] = None) -> Path:
    root = root or project_root()
    return root / "models"


def get_runs_dir(root: Optional[Path] = None) -> Path:
    root = root or project_root()
    return root / "runs"
