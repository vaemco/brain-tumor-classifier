"""
Data Preparation Script.

Splits a raw dataset into Training and Testing sets with reproducible shuffling.
"""

import argparse
import logging
from pathlib import Path
import random
import shutil

from brain_tumor.paths import project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def prepare_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.8,
    classes: list[str] | None = None,
    seed: int = 42,
) -> None:
    """
    Split images in source_dir into training and testing directories.

    Args:
        source_dir: Directory containing class subfolders of raw images
        target_dir: Destination directory where training/ and testing/ folders are created
        train_ratio: Proportion of images assigned to training (e.g. 0.8)
        classes: List of class directory names to process
        seed: Random seed for reproducible splitting
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    rng = random.Random(seed)

    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return

    for split in ("training", "testing"):
        for cls in classes:
            (target_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        class_path = source_dir / cls
        if not class_path.exists():
            logger.warning(f"Class folder not found: {class_path}")
            continue

        images = [
            f.name
            for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

        rng.shuffle(images)

        split_index = int(len(images) * train_ratio)
        train_imgs = images[:split_index]
        test_imgs = images[split_index:]

        for img in train_imgs:
            shutil.copy2(class_path / img, target_dir / "training" / cls / img)

        for img in test_imgs:
            shutil.copy2(class_path / img, target_dir / "testing" / cls / img)

        logger.info(f"{cls}: {len(train_imgs)} train / {len(test_imgs)} test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split raw brain tumor dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root() / "data" / "Brain_Tumor_Dataset" / "external_data",
        help="Source directory with class subfolders",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=project_root() / "data" / "Brain_Tumor_Dataset" / "external_dataset",
        help="Destination directory for splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        target_dir=args.target,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
