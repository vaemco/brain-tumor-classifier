"""
K-Fold Cross-Validation Training Script.

Combines available dataset splits and trains K folds with stratified sampling,
cosine annealing learning rate scheduling, mixed precision, and early stopping.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow importing shared utilities from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor.device import get_device  # noqa: E402
from brain_tumor.model_factory import create_model  # noqa: E402
from brain_tumor.paths import get_models_dir, get_runs_dir, project_root  # noqa: E402
from brain_tumor.transforms import build_train_transforms, build_val_transforms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def get_worker_count() -> int:
    """Return a safe number of dataloader workers based on system cores."""
    cpu_count = os.cpu_count() or 0
    return min(4, cpu_count)


def collect_all_images(data_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Collect image paths and labels across all specified directories."""
    all_images: list[str] = []
    all_labels: list[int] = []
    label_map = {cls.lower(): i for i, cls in enumerate(CLASSES)}

    for data_dir in data_dirs:
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            continue

        for class_dir in data_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.lower()
            if class_name not in label_map:
                continue

            label = label_map[class_name]
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    all_images.append(str(img_path))
                    all_labels.append(label)

    logger.info(f"Total dataset images collected: {len(all_images)}")
    for i, cls in enumerate(CLASSES):
        count = sum(1 for l in all_labels if l == i)
        logger.info(f"Class '{cls}': {count} images")

    return np.array(all_images), np.array(all_labels)


class ImagePathDataset(Dataset):
    """Loads images on-demand from absolute or relative file paths."""

    def __init__(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        transform: nn.Module | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model on evaluation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def train_fold(
    fold: int,
    model_name: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = 5,
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    """Train a single fold with cosine annealing and early stopping."""
    logger.info(f"Starting Fold {fold + 1}: Train={len(train_images)}, Val={len(val_images)}")

    train_transform = build_train_transforms(anti_clever_hans=True)
    val_transform = build_val_transforms()

    train_dataset = ImagePathDataset(train_images, train_labels, train_transform)
    val_dataset = ImagePathDataset(val_images, val_labels, val_transform)

    workers = get_worker_count()
    use_pin = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_pin,
    )

    model = create_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Fold {fold + 1} | Epoch {epoch + 1:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
            logger.info(f"Fold {fold + 1}: New best validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Fold {fold + 1}: Early stopping after {epoch + 1} epochs.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_acc, history


def main() -> None:
    root = project_root()
    models_dir = get_models_dir(root)
    runs_dir = get_runs_dir(root)

    data_dirs = [
        root / "data" / "Brain_Tumor_Dataset" / "Training",
        root / "data" / "Brain_Tumor_Dataset" / "Testing",
        root / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "training",
        root / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "testing",
    ]

    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Training")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet", "densenet"],
        help="Model architecture",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of epochs per fold (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience per fold",
    )
    args = parser.parse_args()

    logger.info("Starting K-Fold Cross-Validation Training")
    logger.info(f"Model: {args.model} | Folds: {args.folds} | Max Epochs: {args.epochs}")

    device = get_device()
    logger.info(f"Using device: {device}")

    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    images, labels = collect_all_images(data_dirs)
    if len(images) == 0:
        logger.error("No images found in data directories.")
        return

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results: list[float] = []
    all_histories: list[dict[str, list[float]]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        train_images = images[train_idx]
        train_labels = labels[train_idx]
        val_images = images[val_idx]
        val_labels = labels[val_idx]

        model, best_acc, history = train_fold(
            fold=fold,
            model_name=args.model,
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
        )

        model_path = models_dir / f"brain_tumor_{args.model}_fold{fold + 1}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved fold model to: {model_path}")

        fold_results.append(best_acc)
        all_histories.append(history)

    mean_acc = float(np.mean(fold_results))
    std_acc = float(np.std(fold_results))
    logger.info(f"K-Fold Results: Mean Accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

    results = {
        "model": args.model,
        "folds": args.folds,
        "epochs_per_fold": args.epochs,
        "fold_accuracies": fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "timestamp": datetime.now().isoformat(),
        "histories": all_histories,
    }

    results_path = runs_dir / f"kfold_{args.model}_{args.folds}fold.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"K-Fold training results saved to: {results_path}")


if __name__ == "__main__":
    main()
