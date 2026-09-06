"""
Training Script for Brain Tumor Classifier (ResNet18).

Transfer learning with fine-tuning, validation early stopping, and metric tracking.
"""

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from torchvision import datasets
import yaml

# Allow importing shared utilities from src/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor.device import get_device  # noqa: E402
from brain_tumor.model_factory import create_model  # noqa: E402
from brain_tumor.paths import get_data_dirs, get_models_dir, get_runs_dir, project_root  # noqa: E402
from brain_tumor.transforms import build_train_transforms, build_val_transforms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def train_model(
    data_dir: Path,
    external_data_dir: Path | None = None,
    output_dir: Path | None = None,
    models_dir: Path | None = None,
    model_name: str = "resnet18",
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 5,
    layer_lr: float = 3e-4,
    fc_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Train a brain tumor classification model with transfer learning and early stopping.
    """
    root = project_root()
    output_dir = output_dir or get_runs_dir(root)
    models_dir = models_dir or get_models_dir(root)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics_v2.json"
    model_save_path = models_dir / f"brain_tumor_{model_name}_v2_trained.pt"

    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")

    train_tf = build_train_transforms()
    val_tf = build_val_transforms()

    logger.info(f"Loading primary dataset from {data_dir}...")
    dataset_primary_train = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    dataset_primary_val = datasets.ImageFolder(root=str(data_dir), transform=val_tf)

    train_datasets = [dataset_primary_train]
    val_datasets = [dataset_primary_val]

    if external_data_dir and external_data_dir.exists():
        try:
            ext_train = datasets.ImageFolder(root=str(external_data_dir), transform=train_tf)
            ext_val = datasets.ImageFolder(root=str(external_data_dir), transform=val_tf)
            train_datasets.append(ext_train)
            val_datasets.append(ext_val)
            logger.info(f"Loaded external dataset from {external_data_dir} ({len(ext_train)} samples)")
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(f"Could not load external dataset from {external_data_dir}: {exc}")

    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)

    total_samples = len(full_train_dataset)
    class_names = dataset_primary_train.classes
    num_classes = len(class_names)
    logger.info(f"Classes ({num_classes}): {class_names}")
    logger.info(f"Total dataset samples: {total_samples}")

    rng = np.random.default_rng(seed)
    indices = np.arange(total_samples)
    rng.shuffle(indices)

    split_idx = int(np.floor(val_split * total_samples))
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        full_val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
    )

    logger.info(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=True)

    # Freeze earlier layers, fine-tune higher representation layers and head
    for param in model.parameters():
        param.requires_grad = False

    fine_tune_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []

    if model_name == "resnet18":
        for name, param in model.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4"):
                param.requires_grad = True
                fine_tune_params.append(param)
            elif name.startswith("fc"):
                param.requires_grad = True
                head_params.append(param)
    else:
        # Generic fine-tuning for other architectures
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                param.requires_grad = True
                fine_tune_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": fine_tune_params, "lr": layer_lr},
            {"params": head_params, "lr": fc_lr},
        ],
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_len = images.size(0)
            train_loss_sum += loss.item() * batch_len
            train_correct += int((outputs.argmax(1) == labels).sum().item())
            train_total += batch_len

        epoch_train_loss = train_loss_sum / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_len = images.size(0)
                val_loss_sum += loss.item() * batch_len
                val_correct += int((outputs.argmax(1) == labels).sum().item())
                val_total += batch_len

        epoch_val_loss = val_loss_sum / val_total
        epoch_val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        logger.info(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_file}")

    return history


def main() -> None:
    root = project_root()
    default_data_dir, default_ext_dir = get_data_dirs(root)
    config_path = root / "configs" / "train.yaml"
    config = load_config(config_path)

    parser = argparse.ArgumentParser(description="Train brain tumor classifier.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.get("model_name", "resnet18"),
        choices=["resnet18", "efficientnet", "densenet"],
        help="Architecture name",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(config.get("data_dir") or default_data_dir),
        help="Primary training dataset directory",
    )
    parser.add_argument(
        "--external-data-dir",
        type=Path,
        default=Path(config.get("external_data_dir") or default_ext_dir),
        help="Optional external training dataset directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(config.get("batch_size", 32)),
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config.get("epochs", 30)),
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(config.get("patience", 5)),
        help="Early stopping patience",
    )
    parser.add_argument(
        "--layer-lr",
        type=float,
        default=float(config.get("layer_lr", 3e-4)),
        help="Learning rate for fine-tuned layers",
    )
    parser.add_argument(
        "--fc-lr",
        type=float,
        default=float(config.get("fc_lr", 1e-3)),
        help="Learning rate for classification head",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(config.get("weight_decay", 1e-4)),
        help="Weight decay parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split reproducibility",
    )
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        external_data_dir=args.external_data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        layer_lr=args.layer_lr,
        fc_lr=args.fc_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
