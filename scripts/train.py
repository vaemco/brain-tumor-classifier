"""
Training Script for Brain Tumor Classifier (ResNet18)
=====================================================

This script handles the training of the PyTorch model.
It uses Transfer Learning with a pre-trained ResNet18 architecture.

Key Features:
- Data Augmentation: Random rotations, flips, blur, and noise to prevent overfitting.
- Transfer Learning: Freezes early layers and only trains the later layers (Fine-Tuning).
- Early Stopping: Stops training if validation loss doesn't improve for `patience` epochs.
- Metrics: Saves loss and accuracy history to `runs/metrics_v2.json`.

How to Modify:
- Hyperparameters: Adjust `epochs`, `batch_size` (in DataLoader), or `lr` (learning rate) in the Optimizer section.
- Model: Change `models.resnet18` to another architecture (e.g., `models.efficientnet_b0`) if needed.
- Augmentation: Modify `train_tf` to add/remove transformations.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, random_split
import yaml
from torchvision import datasets, models

# Allow importing shared utilities from src/
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brain_tumor.paths import get_data_dirs  # noqa: E402
from brain_tumor.transforms import (  # noqa: E402
    build_train_transforms,
    build_val_transforms,
)

CONFIG_PATH = BASE_DIR / "configs" / "train.yaml"


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

# --- Device Configuration ---
# Automatically detects MPS (Mac), CUDA (NVIDIA), or CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

print(f"PyTorch version: {torch.__version__}")

# Data Paths
# Define paths relative to the script
base_dir = BASE_DIR
config = load_config(CONFIG_PATH)
data_dir_default, external_data_dir_default = get_data_dirs(base_dir)
data_dir = Path(config.get("data_dir") or data_dir_default)
external_data_dir = Path(config.get("external_data_dir") or external_data_dir_default)
batch_size = int(config.get("batch_size", 32))
epochs = int(config.get("epochs", 30))
patience = int(config.get("patience", 5))
layer_lr = float(config.get("layer_lr", 3e-4))
fc_lr = float(config.get("fc_lr", 1e-3))
weight_decay = float(config.get("weight_decay", 1e-4))

print(f"Main Data Dir: {data_dir}")
print(f"External Data Dir: {external_data_dir}")


# --- Data Augmentation (Crucial for Generalization) ---
# These transformations are applied to every training image on the fly.
# They help the model learn to recognize tumors even if the image is rotated, blurry, or has different lighting.
train_tf = build_train_transforms()

# Validation transforms: No augmentation, just resizing and normalization.
# We want to evaluate on clean, standard images.
val_tf = build_val_transforms()

# Load Datasets
print("Loading datasets...")
dataset1 = datasets.ImageFolder(root=data_dir, transform=train_tf)
print(f"Original dataset: {len(dataset1)} samples")

try:
    dataset2 = datasets.ImageFolder(root=external_data_dir, transform=train_tf)
    print(f"External dataset: {len(dataset2)} samples")
    full_dataset = ConcatDataset([dataset1, dataset2])
except Exception as e:
    print(f"Could not load external dataset: {e}")
    full_dataset = dataset1

class_names = dataset1.classes
print("Classes:", class_names)
print("Total training samples:", len(full_dataset))

# Split 80/20
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override validation transform (hacky but works for random_split subsets if we iterate)
# Since random_split returns Subset, we can't easily set transform per subset.
# Better approach: Apply transforms in the loop or use a custom wrapper.
# For simplicity here, we'll assume the transform is applied at access time.
# But wait, ImageFolder applies transform at __getitem__.
# So train_dataset has train_tf. val_dataset ALSO has train_tf.
# We need to fix this.


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            # We need to get the original PIL image, not the transformed tensor
            # This is tricky with ImageFolder + random_split
            # Re-loading the image is slow.
            # Alternative: Load dataset twice.
            pass
        return x, y

    def __len__(self):
        return len(self.subset)


# Better approach: Load dataset twice
dataset1_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
dataset1_val = datasets.ImageFolder(root=data_dir, transform=val_tf)

try:
    dataset2_train = datasets.ImageFolder(root=external_data_dir, transform=train_tf)
    dataset2_val = datasets.ImageFolder(root=external_data_dir, transform=val_tf)

    full_train = ConcatDataset([dataset1_train, dataset2_train])
    full_val = ConcatDataset([dataset1_val, dataset2_val])
except:
    full_train = dataset1_train
    full_val = dataset1_val

# Create indices
indices = list(range(len(full_train)))
np.random.shuffle(indices)
split = int(np.floor(0.2 * len(full_train)))
train_idx, val_idx = indices[split:], indices[:split]

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

train_loader = DataLoader(
    full_train, batch_size=batch_size, sampler=train_sampler, num_workers=0
)
val_loader = DataLoader(
    full_val, batch_size=batch_size, sampler=val_sampler, num_workers=0
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# Model Setup
num_classes = len(class_names)
base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
base.fc = nn.Linear(base.fc.in_features, num_classes)

# Freeze/Unfreeze
for p in base.parameters():
    p.requires_grad = False
for name, p in base.named_parameters():
    if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
        p.requires_grad = True

model = base.to(device)

# Optimizer
params = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and (n.startswith("layer3") or n.startswith("layer4"))
        ],
        "lr": layer_lr,
    },
    {"params": model.fc.parameters(), "lr": fc_lr},
]
optimizer = optim.Adam(params, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

print("Model ready for training on", device)

# Training Loop
best_val = float("inf")
bad = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print(
    f"Hyperparameters -> epochs: {epochs}, batch_size: {batch_size}, "
    f"layer_lr: {layer_lr}, fc_lr: {fc_lr}, weight_decay: {weight_decay}, patience: {patience}"
)

output_dir = base_dir / "runs"
model_dir = base_dir / "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

metrics_file = os.path.join(output_dir, "metrics_v2.json")
model_save_path = os.path.join(model_dir, "brain_tumor_resnet18_v2_trained.pt")

print("Starting training...")

for epoch in range(epochs):
    # --- Training ---
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        tl += loss.item() * x.size(0)
        tc += (out.argmax(1) == y).sum().item()
        tt += y.size(0)

    train_loss = tl / tt
    train_acc = 100 * tc / tt

    # --- Validation ---
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            vl += loss.item() * x.size(0)
            vc += (out.argmax(1) == y).sum().item()
            vt += y.size(0)

    val_loss = vl / vt
    val_acc = 100 * vc / vt

    # Save history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d} | Train {train_loss:.4f}, Acc {train_acc:.2f}% | Val {val_loss:.4f}, Acc {val_acc:.2f}%"
    )

    # Early Stopping
    if val_loss < best_val:
        best_val = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, model_save_path)
        print(f"  → Model saved to {model_save_path}")
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

print("\nTraining complete!")
