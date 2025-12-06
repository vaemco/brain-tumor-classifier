
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import numpy as np
from pathlib import Path
import os

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

BASE_DIR = Path.cwd().resolve()
while not (BASE_DIR / "data").exists() and BASE_DIR.parent != BASE_DIR:
    BASE_DIR = BASE_DIR.parent

data_dir = BASE_DIR / "data" / "Brain_Tumor_Dataset" / "Training"
external_data_dir = BASE_DIR / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "training"

# Transforms (Strong Augmentation: Mag=9)
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load Datasets
print("Loading datasets...")
dataset1_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
dataset1_val = datasets.ImageFolder(root=data_dir, transform=val_tf)

try:
    dataset2_train = datasets.ImageFolder(root=external_data_dir, transform=train_tf)
    dataset2_val = datasets.ImageFolder(root=external_data_dir, transform=val_tf)
    full_train = ConcatDataset([dataset1_train, dataset2_train])
    full_val = ConcatDataset([dataset1_val, dataset2_val])
    print(f"Merged original and external datasets.")
except Exception as e:
    print(f"External dataset not found: {e}")
    full_train = dataset1_train
    full_val = dataset1_val

class_names = dataset1_train.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")

if __name__ == "__main__":
    # Data Loaders
    indices = list(range(len(full_train)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(full_train)))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(full_train, batch_size=32, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(full_val, batch_size=32, sampler=val_sampler, num_workers=0)

    # Model Setup (DenseNet121 - Unfrozen Diff LR)
    print("Setting up DenseNet121...")
    base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # 1. Unfreeze Features
    for param in base.features.parameters():
        param.requires_grad = True

    # 2. Classifier Head
    in_features = base.classifier.in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    model = base.to(device)

    # 3. Optimizer with Differential Learning Rate
    optimizer = optim.AdamW([
        {'params': base.features.parameters(), 'lr': 1e-5}, # Backbone: Low LR
        {'params': base.classifier.parameters(), 'lr': 1e-3} # Head: High LR
    ], weight_decay=5e-2)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Short Training Loop (3 Epochs)
    print("Starting verification training (3 epochs)...")
    for epoch in range(3):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            tl += loss.item() * x.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)

            if i % 10 == 0:
                print(f"  Batch {i}/{len(train_loader)} Loss {loss.item():.4f}")

        train_loss = tl / tt
        train_acc = 100 * tc / tt

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

        print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

    print("Verification complete.")
