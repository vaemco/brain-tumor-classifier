
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models
from copy import deepcopy
import matplotlib.pyplot as plt

# M2 MacBook: Use MPS if available, else CPU
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
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data" / "Brain_Tumor_Dataset" / "Training"
external_data_dir = base_dir / "data" / "Brain_Tumor_Dataset" / "external_dataset" / "training"

print(f"Main Data Dir: {data_dir}")
print(f"External Data Dir: {external_data_dir}")

# Custom Noise Transform
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Enhanced Transforms for Training (Realism Simulation)
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.90, 1.10)),
    transforms.RandomHorizontalFlip(p=0.5),
    # Rotation & Affine
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    # Blur
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    # Color/Contrast Jitter
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    # Noise
    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

train_loader = DataLoader(full_train, batch_size=32, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(full_val, batch_size=32, sampler=val_sampler, num_workers=0)

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
    {"params": [p for n,p in model.named_parameters() if p.requires_grad and (n.startswith("layer3") or n.startswith("layer4"))], "lr": 3e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
]
optimizer = optim.Adam(params, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print("Model ready for training on", device)

# Training Loop
epochs = 30
patience = 5
best_val = float("inf")
bad = 0
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

output_dir = base_dir / "runs"
model_dir = base_dir / "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

metrics_file = os.path.join(output_dir, "metrics_v2.json")
model_save_path = os.path.join(model_dir, "brain_tumor_resnet18_v2.pt")

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
    
    print(f"Epoch {epoch+1:02d} | Train {train_loss:.4f}, Acc {train_acc:.2f}% | Val {val_loss:.4f}, Acc {val_acc:.2f}%")
    
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
